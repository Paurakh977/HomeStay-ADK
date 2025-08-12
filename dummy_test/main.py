import os
import logging
from typing import Tuple, Optional, Dict, Any
import json
import base64
import warnings
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv

from google.genai.types import (
    Part,
    Content,
    Blob,
)
from google.genai import types as genai_types
from google.adk.runners import Runner
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load environment variables
load_dotenv()

# Constants
APP_NAME = "ADK Streaming Robust Example"
STATIC_DIR = Path("static")
MAX_MESSAGE_LENGTH = 10000
SUPPORTED_AUDIO_TYPES = ["audio/webm", "audio/wav", "audio/mp3"]
SUPPORTED_TEXT_TYPES = ["text/plain"]

# Pydantic models for request validation
class MessageRequest(BaseModel):
    mime_type: str = Field(..., description="MIME type of the message")
    data: str = Field(..., description="Message data (text or base64 encoded)")

class AgentResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    response_text: Optional[str] = None
    author: Optional[str] = None
    grounding_metadata: Optional[Dict[str, Any]] = None
    usage_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Create root agent with enhanced configuration
root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='homestay_search_agent',
    description='A bilingual homestay search assistant for Nepal with voice message support.',
    instruction='''You are an intelligent homestay search assistant specialized in helping users find homestays in Nepal. You can understand and respond in both English and Nepali, and work seamlessly with voice messages and text queries.

## Core Capabilities

### Language Understanding
- **Bilingual Processing**: Fluent in both English and Nepali
- **Voice Message Support**: Can process Nepali voice messages converted to text
- **Mixed Language Queries**: Handle queries that mix English and Nepali terms
- **Location Recognition**: Understand Nepali place names, municipalities, districts, and provinces

### Location Understanding
Apply intelligent matching for location queries:
- Handle spelling variations (e.g., "Malangawa" → "Malangwa Municipality")  
- Process partial names (e.g., "काठमाडौं" → "Kathmandu Metropolitan City")
- Understand location hierarchies (district within province relationships)

### Feature Recognition
Map user terms to database values:

**Nepali → Database Mapping:**
- "ट्रेकिङ" → Trekking routes
- "माछा मार्ने" → Fishing facilities  
- "सफा खानेपानी" → Clean drinking water
- "इन्टरनेट" → Communication facilities
- "शौचालय" → Toilet/bathroom facilities
- "सफारी" → Wildlife safari activities

### Query Processing
- **Must-have indicators**: "चाहिएको छ", "आवश्यक छ", "need", "required", "must have"
- **Optional indicators**: "यदि सम्भव भए", "हुन्छ भने राम्रो", "if possible", "would be nice"

### Search Strategy
1. **Intent Recognition**: Identify what the user is looking for
2. **Feature Extraction**: Extract specific amenities, activities, or services mentioned  
3. **Location Parsing**: Identify and validate location references
4. **Always use natural_language_description parameter** when calling search tools

### Response Guidelines
- **Nepali Query → Nepali Response** (with English homestay details)
- **English Query → English Response**
- **Mixed Query → Match user's primary language**
- Provide search results, location context, feature summary, and alternatives when needed

Remember: Help users find the perfect homestay by understanding their needs in their preferred language and providing accurate, helpful search results.''',
    tools=[MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="http://localhost:8080/homestay/mcp",
        )
    )],  
)

class SessionManager:
    """Manages agent sessions with proper cleanup and error handling"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Tuple[InMemorySessionService, Runner]] = {}
    
    async def get_or_create_session(self, user_id: str) -> Tuple[InMemorySessionService, Runner]:
        """Get existing session or create a new one"""
        if user_id in self.active_sessions:
            return self.active_sessions[user_id]
        
        try:
            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=user_id  
            )
            runner = Runner(
                app_name=APP_NAME,
                agent=root_agent,
                session_service=session_service,
            )
            
            self.active_sessions[user_id] = (session, runner)
            logger.info(f"Created new session for user {user_id}")
            return session, runner
            
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create agent session: {str(e)}"
            )
    
    def cleanup_session(self, user_id: str):
        """Clean up a user session"""
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
            logger.info(f"Cleaned up session for user {user_id}")

# Global session manager
session_manager = SessionManager()

async def process_agent_response(runner: Runner, content: Content, session_id: str, user_id: str) -> AgentResponse:
    """Process agent response with comprehensive error handling and data extraction"""
    try:
        final_response_text = None
        final_event = None
        all_events = []
        
        # Run the agent and collect events
        async for events in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            all_events.append(events)
            
            if events.is_final_response():
                final_event = events
                
                # Extract response text
                if events.content and events.content.parts:
                    final_response_text = events.content.parts[0].text
                elif events.actions and events.actions.escalate:
                    final_response_text = f"Agent escalated: {events.error_message or 'No specific message.'}"
                break
        
        # Handle case where no final event was received
        if not final_event:
            logger.warning(f"No final event received for user {user_id}")
            return AgentResponse(
                success=False,
                error="No response received from agent"
            )
        
        # Extract additional metadata
        grounding_metadata = None
        if hasattr(final_event, 'groundingMetadata') and final_event.groundingMetadata:
            grounding_metadata = {
                'grounding_chunks': len(final_event.groundingMetadata.groundingChunks) if final_event.groundingMetadata.groundingChunks else 0,
                'grounding_supports': len(final_event.groundingMetadata.groundingSupports) if final_event.groundingMetadata.groundingSupports else 0,
                'search_entry_point': bool(final_event.groundingMetadata.searchEntryPoint) if final_event.groundingMetadata.searchEntryPoint else False
            }
        
        usage_metadata = None
        if hasattr(final_event, 'usageMetadata') and final_event.usageMetadata:
            usage_metadata = {
                'timestamp': getattr(final_event, 'timestamp', None)
            }
        
        return AgentResponse(
            success=True,
            response_text=final_response_text,
            author=getattr(final_event, 'author', 'unknown'),
            grounding_metadata=grounding_metadata,
            usage_metadata=usage_metadata
        )
        
    except Exception as e:
        logger.error(f"Error processing agent response: {e}")
        return AgentResponse(
            success=False,
            error=f"Failed to process agent response: {str(e)}"
        )

def validate_message_data(mime_type: str, data: str) -> Tuple[bool, str]:
    """Validate message data based on MIME type"""
    if mime_type in SUPPORTED_TEXT_TYPES:
        if len(data) > MAX_MESSAGE_LENGTH:
            return False, f"Text message too long. Maximum length: {MAX_MESSAGE_LENGTH}"
        if not data.strip():
            return False, "Text message cannot be empty"
    
    elif any(mime_type.startswith(audio_type) for audio_type in SUPPORTED_AUDIO_TYPES):
        try:
            base64.b64decode(data)
        except Exception:
            return False, "Invalid base64 encoded audio data"
    
    else:
        return False, f"Unsupported MIME type: {mime_type}. Supported types: {SUPPORTED_TEXT_TYPES + SUPPORTED_AUDIO_TYPES}"
    
    return True, ""

# Initialize FastAPI app
app = FastAPI(
    title="Google ADK Robust Chat App",
    description="A robust chat application using Google Agent Development Kit",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "error": f"Validation error: {exc}"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )

@app.get("/")
async def root():
    """Serve the main HTML interface"""
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(static_index)
    else:
        return {"message": "Google ADK Robust Chat App", "status": "running"}

@app.post("/send/{user_id}", response_model=AgentResponse)
async def send_message_endpoint(user_id: int, request: Request):
    """Enhanced endpoint for sending messages to the agent"""
    
    if user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID must be a positive integer"
        )
    
    user_id_str = str(user_id)
    
    try:
        # Parse request body
        body = await request.json()
        message_request = MessageRequest(**body)
        
        # Validate message data
        is_valid, error_msg = validate_message_data(message_request.mime_type, message_request.data)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Build Content object based on MIME type
        if message_request.mime_type in SUPPORTED_TEXT_TYPES:
            parts = [genai_types.Part(text=message_request.data)]
            content = genai_types.Content(role="user", parts=parts)
            
        elif any(message_request.mime_type.startswith(audio_type) for audio_type in SUPPORTED_AUDIO_TYPES):
            try:
                audio_bytes = base64.b64decode(message_request.data)
                parts = [genai_types.Part.from_bytes(data=audio_bytes, mime_type=message_request.mime_type)]
                content = genai_types.Content(role="user", parts=parts)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to decode audio data: {str(e)}"
                )
        
        # Get or create session
        session, runner = await session_manager.get_or_create_session(user_id_str)
        
        # Process the message through the agent
        response = await process_agent_response(runner, content, session.id, user_id_str)
        
        logger.info(f"Processed message for user {user_id_str}: success={response.success}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in send_message_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.delete("/sessions/{user_id}")
async def clear_session_endpoint(user_id: int):
    """Clear a user's session"""
    if user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID must be a positive integer"
        )
    
    user_id_str = str(user_id)
    session_manager.cleanup_session(user_id_str)
    
    return {"success": True, "message": f"Session cleared for user {user_id}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": APP_NAME,
        "active_sessions": len(session_manager.active_sessions)
    }

@app.get("/sessions")
async def list_sessions():
    """List active sessions (for debugging)"""
    return {
        "active_sessions": list(session_manager.active_sessions.keys()),
        "total_sessions": len(session_manager.active_sessions)
    }

if __name__ == "__main__":
    # Ensure static directory exists
    STATIC_DIR.mkdir(exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )