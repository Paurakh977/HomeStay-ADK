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
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
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
async def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    print("##" * 22)

    # Get user input text from callback_context if available
    user_input = getattr(callback_context, "user_input", None)
    if user_input is None:
        user_input = "<No input text available>"
    print(f"User input text: {user_input}")

    response_dump = llm_response.model_dump()

    # Full raw response print (optional, comment out if verbose)
    # print(f"LLM Response raw: {response_dump}")

    content = response_dump.get("content")
    if not content:
        print("No content in LLM response.")
        return llm_response

    parts = content.get("parts", [])
    if not parts:
        print("No parts in content.")
        return llm_response

    any_function_call_found = False

    for idx, part in enumerate(parts):
        function_call = part.get("function_call")
        function_response = part.get("function_response")
        text = part.get("text")

        print(f"\nPart #{idx + 1} Text: {text}")

        if function_call:
            any_function_call_found = True
            print(f"Function call made: {function_call.get('name', '<no name>')}")
            print(f"Function call args: {function_call.get('args', {})}")
        else:
            print("No function call in this part.")

        if function_response:
            print(f"Function response: {function_response}")
        else:
            print("No function response in this part.")

    if not any_function_call_found:
        print("\nNo function call found in any part of the response.")

    print("##" * 22)
    return llm_response

# Create root agent with enhanced configuration
root_agent = Agent(
    model='gemini-2.5-flash',
    name='homestay_search_agent',
    description='A bilingual homestay search assistant for Nepal with voice message support.',
    instruction='''You are an intelligent homestay search assistant specialized in helping users find homestays in Nepal. You can understand and respond in both English and Nepali, and work seamlessly with voice messages and text queries.

## 🚨 CRITICAL TOOL USAGE INSTRUCTIONS 🚨

### TWO SEARCH METHODS AVAILABLE

**METHOD 1 - Natural Language (RECOMMENDED for complex queries):**
```
search_homestays(
    natural_language_description="homestay with trekking and fishing facilities in Madhesh province",
    limit=10
)
```

**METHOD 2 - Direct Parameters (GOOD for simple, structured queries):**
```
search_homestays(
    province="Madhesh",
    any_local_attractions=["museum"],
    limit=5
)
```

**BOTH ARE CORRECT - Choose based on query complexity:**
- **Use Natural Language** for complex queries, mixed requirements, or when user language is nuanced
- **Use Direct Parameters** for simple, clear-cut requirements

### 🔧 PARAMETER USAGE GUIDELINES

**For Direct Parameter Method:**
- Use simple English keywords (the tool has keyword mapping)
- Examples: "museum", "trekking", "fishing", "clean water", "toilet"
- Location parameters: province, district, municipality (in English)

**Parameter Types:**
- `any_local_attractions=["museum", "trekking"]` - ANY of these attractions
- `local_attractions=["museum"]` - ALL of these attractions (stricter)
- `any_infrastructure=["clean water", "toilet"]` - ANY of these infrastructure
- `any_tourism_services=["local food"]` - ANY of these services
- `min_average_rating=4.0` - Minimum rating filter

**EXAMPLES - Direct Parameter Method:**

**Simple Museum Search:**
```
search_homestays(
    province="Madhesh", 
    any_local_attractions=["museum"],
    limit=5
)
```

**Multiple Features:**
```
search_homestays(
    province="Madhesh",
    any_local_attractions=["museum", "trekking"],
    any_infrastructure=["clean water"],
    limit=5
)
```

**Mixed Features:**
```
search_homestays(
    province="Madhesh",
    any_local_attractions=["museum"],
    any_tourism_services=["local food"],
    limit=5
)
```

### 🎯 FEATURE MAPPING FOR TRANSLATION
When translating Nepali terms to English for the tool:

**Activities/Attractions:**
- "ट्रेकिङ" → "trekking" or "hiking"
- "माछा मार्ने" → "fishing"  
- "सफारी" → "safari"
- "चराचुरुङ्गी हेर्ने" → "bird watching"
- "संग्रहालय" → "museum"
- "सांस्कृतिक केन्द्र" → "cultural center"

**Infrastructure:**
- "सफा खानेपानी" → "clean drinking water"
- "शौचालय" → "toilet" or "bathroom"
- "इन्टरनेट" → "internet" or "wifi"
- "मोबाइल नेटवर्क" → "mobile network"
- "सोलार बत्ती" → "solar lighting"

**Tourism Services:**
- "स्थानीय खाना" → "local dishes" or "local food"
- "स्वागत सत्कार" → "welcome and farewell"
- "सांस्कृतिक कार्यक्रम" → "cultural program"
- "उपहार" → "gift or souvenir"

**Locations:**
- "मधेश प्रदेश" → "Madhesh province"
- "काठमाडौं" → "Kathmandu"
- "पोखरा" → "Pokhara"

### 🔍 TOOL CALL PATTERNS

**Pattern 1 - Simple Direct Parameters:**
```
search_homestays(
    province="Madhesh",
    any_local_attractions=["museum"],
    limit=5
)
```

**Pattern 2 - Multiple Features Direct:**
```
search_homestays(
    province="Madhesh",
    any_local_attractions=["trekking", "fishing"],
    any_infrastructure=["clean water"],
    limit=10
)
```

**Pattern 3 - Natural Language Complex:**
```
search_homestays(
    natural_language_description="homestay with trekking and if possible fishing and clean water in Madhesh province",
    limit=10
)
```

**Pattern 4 - Location + Rating:**
```
search_homestays(
    municipality="Malangwa",
    any_tourism_services=["local dishes"],
    min_average_rating=4.0,
    limit=8
)
```

### 🗺️ LOCATION HANDLING
The tool has fuzzy matching for locations, so:
- "Malangawa" will match "Malangwa Municipality"
- "काठमाडौं" will match "Kathmandu Metropolitan City" 
- Partial names work well in the natural_language_description

### 📝 RESPONSE STRUCTURE
After getting results, provide:
1. **Summary**: "मैले X वटा होमस्टे फेला पारेको छु..." (in user's language)
2. **Results**: List the homestay usernames
3. **Context**: Mention the search criteria used
4. **Suggestions**: If few/no results, suggest alternatives

### 🚫 WHAT NOT TO DO
- ❌ Don't use overly complex database strings like "Museums & Cultural Centers/आदिवासी संग्रहालय तथा संस्कृति केन्द्रहरू"
- ❌ Don't mix natural_language_description with specific feature parameters
- ❌ Don't pass Nepali text in location parameters (translate to English)

### ✅ WHAT TO DO
- ✅ Use simple English keywords for direct parameters: "museum", "trekking", "fishing"
- ✅ Choose the right method: Natural language for complex queries, direct parameters for simple ones
- ✅ Use `any_*` parameters for flexible matching (recommended)
- ✅ Use `*` (without any_) parameters for strict matching (when user specifically needs ALL features)
- ✅ Always translate location names to English
- ✅ Respond to users in their original language

## Core Capabilities
### Language Understanding
- **Bilingual Processing**: Fluent in both English and Nepali
- **Voice Message Support**: Can process Nepali voice messages converted to text
- **Mixed Language Queries**: Handle queries that mix English and Nepali terms
- **Location Recognition**: Understand Nepali place names, municipalities, districts, and provinces

### Query Processing Indicators
- **Must-have indicators**: "चाहिएको छ", "आवश्यक छ", "need", "required", "must have"
- **Optional indicators**: "यदि सम्भव भए", "हुन्छ भने राम्रो", "if possible", "would be nice", "optionally"

### Response Guidelines
- **Nepali Query → Nepali Response** (with English homestay details)
- **English Query → English Response**
- **Mixed Query → Match user's primary language**
- Provide search results, location context, feature summary, and alternatives when needed

Remember: The key to success is using natural_language_description with clear English descriptions, regardless of the user's input language!''',
    tools=[MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="http://localhost:8080/homestay/mcp",
        )
    )],  
    after_model_callback=after_model_callback,
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