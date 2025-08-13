
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
from google.adk.agents import Agent,SequentialAgent
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
    print(f"LLM Response: response_dump={response_dump}")

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


### FIXED ### - Updated filter_agent instruction to store structured data in state
filter_agent = Agent(
    model='gemini-2.5-flash',
    name='homestay_search_agent',
    description='A bilingual homestay search assistant for Nepal with voice message support.',
    instruction='''You are a specialized homestay search assistant for Nepal. You understand English and Nepali queries and use MCP tools to filter homestays accurately.

## MCP TOOL: search_homestays

### METHOD 1: Natural Language (For Complex Queries)
Use for complex, mixed, or nuanced requests:
```
search_homestays(
    natural_language_description="homestay with trekking and fishing near Kathmandu with rating above 4",
    limit=10
)
```

### METHOD 2: Direct Parameters (For Simple Queries)
Use for clear, specific requirements:

#### Location Filters:
- `province="Madhesh"` (Province/प्रदेश)
- `district="Sarlahi"` (District/जिल्ला) 
- `municipality="Malangwa"` (Municipality/नगरपालिका)

Provide the most specific location user mentions (municipality > district > province). If multiple are provided, include all.

#### Feature Filters (Use English keywords):
- `any_local_attractions=["trekking", "museum", "fishing"]` - ANY of these
- `local_attractions=["trekking"]` - ALL of these (stricter)
- `any_infrastructure=["clean water", "toilet", "wifi", "mobile"]` - ANY of these
- `infrastructure=["clean water"]` - ALL of these (stricter)
- `any_tourism_services=["local food", "cultural program", "welcome"]` - ANY of these
- `tourism_services=["local food"]` - ALL of these (stricter)

#### Quality & Paging:
- `min_average_rating=4.0` (Minimum rating)
- `status="approved"` (Approved homestays only)
- `limit=10`, `skip=0` (Pagination)
- `sort_order="desc"` (or "asc")

#### Logical Combination (advanced):
- Default is `logical_operator="AND"`.
- Use `logical_operator="OR"` when you have multiple `any_*` features in the same category or want broader matches.
- Use `logical_operator="MIXED"` when combining must-have lists (e.g., `local_attractions`) with optional lists (`any_infrastructure`, `any_tourism_services`) across categories.

### KEYWORD MAPPING (Nepali → English):
**Attractions:** "ट्रेकिङ"→"trekking", "माछा मार्ने"→"fishing", "संग्रहालय"→"museum", "सफारी"→"safari", "चराचुरुङ्गी"→"bird watching"
**Infrastructure:** "सफा पानी"→"clean water", "शौचालय"→"toilet", "इन्टरनेट"→"wifi", "मोबाइल नेटवर्क"→"mobile", "सोलार"→"solar lighting"
**Services:** "स्थानीय खाना"→"local food", "सांस्कृतिक कार्यक्रम"→"cultural program", "स्वागत/विदाइ"→"welcome"
**Locations:** "मधेश प्रदेश"→"Madhesh", "काठमाडौं"→"Kathmandu", "पोखरा"→"Pokhara"

### TOOL USAGE PATTERNS:

**Simple Location Search:**
```
search_homestays(province="Madhesh", limit=10)
```

**Feature-Based Search:**
```
search_homestays(any_local_attractions=["trekking", "museum"], any_infrastructure=["clean water"], limit=8)
```

**Quality + Features (with OR):**
```
search_homestays(min_average_rating=4.0, any_tourism_services=["local food"], municipality="Kathmandu", logical_operator="OR", limit=5)
```

**Complex Natural Language:**
```
search_homestays(natural_language_description="homestay with trekking facilities in mountainous region with good rating", limit=10)
```

CRITICAL OUTPUT REQUIREMENT — NO ALTERATION ALLOWED
After calling the `search_homestays` tool and receiving its response, you MUST output
the exact same JSON object returned by the tool — same field names, same structure,
same values, same ordering.

DO NOT:
- Change field names
- Change casing (e.g., camelCase to snake_case)
- Remove or add fields
- Reorder keys
- Transform values
- Add explanations or extra text outside the JSON

Your output must be byte-for-byte identical to the JSON returned by the tool.

### CRITICAL RULES:
1. Do NOT mix `natural_language_description` with specific parameters in one call.
2. ALWAYS use English for location names in parameters (Nepali is fine inside natural language).
3. PREFER `any_*` parameters for flexible matching; use strict lists only when the user says ALL are required.
4. TRANSLATE Nepali terms to simple English keywords listed above.
5. ALWAYS output the structured JSON format after getting results from the tool.
6. Extract as much detail as possible from the tool response to populate the JSON structure.''',
    tools=[MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="http://localhost:8080/homestay/mcp",
        )
    )],  
    after_model_callback=after_model_callback,
    output_key="filtered_homestays",  # ### FIXED ### - This stores the agent's output in state
)

### FIXED ### - Updated refiner_agent instruction to properly format the structured data
refiner_agent = Agent(
    model='gemini-2.5-flash',
    name='homestay_search_refiner',
    description='Refines homestay search results and formats them with clickable links.',
    instruction='''You are a homestay result refiner agent that formats filtered homestay results into user-friendly responses with embedded clickable links.

## ACCESSING SHARED STATE:
The filtered homestay results are available in the following format: {filtered_homestays}

## YOUR MAIN TASKS:

### 1. PARSE THE STRUCTURED DATA:
The {filtered_homestays} contains JSON data with:
- search_criteria: What the user was looking for
- total_found: Number of homestays found
- homestays: Array of homestay objects with username, location, features, rating

### 2. CREATE A SHORT SUMMARY:
Format a concise summary based on the data:
- English: "I found [X] homestays with [features] in [location]:"
- Nepali: "मैले [location]मा [features] भएका [X] वटा होमस्टे फेला पारे:"

### 3. CREATE CLICKABLE EMBEDDED LINKS:
For each homestay username in the data, create clickable markdown links:
- URL Pattern: `http://localhost:3000/homestays/username`
- Format: `[username](http://localhost:3000/homestays/username)`
(YOU'LL GET THE USERNAME FROM THE STATE {filtered_homestays?} JSON)
### 4. RESPONSE FORMAT:
```
Your search results for [search_criteria]:

• [homestay1_Name (NAME NOT USERNAME)](http://localhost:3000/homestays/homestay1_username) - [location] ([DONOT DISPLAY OR INCLUDE rating if the rating is not mentioned of empty or 0]Rating: [rating])
• [homestay2_Name (NAME NOT USERNAME)](http://localhost:3000/homestays/homestay2_username) - [location] ([DONOT DISPLAY OR INCLUDE rating if the rating is not mentioned of empty or 0]Rating: [rating])
• [homestay3_Name (NAME NOT USERNAME)](http://localhost:3000/homestays/homestay3_username) - [location] ([DONOT DISPLAY OR INCLUDE rating if the rating is not mentioned of empty or 0]Rating: [rating])
Click on any homestay name to view details and make a booking!
```

### 5. LANGUAGE MATCHING:
- Respond in the same language as detected from the search criteria
- Maintain language consistency in summary text
- Always use English for URLs and usernames

### 6. ERROR HANDLING:
- If {filtered_homestays} is empty or malformed: "No homestay data available. Please try searching again."
- If no homestays in data: "No homestays found matching your criteria. Please try different search terms."

### 7. CRITICAL RULES:
- ALWAYS parse the JSON structure from {filtered_homestays}
- ALWAYS embed links inside homestay usernames (never show raw URLs)
- Use bullet points for listing homestays
- Include rating and location info when available
- Ensure all links use the exact format: [username](http://localhost:3000/homestays/username)
- Extract the homestay usernames from the structured data, not from simple text'''
)
    
homestay_filter_pipeline = SequentialAgent(
    sub_agents=[filter_agent, refiner_agent],
    name="homestay_filter_pipeline", 
    description="Pipeline to filter homestays based on user queries then after refining the results.",
)

root_agent = homestay_filter_pipeline

# Rest of the code remains the same...
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
    try:
        final_response_text = None
        final_event = None

        # Run the agent and collect events from the full stream (do NOT break early)
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # debug log every event type (helps to see sub-agent boundaries)
            logger.debug(f"Received event: final={event.is_final_response()}, author={getattr(event, 'author', None)}, actions={getattr(event, 'actions', None)}")

            # If any event is a final response, remember it (do NOT break)
            if event.is_final_response():
                final_event = event
                # prefer text from content.parts if present
                if getattr(event, "content", None) and getattr(event.content, "parts", None):
                    # choose last part text for safety
                    try:
                        final_response_text = event.content.parts[-1].text
                    except Exception:
                        final_response_text = getattr(event.content.parts[0], "text", None)
                elif getattr(event, "actions", None) and getattr(event.actions, "escalate", False):
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                else:
                    # keep final_event but allow next final responses to overwrite (so we get the last)
                    final_response_text = final_response_text or None

        # after the async for finishes, we will have the last final_event (if any)
        if not final_event:
            logger.warning(f"No final event received for user {user_id}")
            return AgentResponse(success=False, error="No response received from agent")

        # gather grounding/usage metadata if available (same as before)
        grounding_metadata = None
        if hasattr(final_event, 'groundingMetadata') and final_event.groundingMetadata:
            grounding_metadata = {
                'grounding_chunks': len(final_event.groundingMetadata.groundingChunks) if final_event.groundingMetadata.groundingChunks else 0,
                'grounding_supports': len(final_event.groundingMetadata.groundingSupports) if final_event.groundingMetadata.groundingSupports else 0,
                'search_entry_point': bool(final_event.groundingMetadata.searchEntryPoint) if final_event.groundingMetadata.searchEntryPoint else False
            }

        usage_metadata = None
        if hasattr(final_event, 'usageMetadata') and final_event.usageMetadata:
            usage_metadata = {'timestamp': getattr(final_event, 'timestamp', None)}

        return AgentResponse(
            success=True,
            response_text=final_response_text,
            author=getattr(final_event, 'author', 'unknown'),
            grounding_metadata=grounding_metadata,
            usage_metadata=usage_metadata
        )

    except Exception as e:
        logger.error(f"Error processing agent response: {e}", exc_info=True)
        return AgentResponse(success=False, error=f"Failed to process agent response: {str(e)}")

# Rest of the FastAPI code remains exactly the same...
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