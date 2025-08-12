

import os
from typing import Tuple
import json
import base64
import warnings
from pydantic import BaseModel
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

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    tools=[google_search],  
)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")



#
# ADK Streaming
#

# Load Gemini API Key
load_dotenv()

APP_NAME = "ADK Streaming example"


async def start_agent_session(user_id) -> Tuple[InMemorySessionService, Runner]:
    """Starts an agent session"""

    # Create a Runner
    session_service= InMemorySessionService()
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

    # Create a Session
    
    return session,runner

async def call_agent(runner: Runner, user_input: str, session_id, user_id) -> dict:
    content = user_input if isinstance(user_input, Content) else Content(
        role="user",
        parts=[Part.from_text(text=user_input, mime_type="text/plain")]
    )
    
    final_response_text = None
    final_event = None
    all_events = []  # Store all events to find grounding metadata
    
    async for events in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        all_events.append(events)
        
        if events.is_final_response():
            final_event = events
            if events.content and events.content.parts:
                final_response_text = events.content.parts[0].text
            elif events.actions and events.actions.escalate:
                final_response_text = f"Agent escalated: {events.error_message or 'No specific message.'}"
            break
    
    return {
        # "author": final_event.author if final_event else "unknown",
        # "content": final_event.content if final_event else None,
        # "type": type(final_event).__name__ if final_event else "unknown",
        # "final_response": True,
        # "final_response_text": final_response_text,
        "final_event": final_event,
        # "all_events": all_events
    }

# FastAPI web app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Store active sessions
active_sessions = {}


@app.get("/")
async def root():
    """Serves the index.html"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# @app.get("/events/{user_id}")
# async def sse_endpoint(user_id: int, is_audio: str = "false"):
#     """SSE endpoint for agent to client communication"""

#     # Start agent session
#     user_id_str = str(user_id)
#     ...

@app.post("/send/{user_id}")
async def send_message_endpoint(user_id: int, request: Request):
    """HTTP endpoint for client to agent communication"""

    user_id_str = str(user_id)

    body = await request.json()
    mime_type = body.get("mime_type", "").strip()
    data = body.get("data")

    # ✅ Check supported MIME types first
    if not (mime_type.startswith("audio/webm") or mime_type == "text/plain"):
        return {"error": f"Unsupported mime_type: {mime_type}"}

    # ✅ Then check if data exists
    if not data:
        return {"error": "Missing data"}

    # ✅ Build the Content object
    if mime_type.startswith("audio/webm"):
        try:
            audio_bytes = base64.b64decode(data)
        except Exception as e:
            return {"error": f"Invalid base64 audio data: {e}"}

        # Here we must set mime_type because it's binary
        parts = [genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/webm")]
        content = genai_types.Content(role="user", parts=parts)

    elif mime_type == "text/plain":
        # No mime_type arg for from_text()
        parts = [genai_types.Part(text=data)]
        content = genai_types.Content(role="user", parts=parts)

    # ✅ Start session and call agent
    session, runner = await start_agent_session(user_id_str)
    response = await call_agent(runner, content, session_id=session.id, user_id=user_id_str)

    # ✅ Safety check for no response
    if not response.get("final_event"):
        return {
            "error": "No final response from agent.",
            "details": response
        }

    return response


