"""
Expert-level Chat-only Personal Assistant backend (FastAPI)

Save as `main.py` and run:
  pip install fastapi uvicorn python-dotenv
  uvicorn main:app --reload

Set your .env:
  GEMINI_API_KEY=your_key_here

This file expects an AsyncOpenAI-compatible client (your snippet).
It has fallbacks for different SDK shapes (common wrappers).
"""

import os
import asyncio
import time
from typing import Any, Dict, List, Optional

# ---- Suppress noisy TF / C++ logs early ----
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---- Try to import your SDK (graceful fallback if layout differs) ----
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

try:
    from agents import OpenAIChatCompletionsModel, RunConfig, Runner
except Exception:
    OpenAIChatCompletionsModel = None
    RunConfig = None
    Runner = None

# ---- Load env ----
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables. Create .env with GEMINI_API_KEY=...")

# ---- Configure client & optional Runner ----
external_client = None
model_wrapper = None
runner = None

if AsyncOpenAI is not None:
    external_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

if OpenAIChatCompletionsModel is not None and external_client is not None:
    try:
        model_wrapper = OpenAIChatCompletionsModel(
            model="gemini-2.0-flash",
            openai_client=external_client,
        )
        if RunConfig is not None:
            cfg = RunConfig(model=model_wrapper, model_provider=external_client, tracing_disabled=True)
            try:
                runner = Runner(cfg)
            except Exception:
                runner = None
    except Exception:
        model_wrapper = None

# ---- App ----
app = FastAPI(title="Chat-only Personal Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Conversation store (in-memory) ----
# Structure: conversations[user_id] = [ {role: "system"/"user"/"assistant", "content": "..."} , ... ]
conversations: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = 40  # cap memory per user (adjust as needed)

# ---- API models ----
class ChatRequest(BaseModel):
    user_id: Optional[str] = "default"
    message: str
    system_prompt: Optional[str] = None  # optional per-request system prompt

class ChatResponse(BaseModel):
    reply: str
    conversation: List[Dict[str, str]]

# ---- Helpers ----
def _get_or_create_history(user_id: str) -> List[Dict[str, str]]:
    history = conversations.get(user_id)
    if history is None:
        # You can set a global system prompt here
        history = [
    {
        "role": "system",
        "content": (
            "You are 'UmarAI', the official personal AI assistant of Muhammad Umar — "
            "an expert Agentic AI and Full Stack Developer specialized in Next.js, React, Node.js, TypeScript, "
            "TailwindCSS, MongoDB, Express, and OpenAI Agents SDK. "
            "You only respond to questions directly related to Muhammad Umar, his projects, portfolio, or his area of expertise "
            "(Agentic AI, Full Stack, or related technologies). "
            "If a question is outside these topics — politely refuse by saying: "
            "'I'm sorry, but I only answer questions related to Muhammad Umar or his technical expertise.' "
            "Be concise, polite, and confident."
        )
    }
]


        conversations[user_id] = history
    return history


def _truncate_history(history: List[Dict[str, str]], max_msgs: int = MAX_HISTORY_MESSAGES) -> None:
    # keep last `max_msgs` messages plus the system message (if present)
    system_msgs = [m for m in history if m["role"] == "system"]
    non_system = [m for m in history if m["role"] != "system"]
    if len(non_system) > max_msgs:
        non_system = non_system[-max_msgs:]
    history[:] = (system_msgs + non_system)

async def _call_model(messages: List[Dict[str, str]]) -> str:
    """
    Try different common AsyncOpenAI shapes; adapt to your installed SDK.
    Returns assistant text or raises RuntimeError.
    """
    if runner is not None:
        # If you have a Runner/Agents SDK that accepts structured messages, try to use it.
        try:
            # Many runners accept a single prompt string; this is best-effort.
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages if m['role'] != 'system'])
            res = await runner.run(prompt_text)
            if isinstance(res, dict) and "output" in res:
                return str(res["output"])
            return str(res)
        except Exception:
            # fallback to direct client
            pass

    if external_client is None:
        raise RuntimeError("No model client configured (external_client is None).")

    # Try shape: external_client.chat.completions.create(...)
    try:
        chat_attr = getattr(external_client, "chat", None)
        if chat_attr is not None and hasattr(chat_attr, "completions"):
            resp = await external_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages,
            )
            # Extract result: support multiple response shapes
            if hasattr(resp, "choices") and resp.choices:
                # Common shape: resp.choices[0].message.content or .message.get("content")
                choice = resp.choices[0]
                # try attribute access
                msg = getattr(choice, "message", None)
                if isinstance(msg, dict):
                    return msg.get("content", "")
                if msg and hasattr(msg, "get"):
                    return msg.get("content", "")
                if msg and hasattr(msg, "content"):
                    return msg.content
            # fallback parsing if dict-like
            if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                return resp["choices"][0]["message"]["content"]
    except Exception:
        pass

    # Try shape: external_client.create_chat_completion(...)
    try:
        if hasattr(external_client, "create_chat_completion"):
            resp = await external_client.create_chat_completion(model="gemini-2.0-flash", messages=messages)
            if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                return resp["choices"][0]["message"]["content"]
            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                msg = getattr(choice, "message", None)
                if isinstance(msg, dict):
                    return msg.get("content", "")
    except Exception:
        pass

    # Last-resort: try a generic `chat_completion` call
    try:
        if hasattr(external_client, "chat_completion") or hasattr(external_client, "create_completion"):
            # Attempt generic call shape (not guaranteed)
            resp = await getattr(external_client, "create_chat_completion", getattr(external_client, "chat_completion", None))(
                model="gemini-2.0-flash", messages=messages
            )
            if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                return resp["choices"][0]["message"]["content"]
    except Exception:
        pass

    raise RuntimeError("Unable to call model with the current AsyncOpenAI client shape. Check SDK version and available methods.")

# ---- Routes ----
@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """
    Send a chat message. Maintains per-user conversation history.
    Returns assistant reply + current conversation (for frontend display).
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")

    user_id = req.user_id or "default"
    history = _get_or_create_history(user_id)

    # Optionally inject a per-request system prompt (e.g., "Act like a teacher")
    if req.system_prompt:
        # add ephemeral system message at position 0 (do not persist long-term)
        history.insert(0, {"role": "system", "content": req.system_prompt})

    # Append user message
    history.append({"role": "user", "content": req.message})
    _truncate_history(history)

    # Call model
    try:
        assistant_text = await _call_model(history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Append assistant reply to history and truncate
    history.append({"role": "assistant", "content": assistant_text})
    _truncate_history(history)

    # If we added an ephemeral per-request system prompt, remove it so it doesn't persist
    if req.system_prompt:
        # remove the first system prompt occurrence that equals the injected prompt
        if history and history[0]["role"] == "system" and history[0]["content"] == req.system_prompt:
            history.pop(0)

    return ChatResponse(reply=assistant_text, conversation=history)

@app.get("/api/conversations")
async def api_list_conversations():
    """
    Return a list of user_ids and their last message times (basic).
    """
    data = {uid: (conv[-1]["content"] if conv else None) for uid, conv in conversations.items()}
    return {"count": len(conversations), "conversations": data}

@app.post("/api/reset")
async def api_reset(user_id: Optional[str] = "default"):
    """
    Reset conversation for given user_id (clears memory).
    """
    if user_id in conversations:
        conversations.pop(user_id)
    return {"status": "ok", "user_id": user_id}

@app.get("/")
async def root():
    return {"service": "Chat-only Personal Assistant", "version": "1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)