"""
Expert-level Chat-only Personal Assistant backend (FastAPI)

Save as `main.py` and run:
  pip install fastapi uvicorn python-dotenv
  uvicorn main:app --reload

Set your .env:
  GEMINI_API_KEY=your_key_here

Notes:
- This backend strictly answers questions related to Muhammad Umar and his expertise.
- CORS restricted to your portfolio and localhost.
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

# ---- Load environment variables ----
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

# ---- CORS whitelist ----
origins = [
    "https://muhammadumar-agentic-portfolio.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Conversation store (in-memory) ----
conversations: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = 40  # cap memory per user

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
        # System prompt: strictly Muhammad Umar related
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
    system_msgs = [m for m in history if m["role"] == "system"]
    non_system = [m for m in history if m["role"] != "system"]
    if len(non_system) > max_msgs:
        non_system = non_system[-max_msgs:]
    history[:] = system_msgs + non_system

async def _call_model(messages: List[Dict[str, str]]) -> str:
    """Call model via Runner or AsyncOpenAI client"""
    if runner is not None:
        try:
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages if m['role'] != 'system'])
            res = await runner.run(prompt_text)
            if isinstance(res, dict) and "output" in res:
                return str(res["output"])
            return str(res)
        except Exception:
            pass

    if external_client is None:
        raise RuntimeError("No model client configured (external_client is None).")

    try:
        chat_attr = getattr(external_client, "chat", None)
        if chat_attr is not None and hasattr(chat_attr, "completions"):
            resp = await external_client.chat.completions.create(model="gemini-2.0-flash", messages=messages)
            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                msg = getattr(choice, "message", None)
                if isinstance(msg, dict):
                    return msg.get("content", "")
                if msg and hasattr(msg, "get"):
                    return msg.get("content", "")
                if msg and hasattr(msg, "content"):
                    return msg.content
            if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                return resp["choices"][0]["message"]["content"]
    except Exception:
        pass

    # fallback
    try:
        if hasattr(external_client, "create_chat_completion"):
            resp = await external_client.create_chat_completion(model="gemini-2.0-flash", messages=messages)
            if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                return resp["choices"][0]["message"]["content"]
    except Exception:
        pass

    raise RuntimeError("Unable to call model with the current AsyncOpenAI client shape. Check SDK version.")

# ---- Routes ----
@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")

    user_id = req.user_id or "default"
    history = _get_or_create_history(user_id)

    if req.system_prompt:
        history.insert(0, {"role": "system", "content": req.system_prompt})

    history.append({"role": "user", "content": req.message})
    _truncate_history(history)

    try:
        assistant_text = await _call_model(history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    history.append({"role": "assistant", "content": assistant_text})
    _truncate_history(history)

    if req.system_prompt:
        if history and history[0]["role"] == "system" and history[0]["content"] == req.system_prompt:
            history.pop(0)

    return ChatResponse(reply=assistant_text, conversation=history)

@app.get("/api/conversations")
async def api_list_conversations():
    data = {uid: (conv[-1]["content"] if conv else None) for uid, conv in conversations.items()}
    return {"count": len(conversations), "conversations": data}

@app.post("/api/reset")
async def api_reset(user_id: Optional[str] = "default"):
    if user_id in conversations:
        conversations.pop(user_id)
    return {"status": "ok", "user_id": user_id}

@app.get("/")
async def root():
    return {"service": "Chat-only Personal Assistant", "version": "1.0"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8002)
