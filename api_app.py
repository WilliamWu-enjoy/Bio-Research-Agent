from typing import List, Optional, Dict, Any
import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from marshmallow.utils import to_iso_time
from pydantic import BaseModel, Field

from react_agent import ReactAgent

app = FastAPI(title="Bio Agent API", version="0.1.0")

# 懒加载，避免服务启动时因模型配置问题直接挂掉
_agent_instance: Optional[ReactAgent] = None
_agent_init_error: Optional[str] = None


class ChatMessage(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str = Field(..., description="message content")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    chat_history: List[ChatMessage] = []


class ChatResponseData(BaseModel):
    answer: str
    request_id: str
    latency_ms: int
    model: str


class ApiResponse(BaseModel):
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

def get_agent() -> ReactAgent:
    global _agent_instance,_agent_init_error
    if _agent_instance is not None:
        return _agent_instance
    if _agent_init_error is not None:
        raise  RuntimeError(_agent_init_error)
    try:
        _agent_instance = ReactAgent()
        return _agent_instance
    except Exception as e:
        _agent_init_error = str(e)
        raise RuntimeError(_agent_init_error)

@app.get("/v1/health")
def health() -> Dict[str, Any]:
    model_key_exists = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "status": "UP",
            "service": "bio-agent-api",
            "model_config_loaded": model_key_exists,
            "agent_initialized": _agent_instance is not None,
            "time": int(time.time())
        },
        "error": None
    }
@app.post("/v1/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    start = time.time()
    try:
        agent = get_agent()
        history = [{"role": m.role, "content": m.content} for m in req.chat_history]
        answer = agent.execute(query=req.query, chat_history=history)
        latency_ms = int((time.time() - start) * 1000)
        return {
            "code": 0,
            "message": "ok",
            "data": {
                "answer": answer,
                "request_id": request_id,
                "latency_ms": latency_ms,
                # 先给固定值，T2再改成读取配置
                "model": "Qwen/Qwen3.5-397B-A17B"
            },
            "error": None
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": 5000,
                "message": "INTERNAL_ERROR",
                "request_id": request_id,
                "detail": str(e)
            }
        )