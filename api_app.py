import json
from typing import List, Optional, Dict, Any
import os
import time
import uuid
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from config import model_config
from react_agent import ReactAgent

app = FastAPI(title="Bio Agent API", version="0.1.0")

# 懒加载，避免服务启动时因模型配置问题直接挂掉
_agent_instance: Optional[ReactAgent] = None
_agent_init_error: Optional[str] = None

# =========================
# Pydantic 模型
# =========================
class ChatMessage(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str = Field(..., description="message content")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    # 不要用[]，避免可变默认值坑
    chat_history: List[ChatMessage] = Field(default_factory=list)


# class ChatResponseData(BaseModel):
#     answer: str
#     request_id: str
#     latency_ms: int
#     model: str
#
#
# class ApiResponse(BaseModel):
#     code: int
#     message: str
#     data: Optional[Dict[str, Any]] = None
#     error: Optional[Dict[str, Any]] = None

# =========================
# 统一异常
# =========================
class AppError(Exception):
    def __init__(self, code: int, message: str, detail: str = ""):
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(message)

# =========================
# 工具函数
# =========================
def api_success(
    request_id:str,
    data:Optional[Dict[str,Any]] = None,
    message:str="ok",
    code:int = 0,
) -> Dict[str,Any]:
    return  {
        "code":code,
        "message":message,
        "data":data,
        "error":None,
        "request_id":request_id,
    }

def api_error(
    request_id:str,
    code:int,
    message:str,
    detail:str = "",
) -> Dict[str,Any]:
    return {
        "code":code,
        "message":message,
        "data":None,
        "error":{"detail":detail},
        "request_id":request_id,
    }

def log_event(
    *,
    level:str,
    request_id:str,
    path:str,
    method:str,
    status_code:int,
    latency_ms:int,
    error_code:Optional[int] = None,
    detail: str = "",
) -> None:
    payload = {
        "level":level,
        "request_id":request_id,
        "path":path,
        "method":method,
        "status_code":status_code,
        "latency_ms":latency_ms,
        "error_code":error_code,
        "detail":detail,
        "ts":int(time.time()),
    }
    print(json.dumps(payload,ensure_ascii=False))


def get_agent() -> ReactAgent:
    global _agent_instance, _agent_init_error
    if _agent_instance is not None:
        return _agent_instance

    if _agent_init_error is not None:
        raise AppError(
            code = 2001,
            message="MODEL_NOT_READY",
            detail=_agent_init_error
        )

    try:
        _agent_instance = ReactAgent()
        return _agent_instance
    except Exception as e:
        _agent_init_error = str(e)
        raise AppError(
            code=2001,
            message="MODEL_NOT_READY",
            detail=_agent_init_error
        )

# =========================
# 中间件： request_id + 基础鉴权
# =========================

PUBLIC_PATH_PREFIXES = ("/docs", "/redoc", "/openapi.json", "/favicon.ico")
PUBLIC_PATH_EXACT = {"/", "/v1/health"}


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    start_ts = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start_ts = start_ts
    path = request.url.path
    method = request.method
    # 1) 路由白名单（不鉴权）
    if path not in PUBLIC_PATH_EXACT and not path.startswith(PUBLIC_PATH_PREFIXES):
        expected_api_key = os.getenv("APP_API_KEY", "").strip()
        provided_api_key = request.headers.get("X-API-Key", "").strip()
        # 若配置了 APP_API_KEY，则强制校验；未配置时可按需放开
        if expected_api_key and provided_api_key != expected_api_key:
            latency_ms = int((time.time() - start_ts) * 1000)
            body = api_error(
                request_id=request_id,
                code=1002,
                message="AUTH_FAILED",
                detail="invalid or missing X-API-Key",
            )
            log_event(
                level="warning",
                request_id=request_id,
                path=path,
                method=method,
                status_code=401,
                latency_ms=latency_ms,
                error_code=1002,
                detail="invalid or missing X-API-Key",
            )
            return JSONResponse(status_code=401, content=body)
    # 2) 正常放行
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response

# =========================
# 全局异常处理
# =========================
@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start = getattr(request.state, "start_ts", time.time())
    latency_ms = int((time.time() - start) * 1000)
    log_event(
        level="warning",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        status_code=400,
        latency_ms=latency_ms,
        error_code=exc.code,
        detail=exc.detail,
    )
    return JSONResponse(
        status_code=400,
        content=api_error(
            request_id=request_id,
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
        ),
    )

@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start = getattr(request.state, "start_ts", time.time())
    latency_ms = int((time.time() - start) * 1000)
    log_event(
        level="error",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        status_code=500,
        latency_ms=latency_ms,
        error_code=5000,
        detail=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content=api_error(
            request_id=request_id,
            code=5000,
            message="INTERNAL_ERROR",
            detail=str(exc),
        ),
    )


# =========================
# 路由
# =========================
@app.get("/")
def root(request: Request):
    request_id = request.state.request_id
    return api_success(
        request_id=request_id,
        data={"service": "bio-agent-api", "docs": "/docs"},
    )

@app.get("/v1/health")
def health(request: Request):

    request_id = request.state.request_id

    model_key_exists = bool(os.environ.get("OPENAI_API_KEY"))
    data = {
        "status": "UP",
        "service": "bio-agent-api",
        "model_config_loaded": model_key_exists,
        "agent_initialized": _agent_instance is not None,
        "model_name": model_config.model_name,
        "time": int(time.time()),
    }
    latency_ms = int((time.time() - request.state.start_ts) * 1000)
    log_event(
        level="info",
        request_id=request_id,
        path="/v1/health",
        method="GET",
        status_code=200,
        latency_ms=latency_ms,
    )
    return api_success(request_id=request_id, data=data)


@app.post("/v1/chat")
def chat(req: ChatRequest, request: Request):
    request.state.start_ts = time.time()
    request_id = request.state.request_id
    if not req.query.strip():
        raise AppError(code=1001, message="INVALID_PARAM", detail="query 不能为空")
    agent = get_agent()
    history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    answer = agent.execute(query=req.query, chat_history=history)
    latency_ms = int((time.time() - request.state.start_ts) * 1000)
    data = {
        "answer": answer,
        "latency_ms": latency_ms,
        "model": model_config.model_name,
        "session_id": req.session_id,
    }
    log_event(
        level="info",
        request_id=request_id,
        path="/v1/chat",
        method="POST",
        status_code=200,
        latency_ms=latency_ms,
    )
    return api_success(request_id=request_id, data=data)


# 该文档的说明：
'''
ChatRequest:作用是定义/v1/chat的输入契约，并让FastAPI自动做参数校验
get_agent(): 懒加载避免启动即失败
/v1/health检查了环境变量是否有OPENAI_API_KEY, Agent是否已经初始化
/1/chat 的五步流程：
1生成 request_id 和开始时间
2通过 get_agent() 获取（或初始化）Agent
3把 chat_history 转换成 Agent 可用的消息格式
4调用 agent.execute() 获取回答
5计算 latency_ms 并返回统一 JSON 结果
request_id可以从日志里精准定位那次请求，把 API 日志、工具日志、模型调用日志串成一条链路
HTTPException写法的不足：与成功返回的结果不一致，导致前端处理麻烦，生产做法：加全局异常处理器，把所有异常统一包装成同一响应结构
'''
