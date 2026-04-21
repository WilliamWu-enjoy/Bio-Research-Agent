import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class ModelConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "")
    model_name: str = os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen3.5-397B-A17B")
    temperature: float = _get_float("CHAT_TEMPERATURE", 0.1)
    streaming: bool = _get_bool("CHAT_STREAMING", True)


@dataclass(frozen=True)
class AgentConfig:
    recursion_limit: int = _get_int("AGENT_RECURSION_LIMIT", 5)
    system_prompt: str = os.getenv(
        "AGENT_SYSTEM_PROMPT",
        "你是一个顶级的生物医学工程科研辅助智能体。"
        "你的任务是协助科研人员解决文献检索、实验参数计算和前沿数据探索等问题。"
        "你有权自主决定是否调用工具。在回复时，请保持严谨、专业的学术口吻。"
    )


@dataclass(frozen=True)
class RagConfig:
    persist_directory: str = os.getenv("RAG_PERSIST_DIR", "./chroma_db")
    embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
    chunk_size: int = _get_int("RAG_CHUNK_SIZE", 500)
    chunk_overlap: int = _get_int("RAG_CHUNK_OVERLAP", 50)
    retrieval_top_k: int = _get_int("RAG_TOP_K", 3)
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    )


model_config = ModelConfig()
agent_config = AgentConfig()
rag_config = RagConfig()