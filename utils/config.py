from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_timeout: int = int(os.getenv("GROQ_TIMEOUT", "60"))
    groq_max_retries: int = int(os.getenv("GROQ_MAX_RETRIES", "3"))
    groq_retry_backoff: float = float(os.getenv("GROQ_RETRY_BACKOFF", "1.0"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    asr_model: str = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3")
    voice_enabled: bool = _env_flag("VOICE_ENABLED", True)
    top_k: int = int(os.getenv("TOP_K", "4"))
    rag_min_relevance: float = float(os.getenv("RAG_MIN_RELEVANCE", "0.25"))
    debug: bool = _env_flag("DEBUG")

settings = Settings()
