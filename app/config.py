import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    dashboard_ai_enabled: bool = _as_bool(os.getenv("DASHBOARD_AI_ENABLED", "true"), default=True)

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


settings = Settings()
