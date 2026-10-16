import os


class Settings:
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "28080"))
    APP_NAME: str = os.getenv("APP_NAME", "llm-spring-ai")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:31b")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    OLLAMA_API_KEY: str = "74de72c56f844e95856064cabbabc130.hFB7pfrT19ytZQrTz9qQVaXP"


settings = Settings()
