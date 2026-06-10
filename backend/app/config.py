from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ─── App ─────────────────────────────────────────────────
    app_env: str = "development"
    secret_key: str = "change_me"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    # ─── Database ────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://football:football@localhost:5432/football_db"
    database_url_sync: str = "postgresql://football:football@localhost:5432/football_db"

    # ─── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ─── API-Football ────────────────────────────────────────
    api_football_key: str = ""
    api_football_host: str = "api-football-v1.p.rapidapi.com"
    api_football_base_url: str = "https://api-football-v1.p.rapidapi.com/v3"

    # ─── Odds API ────────────────────────────────────────────
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # ─── Telegram ────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ─── Twilio ──────────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_number: str = ""

    # ─── ML ──────────────────────────────────────────────────
    model_dir: str = "./ml/saved_models"
    value_bet_edge_threshold: float = 0.10

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
