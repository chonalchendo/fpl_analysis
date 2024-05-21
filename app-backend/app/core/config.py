import enum
from functools import lru_cache

from pydantic import BaseSettings


class LogLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    HOSTL: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKER_COUNT: int = 1
    PROJECT_NAME: str = "Football API"
    API_V1_STR: str = "/api/v1"

    DESCRIPTION: str = "API for player valuation predictions"

    GCP_PROJECT: str = ""
    GCP_BUCKET: str = ""
    GCP_CREDENTIALS: str = ""

    LOG_LEVEL: LogLevel = LogLevel.INFO

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = "utf-8"


@lru_cache()
def settings():
    return Settings()
