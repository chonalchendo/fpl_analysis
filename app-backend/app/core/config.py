import enum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class LogLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    RELOAD: bool = False
    WORKER_COUNT: int = 1
    PROJECT_NAME: str = "Football API"
    API_V1_STR: str = "/api/v1"

    DESCRIPTION: str = "API for player valuation predictions"

    # buckets for gcp storage in deployment
    GCP_PROJECT: str | None = None
    GCP_BUCKET: str | None = None
    GCP_SERVICE_ACCOUNT_JSON_PATH: str | None = None

    # buckets for gcp storage in local development
    LOCAL_GCP_PROJECT: str | None = None
    LOCAL_GCP_BUCKET: str | None = None
    LOCAL_GCP_SERVICE_ACCOUNT_JSON_PATH: str | None = None

    PREDICTIONS: Path = Path().cwd() / "app" / "data" / "predictions.parquet"

    LOG_LEVEL: LogLevel = LogLevel.INFO

    class Config:
        env_prefix = "APP_BACKEND_"
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()
