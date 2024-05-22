import uvicorn

from app.core.config import get_settings


def main() -> None:
    uvicorn.run(
        "app.application:get_app",
        workers=get_settings().WORKER_COUNT,
        host=get_settings().HOST,
        port=get_settings().PORT,
        log_level=get_settings().LOG_LEVEL.value.lower(),
        reload=get_settings().RELOAD,
        factory=True,
    )


if __name__ == "__main__":
    main()
