import uvicorn

from app.core.config import settings


def main() -> None:
    uvicorn.run(
        "app:application:get_app",
        workers=settings().WORKER_COUNT,
        host=settings().HOST,
        port=settings().PORT,
        log_level=settings().LOG_LEVEL,
        reload=settings().RELOAD,
        factory=True,
    )


if __name__ == "__main__":
    main()
