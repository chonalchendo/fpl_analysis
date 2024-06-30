from components import configure_sidebar, create_dataframe, create_plot
from core.config import configure_app, configure_overview
from services import query_api


def main() -> None:
    configure_app()
    configure_overview()
    query = configure_sidebar()
    data = query_api(query)
    create_plot(data)
    create_dataframe(data)


if __name__ == "__main__":
    main()
