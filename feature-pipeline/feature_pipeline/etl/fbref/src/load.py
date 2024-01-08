import pandas as pd
from feature_pipeline.db.database import create_database_connection


def to_sql_database(data: pd.DataFrame, table_name: str, database: str) -> None:
    """Load data into SQL database.

    Args:
        data (pd.DataFrame): players data.
        table_name (str): name of table to load data into.
    """
    conn = create_database_connection(database)
    data.to_sql(table_name, conn, if_exists="replace")