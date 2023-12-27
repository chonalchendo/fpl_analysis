import psycopg2
from sqlalchemy.engine import Connection, create_engine
from feature_pipeline.utils import get_logger
from feature_pipeline.settings import SETTINGS


logger = get_logger(__name__)


def create_db(database_name: str) -> None:
    """Create a database in PostgreSQL.

    Args:
        database_name (str): name of database to create.
    """
    try:
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user="postgres", host="127.0.0.1", port="5432"
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        # Preparing query to create a database
        sql = f"""CREATE database {database_name}"""

        # Creating a database
        cursor.execute(sql)
        logger.info("---- Database created successfully ----")

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        # Closing the connection
        conn.close()
        logger.info("---- Connection closed ----")


def create_database_connection() -> Connection:
    """Create connection to SQL database.

    Returns:
        Connection: Connection to SQL database.
    """
    engine = create_engine(SETTINGS["SQLALCHEMY_DATABASE_URI"])
    return engine.connect()


if __name__ == "__main__":
    create_db("fantasy_premier_league")
