import psycopg2
from sqlalchemy.engine import Connection, create_engine
from sqlalchemy_utils.functions import database_exists
from feature_pipeline.utilities.utils import get_logger
from feature_pipeline.core.settings import SETTINGS


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


def create_database_connection(database: str) -> Connection:
    """Create connection to SQL database.

    Returns:
        Connection: Connection to SQL database.
    """
    db = f'{SETTINGS["SQLALCHEMY_DATABASE_URI"]}/{database}'
    if database_exists(db):
        engine = create_engine(db)
    else:
        create_db(database)
        engine = create_engine(db)
    return engine.connect()


# if __name__ == "__main__":
#     for db in ["transfermarkt", "fbref"]:
#         create_db(db)
