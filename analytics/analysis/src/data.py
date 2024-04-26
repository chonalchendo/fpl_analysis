import pandas as pd
import sqlalchemy
from sqlalchemy import Connection

from analysis.core.settings import SETTINGS
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


class Database:
    def connection(self) -> Connection:
        """Connect to the database.

        Returns:
            Connection: database connection
        """
        try:
            engine = sqlalchemy.create_engine(SETTINGS["SQLALCHEMY_DATABASE_URI"])
            return engine.connect()
        except Exception as e:
            logger.error(e)
        finally:
            logger.info("Database connection created.")

    def query(self, query: str) -> pd.DataFrame:
        """Execute a query and return the results as a pandas DataFrame.

        Args:
            query (str): SQL query

        Returns:
            pd.DataFrame: query results
        """
        conn = self.connection()
        try:
            df = pd.read_sql(query, conn)
            logger.info(f"Query successfully returned {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(e)
        finally:
            conn.close()
            logger.info("Database connection closed.")


db = Database()
