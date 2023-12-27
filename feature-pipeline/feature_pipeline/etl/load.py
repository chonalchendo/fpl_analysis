import pandas as pd
import hopsworks
from feature_pipeline.database import create_database_connection
from great_expectations.core import ExpectationSuite
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from feature_pipeline.settings import SETTINGS


def connect_to_feature_store() -> FeatureStore:
    """Connect to feature store.

    Returns:
        FeatureStore: Feature store connected to.
    """
    # connect to feature store
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )
    return project.get_feature_store()


def create_feature_group(
    feature_store: FeatureStore,
    feature_group_version: int,
    # validation_expectation_suite: ExpectationSuite,
) -> FeatureGroup:
    """Create feature group in feature store.

    Args:
        feature_store (FeatureStore): feature store to create feature group in.
        feature_group_version (int): version of feature group.
        validation_expectation_suite (ExpectationSuite): validation suite for data.

    Returns:
        FeatureGroup: Feature group created.
    """
    return feature_store.get_or_create_feature_group(
        name="fpl_player_statistics",
        version=feature_group_version,
        description="Player statistics for the 2023/24 fantasy premier league season. Data is updated after each game week.",
        primary_key=["player_id", "fixture_id"],
        event_time="kickoff_time_utc",
        online_enabled=False,
        # expectation_suite=validation_expectation_suite,
    )


def to_feature_store(
    data: pd.DataFrame,
    data_description: list[dict[str, str]],
    # validation_expectation_suite: ExpectationSuite,
    featuregroup_version: int,
) -> FeatureGroup:
    """Load data into feature store.

    Args:
        data (pd.DataFrame): players data.
        data_description (list[dict[str, str]]): metadata description for each column.
        validation_expectation_suite (ExpectationSuite): validation suite for data.
        featuregroup_version (int): version of feature group.

    Returns:
        FeatureGroup: Feature group with data loaded.
    """
    feature_store = connect_to_feature_store()
    feature_group = create_feature_group(
        feature_store,
        featuregroup_version,
        # validation_expectation_suite
    )

    # upload data to feature group
    feature_group.insert(
        features=data, overwrite=False, write_options={"wait_for_job": True}
    )

    # add feature descriptions
    for description in data_description:
        feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # update statistics
    feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }

    feature_group.update_statistics_config()
    feature_group.compute_statistics()

    return feature_group


def to_sql_database(data: pd.DataFrame, table_name: str) -> None:
    """Load data into SQL database.

    Args:
        data (pd.DataFrame): players data.
        table_name (str): name of table to load data into.
    """
    conn = create_database_connection()
    data.to_sql(table_name, conn, if_exists="replace")
