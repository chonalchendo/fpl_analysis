from processing.gcp.storage import gcp


def generate_unique_id() -> dict[str, int]:
    """Generates a unique id for each player."""
    df = gcp.read_df_from_bucket(bucket_name="fbref_db", blob_name="standard.csv")
    players = df["player"].unique().tolist()
    ids = [i for i in range(1, len(players) + 1)]
    return dict(zip(players, ids))
