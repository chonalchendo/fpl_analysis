from processing.gcp.files import gcs


def generate_unique_id() -> dict[str, int]:
    """Generates a unique id for each player."""
    df = gcs.read_csv("fbref_db/standard.csv")
    players = df["player"].unique().tolist()
    ids = [i for i in range(1, len(players) + 1)]
    return dict(zip(players, ids))
