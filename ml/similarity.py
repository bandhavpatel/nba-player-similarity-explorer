from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/nba_players_2025_26_clean.csv")

FEATURES = [
    "PTS", "REB", "AST", "STL", "BLK",
    "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
]


def find_similar_players(player_name, count=5):
    players = pd.read_csv(DATA_PATH)

    matches = players.index[
        players["PLAYER_NAME"].str.casefold() == player_name.casefold()
    ]
    if len(matches) == 0:
        raise ValueError(f"Player not found among eligible players: {player_name}")

    # Put stats with different units on a comparable scale.
    scaled_stats = StandardScaler().fit_transform(players[FEATURES])

    # The closest row is the selected player, so request one extra result.
    model = NearestNeighbors(n_neighbors=count + 1, metric="euclidean")
    model.fit(scaled_stats)

    distances, indices = model.kneighbors(
        scaled_stats[matches[0]].reshape(1, -1)
    )

    results = players.iloc[indices[0][1:]][
        ["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "AST", "FG3A"]
    ].copy()
    results["DISTANCE"] = distances[0][1:]
    return results


if __name__ == "__main__":
    print("Players most similar to Jalen Brunson:")
    print(find_similar_players("Jalen Brunson").to_string(index=False))