from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/nba_players_2025_26.csv")
OUTPUT_PATH = Path("data/processed/nba_players_2025_26_clean.csv")

IDENTITY_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"]
STAT_COLUMNS = [
    "PTS", "REB", "AST", "STL", "BLK",
    "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
]


def main():
    players = pd.read_csv(RAW_PATH)
    eligible = players.loc[(players["GP"] >= 10) & (players["MIN"] >= 10)]
    cleaned = eligible[IDENTITY_COLUMNS + STAT_COLUMNS].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print(f"Original players: {len(players)}")
    print(f"Eligible players: {len(cleaned)}")
    print(f"Excluded players: {len(players) - len(cleaned)}")
    print(f"Saved: {OUTPUT_PATH}")
    print(cleaned[["PLAYER_NAME", "GP", "MIN", "PTS"]].head())


if __name__ == "__main__":
    main()