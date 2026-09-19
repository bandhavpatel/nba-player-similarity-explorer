from pathlib import Path

import pandas as pd


data_path = Path("data/raw/nba_players_2025_26.csv")
players = pd.read_csv(data_path)

print(f"Rows: {len(players)}")
print(f"Unique player IDs: {players['PLAYER_ID'].nunique()}")
print("\nFirst five players:")
print(players[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST"]].head())

print("\nMissing values in candidate stats:")
stats = ["GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT"]
print(players[stats].isna().sum())