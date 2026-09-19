from pathlib import Path

from nba_api.stats.endpoints import leaguedashplayerstats


def main():
    print("Fetching 2025-26 regular season player stats...")

    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2025-26",
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        timeout=30,
    )

    players = response.get_data_frames()[0]
    output_path = Path("data/raw/nba_players_2025_26.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(output_path, index=False)

    print(f"Saved {len(players)} rows to {output_path}")
    print("Columns:", list(players.columns))


if __name__ == "__main__":
    main()