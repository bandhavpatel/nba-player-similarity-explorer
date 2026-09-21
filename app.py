from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from ml.roster_recommender import (
    analyze_roster,
    build_roster,
    calculate_roster_averages,
)
from ml.similarity import find_similar_players

app = Flask(__name__)

DATA_PATH = Path("data/processed/nba_players_2025_26_clean.csv")


@app.route("/")
def home():
    project_features = [
        "Find statistically similar NBA players",
        "Build and analyze a five-player roster",
        "Receive data-driven roster recommendations",
    ]

    player_name = request.args.get("player_name", "").strip()
    results = None
    error = None
    player_names = []

    if DATA_PATH.exists():
        players = pd.read_csv(DATA_PATH)
        player_names = sorted(players["PLAYER_NAME"].tolist())

        if player_name:
            try:
                matches = find_similar_players(player_name)
                results = matches.to_dict(orient="records")
            except ValueError as exc:
                error = str(exc)
    else:
        error = "Player data is missing. Run the fetch and preprocessing scripts first."

    return render_template(
        "index.html",
        project_features=project_features,
        player_names=player_names,
        player_name=player_name,
        results=results,
        error=error,
    )


@app.route("/roster", methods=["POST"])
def roster():
    if not DATA_PATH.exists():
        return render_template(
            "roster_results.html",
            roster=None,
            roster_averages=None,
            roster_analysis=None,
            error="Player data is missing. Run the fetch and preprocessing scripts first.",
        )

    players = pd.read_csv(DATA_PATH)
    selected_names = [
        request.form.get(f"player_{position}", "")
        for position in range(1, 6)
    ]

    try:
        selected_roster = build_roster(players, selected_names)
        roster_records = selected_roster.to_dict(orient="records")
        roster_averages = calculate_roster_averages(selected_roster)
        roster_analysis = analyze_roster(players, selected_roster)
        error = None
    except ValueError as exc:
        roster_records = None
        roster_averages = None
        roster_analysis = None
        error = str(exc)

    return render_template(
        "roster_results.html",
        roster=roster_records,
        roster_averages=roster_averages,
        roster_analysis=roster_analysis,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)