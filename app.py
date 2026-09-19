from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

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


if __name__ == "__main__":
    app.run(debug=True)