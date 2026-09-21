import pandas as pd


ROSTER_SIZE = 5

RECOMMENDATION_COUNT = 5

ROSTER_SUMMARY_COLUMNS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "FG3A": "Three-point attempts",
}


ROSTER_ANALYSIS_COLUMNS = {
    "PTS": "Scoring",
    "REB": "Rebounding",
    "AST": "Playmaking",
    "STL": "Perimeter defense",
    "BLK": "Rim protection",
    "FG3A": "Three-point volume",
}


def build_roster(players: pd.DataFrame, player_names: list[str]) -> pd.DataFrame:
    """Validate five selected players and return their records in selection order."""

    cleaned_names = [name.strip() for name in player_names]

    if len(cleaned_names) != ROSTER_SIZE:
        raise ValueError("A roster must contain exactly five players.")

    if any(not name for name in cleaned_names):
        raise ValueError("Please select a player for every roster position.")

    normalized_names = [name.casefold() for name in cleaned_names]

    if len(set(normalized_names)) != ROSTER_SIZE:
        raise ValueError("Each roster position must contain a different player.")

    player_lookup = {
        name.casefold(): name
        for name in players["PLAYER_NAME"].tolist()
    }

    invalid_names = [
        name
        for name in cleaned_names
        if name.casefold() not in player_lookup
    ]

    if invalid_names:
        raise ValueError(
            f"Player not found in the dataset: {invalid_names[0]}"
        )

    official_names = [
        player_lookup[name.casefold()]
        for name in cleaned_names
    ]

    roster = (
        players.set_index("PLAYER_NAME")
        .loc[official_names]
        .reset_index()
    )

    return roster


def calculate_roster_averages(roster: pd.DataFrame) -> list[dict]:
    """Calculate the five-player roster's average per-game statistics."""

    averages = []

    for column, label in ROSTER_SUMMARY_COLUMNS.items():
        if column not in roster.columns:
            raise ValueError(
                f"Required roster statistic is missing: {column}"
            )

        averages.append(
            {
                "label": label,
                "abbreviation": column,
                "value": round(float(roster[column].mean()), 1),
            }
        )

    return averages

def analyze_roster(
    players: pd.DataFrame,
    roster: pd.DataFrame,
) -> dict:
    """Compare roster averages with the eligible-player population."""

    comparisons = []

    for column, label in ROSTER_ANALYSIS_COLUMNS.items():
        if column not in players.columns or column not in roster.columns:
            raise ValueError(
                f"Required roster-analysis statistic is missing: {column}"
            )

        population_average = float(players[column].mean())
        population_std = float(players[column].std(ddof=0))
        roster_average = float(roster[column].mean())

        if population_std == 0:
            standardized_score = 0.0
        else:
            standardized_score = (
                roster_average - population_average
            ) / population_std

        if standardized_score >= 0.5:
            status = "Strength"
        elif standardized_score <= -0.5:
            status = "Needs improvement"
        else:
            status = "Balanced"

        comparisons.append(
            {
                "label": label,
                "abbreviation": column,
                "roster_average": round(roster_average, 1),
                "population_average": round(population_average, 1),
                "score": round(standardized_score, 2),
                "status": status,
            }
        )

    ranked_comparisons = sorted(
        comparisons,
        key=lambda statistic: statistic["score"],
        reverse=True,
    )

    return {
        "comparisons": comparisons,
        "strengths": ranked_comparisons[:2],
        "priorities": ranked_comparisons[-2:][::-1],
    }


def recommend_players(
    players: pd.DataFrame,
    roster: pd.DataFrame,
    limit: int = RECOMMENDATION_COUNT,
) -> list[dict]:
    """Recommend players who complement the roster's weakest categories."""

    if limit < 1:
        raise ValueError("The recommendation limit must be at least one.")

    statistic_columns = list(ROSTER_ANALYSIS_COLUMNS)

    required_columns = [
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        *statistic_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in players.columns or column not in roster.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required recommendation column is missing: {missing_columns[0]}"
        )

    roster_names = {
        name.casefold()
        for name in roster["PLAYER_NAME"].tolist()
    }

    candidates = players[
        ~players["PLAYER_NAME"]
        .str.casefold()
        .isin(roster_names)
    ].copy()

    if candidates.empty:
        raise ValueError("No eligible players are available for recommendation.")

    population_averages = players[statistic_columns].mean()
    population_standard_deviations = (
        players[statistic_columns]
        .std(ddof=0)
        .replace(0, 1)
    )

    roster_scores = (
        roster[statistic_columns].mean() - population_averages
    ) / population_standard_deviations

    priority_columns = (
        roster_scores
        .sort_values()
        .head(2)
        .index
        .tolist()
    )

    candidate_scores = (
        candidates[statistic_columns] - population_averages
    ) / population_standard_deviations

    priority_score = candidate_scores[priority_columns].mean(axis=1)
    overall_score = candidate_scores[statistic_columns].mean(axis=1)

    candidates["FIT_SCORE"] = (
        (priority_score * 0.75) + (overall_score * 0.25)
    ).round(2)

    candidates["BEST_FIT_CATEGORY"] = (
        candidate_scores[priority_columns]
        .idxmax(axis=1)
        .map(ROSTER_ANALYSIS_COLUMNS)
    )

    recommendations = (
        candidates
        .sort_values("FIT_SCORE", ascending=False)
        .head(limit)
    )

    display_columns = [
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "FG3A",
        "FIT_SCORE",
        "BEST_FIT_CATEGORY",
    ]

    return recommendations[display_columns].to_dict(orient="records")