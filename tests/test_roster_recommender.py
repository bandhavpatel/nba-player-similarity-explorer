import pandas as pd
import pytest

from ml.roster_recommender import (
    analyze_roster,
    build_roster,
    calculate_roster_averages,
    recommend_players,
)


@pytest.fixture
def sample_players() -> pd.DataFrame:
    """Create predictable player data for roster tests."""

    return pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Player One",
                "TEAM_ABBREVIATION": "AAA",
                "PTS": 10.0,
                "REB": 2.0,
                "AST": 1.0,
                "STL": 0.5,
                "BLK": 0.2,
                "FG3A": 1.0,
            },
            {
                "PLAYER_NAME": "Player Two",
                "TEAM_ABBREVIATION": "BBB",
                "PTS": 20.0,
                "REB": 4.0,
                "AST": 2.0,
                "STL": 1.0,
                "BLK": 0.4,
                "FG3A": 2.0,
            },
            {
                "PLAYER_NAME": "Player Three",
                "TEAM_ABBREVIATION": "CCC",
                "PTS": 30.0,
                "REB": 6.0,
                "AST": 3.0,
                "STL": 1.5,
                "BLK": 0.6,
                "FG3A": 3.0,
            },
            {
                "PLAYER_NAME": "Player Four",
                "TEAM_ABBREVIATION": "DDD",
                "PTS": 40.0,
                "REB": 8.0,
                "AST": 4.0,
                "STL": 2.0,
                "BLK": 0.8,
                "FG3A": 4.0,
            },
            {
                "PLAYER_NAME": "Player Five",
                "TEAM_ABBREVIATION": "EEE",
                "PTS": 50.0,
                "REB": 10.0,
                "AST": 5.0,
                "STL": 2.5,
                "BLK": 1.0,
                "FG3A": 5.0,
            },
            {
                "PLAYER_NAME": "Player Six",
                "TEAM_ABBREVIATION": "FFF",
                "PTS": 60.0,
                "REB": 12.0,
                "AST": 6.0,
                "STL": 3.0,
                "BLK": 1.2,
                "FG3A": 6.0,
            },
        ]
    )


def test_build_roster_preserves_selection_order(sample_players):
    selected_names = [
        "Player Three",
        "Player One",
        "Player Five",
        "Player Two",
        "Player Four",
    ]

    roster = build_roster(sample_players, selected_names)

    assert roster["PLAYER_NAME"].tolist() == selected_names
    assert len(roster) == 5


def test_build_roster_rejects_duplicate_players(sample_players):
    selected_names = [
        "Player One",
        "Player One",
        "Player Three",
        "Player Four",
        "Player Five",
    ]

    with pytest.raises(ValueError, match="different player"):
        build_roster(sample_players, selected_names)


def test_calculate_roster_averages(sample_players):
    selected_names = [
        "Player One",
        "Player Two",
        "Player Three",
        "Player Four",
        "Player Five",
    ]

    roster = build_roster(sample_players, selected_names)
    averages = calculate_roster_averages(roster)

    values = {
        statistic["abbreviation"]: statistic["value"]
        for statistic in averages
    }

    assert values == {
        "PTS": 30.0,
        "REB": 6.0,
        "AST": 3.0,
        "FG3A": 3.0,
    }


def test_analyze_roster_returns_six_comparisons(sample_players):
    selected_names = [
        "Player One",
        "Player Two",
        "Player Three",
        "Player Four",
        "Player Five",
    ]

    roster = build_roster(sample_players, selected_names)
    analysis = analyze_roster(sample_players, roster)

    assert len(analysis["comparisons"]) == 6
    assert len(analysis["strengths"]) == 2
    assert len(analysis["priorities"]) == 2

    categories = {
        statistic["label"]
        for statistic in analysis["comparisons"]
    }

    assert categories == {
        "Scoring",
        "Rebounding",
        "Playmaking",
        "Perimeter defense",
        "Rim protection",
        "Three-point volume",
    }


def test_recommendations_exclude_selected_players(sample_players):
    selected_names = [
        "Player One",
        "Player Two",
        "Player Three",
        "Player Four",
        "Player Five",
    ]

    roster = build_roster(sample_players, selected_names)
    recommendations = recommend_players(sample_players, roster)

    recommended_names = {
        player["PLAYER_NAME"]
        for player in recommendations
    }

    assert recommended_names.isdisjoint(selected_names)
    assert recommended_names == {"Player Six"}


def test_recommendations_are_sorted_by_fit_score(sample_players):
    extra_player = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Player Seven",
                "TEAM_ABBREVIATION": "GGG",
                "PTS": 70.0,
                "REB": 14.0,
                "AST": 7.0,
                "STL": 3.5,
                "BLK": 1.4,
                "FG3A": 7.0,
            }
        ]
    )

    expanded_players = pd.concat(
        [sample_players, extra_player],
        ignore_index=True,
    )

    selected_names = [
        "Player One",
        "Player Two",
        "Player Three",
        "Player Four",
        "Player Five",
    ]

    roster = build_roster(expanded_players, selected_names)
    recommendations = recommend_players(
        expanded_players,
        roster,
        limit=2,
    )

    fit_scores = [
        player["FIT_SCORE"]
        for player in recommendations
    ]

    assert len(recommendations) == 2
    assert recommendations[0]["PLAYER_NAME"] == "Player Seven"
    assert fit_scores == sorted(fit_scores, reverse=True)