import pandas as pd
import pytest

from ml.similarity import find_similar_players


@pytest.fixture
def sample_players() -> pd.DataFrame:
    """Create predictable player data for similarity tests."""

    return pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Target Player",
                "TEAM_ABBREVIATION": "AAA",
                "PTS": 20.0,
                "REB": 5.0,
                "AST": 6.0,
                "STL": 1.0,
                "BLK": 0.4,
                "FG_PCT": 0.48,
                "FG3A": 6.0,
                "FG3_PCT": 0.38,
                "FTA": 4.0,
                "FT_PCT": 0.82,
            },
            {
                "PLAYER_NAME": "Close Player",
                "TEAM_ABBREVIATION": "BBB",
                "PTS": 20.5,
                "REB": 5.2,
                "AST": 5.8,
                "STL": 1.1,
                "BLK": 0.5,
                "FG_PCT": 0.49,
                "FG3A": 6.2,
                "FG3_PCT": 0.37,
                "FTA": 4.2,
                "FT_PCT": 0.81,
            },
            {
                "PLAYER_NAME": "Different Player",
                "TEAM_ABBREVIATION": "CCC",
                "PTS": 10.0,
                "REB": 10.0,
                "AST": 2.0,
                "STL": 0.5,
                "BLK": 2.0,
                "FG_PCT": 0.56,
                "FG3A": 1.0,
                "FG3_PCT": 0.25,
                "FTA": 7.0,
                "FT_PCT": 0.70,
            },
            {
                "PLAYER_NAME": "Another Player",
                "TEAM_ABBREVIATION": "DDD",
                "PTS": 14.0,
                "REB": 3.0,
                "AST": 3.0,
                "STL": 0.7,
                "BLK": 0.2,
                "FG_PCT": 0.42,
                "FG3A": 3.0,
                "FG3_PCT": 0.32,
                "FTA": 2.0,
                "FT_PCT": 0.75,
            },
        ]
    )


def test_find_similar_players_returns_requested_count(
    sample_players,
    monkeypatch,
):
    monkeypatch.setattr(
        "ml.similarity.pd.read_csv",
        lambda _: sample_players,
    )

    results = find_similar_players("Target Player", count=2)

    assert len(results) == 2
    assert "Target Player" not in results["PLAYER_NAME"].tolist()
    assert results.iloc[0]["PLAYER_NAME"] == "Close Player"


def test_find_similar_players_is_case_insensitive(
    sample_players,
    monkeypatch,
):
    monkeypatch.setattr(
        "ml.similarity.pd.read_csv",
        lambda _: sample_players,
    )

    results = find_similar_players("target player", count=2)

    assert len(results) == 2
    assert results.iloc[0]["PLAYER_NAME"] == "Close Player"


def test_find_similar_players_rejects_unknown_player(
    sample_players,
    monkeypatch,
):
    monkeypatch.setattr(
        "ml.similarity.pd.read_csv",
        lambda _: sample_players,
    )

    with pytest.raises(ValueError, match="Player not found"):
        find_similar_players("Unknown Player")