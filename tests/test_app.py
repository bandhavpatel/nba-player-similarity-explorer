import pandas as pd
import pytest

import app as flask_app


@pytest.fixture
def client():
    """Create a Flask test client."""

    flask_app.app.config["TESTING"] = True

    with flask_app.app.test_client() as test_client:
        yield test_client


def test_home_page_loads_when_data_is_missing(
    client,
    monkeypatch,
    tmp_path,
):
    missing_path = tmp_path / "missing.csv"
    monkeypatch.setattr(flask_app, "DATA_PATH", missing_path)

    response = client.get("/")

    assert response.status_code == 200
    assert b"NBA Player Similarity Explorer" in response.data
    assert b"Player data is missing" in response.data


def test_roster_route_rejects_empty_selection(
    client,
    monkeypatch,
    tmp_path,
):
    existing_path = tmp_path / "players.csv"
    existing_path.touch()

    sample_players = pd.DataFrame(
        {
            "PLAYER_NAME": [
                "Player One",
                "Player Two",
                "Player Three",
                "Player Four",
                "Player Five",
            ]
        }
    )

    monkeypatch.setattr(flask_app, "DATA_PATH", existing_path)
    monkeypatch.setattr(
        flask_app.pd,
        "read_csv",
        lambda _: sample_players,
    )

    response = client.post("/roster", data={})

    assert response.status_code == 200
    assert b"Please select a player for every roster position" in response.data