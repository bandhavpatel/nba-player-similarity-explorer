import pandas as pd


ROSTER_SIZE = 5


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