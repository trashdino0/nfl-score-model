import pandas as pd
from nflscore import data as nfl
from nflscore.build import _player_snap_state

inj = nfl.load_injuries([2025])
players = nfl.load_players()[["gsis_id", "pfr_id", "position"]].drop_duplicates(
    "gsis_id"
)
inj = inj.merge(players, on="gsis_id", how="left", suffixes=("", "_player"))
state = _player_snap_state([2025])
print("inj dtypes:")
print(inj[["season", "week", "team", "pfr_id"]].dtypes)
print("state dtypes:")
print(state[["season", "week", "team", "pfr_player_id"]].dtypes)
print("\ninj sample:")
print(inj[["season", "week", "team", "pfr_id"]].head().to_string())
print("\nstate sample:")
print(state[["season", "week", "team", "pfr_player_id"]].head().to_string())
