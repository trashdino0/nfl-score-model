import pandas as pd
from nflscore import data as nfl
from nflscore.build import _player_snap_state
from nflscore.config import SEASONS

inj = nfl.load_injuries(SEASONS)
players = nfl.load_players()[["gsis_id", "pfr_id", "position"]].drop_duplicates(
    "gsis_id"
)
inj = inj.merge(players, on="gsis_id", how="left", suffixes=("", "_player"))
print("inj dtypes:", inj[["season", "week", "team", "pfr_id"]].dtypes.to_dict())
print("null pfr_id by season:")
print(inj.assign(null=inj.pfr_id.isna()).groupby("season").null.sum().to_string())
print("season dtype by season:", inj.groupby("season").season.dtype)
print(inj[["season", "week", "team", "pfr_id"]].head(20).to_string())

state = _player_snap_state(SEASONS)
print(
    "\nstate dtypes:",
    state[["season", "week", "team", "pfr_player_id"]].dtypes.to_dict(),
)
print("state seasons:", state.season.dtype, state.season.min(), state.season.max())
print("state null pfr:", state.pfr_player_id.isna().sum())
