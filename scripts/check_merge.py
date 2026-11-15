import pandas as pd
from nflscore import data as nfl
from nflscore.build import _player_snap_state

inj = nfl.load_injuries([2025])
players = nfl.load_players()[["gsis_id", "pfr_id", "position"]].drop_duplicates(
    "gsis_id"
)
inj = inj.merge(players, on="gsis_id", how="left", suffixes=("", "_player"))
state = _player_snap_state([2025])
inj = inj.sort_values(["season", "week", "team", "pfr_id"])
state = state.sort_values(["season", "week", "team", "pfr_player_id"])
inj["week"] = pd.to_numeric(inj["week"], errors="coerce").astype("int64")
state["week"] = pd.to_numeric(state["week"], errors="coerce").astype("int64")
try:
    out = pd.merge_asof(
        inj,
        state.rename(columns={"pfr_player_id": "pfr_id", "week": "state_week"}),
        left_on="week",
        right_on="state_week",
        by=["season", "team", "pfr_id"],
        direction="backward",
        allow_exact_matches=False,
    )
    print("ok", out.shape)
except Exception as e:
    print("error", type(e), e)

print("\nTry dropna pfr_id:")
inj2 = inj.dropna(subset=["pfr_id"]).copy()
try:
    out = pd.merge_asof(
        inj2,
        state.rename(columns={"pfr_player_id": "pfr_id", "week": "state_week"}),
        left_on="week",
        right_on="state_week",
        by=["season", "team", "pfr_id"],
        direction="backward",
        allow_exact_matches=False,
    )
    print("ok", out.shape)
except Exception as e:
    print("error", type(e), e)
