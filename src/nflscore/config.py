"""Project paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
for _p in (RAW, PROCESSED, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)

SEASONS = list(range(2014, 2027))
SEASON_TYPE = "REG"

# Coverage / pressure data starts in 2018; the main model trains from 2018.
TRAIN_START = 2018

# A small home-field prior used only for naive baselines / sanity checks.
LEAGUE_POINTS = 22.0
HOME_FIELD_POINTS = 1.5

# Market blend / model selection grids
RIDGE_ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
HGB_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.03,
    max_iter=300,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    max_features=0.8,
    random_state=42,
)

# Columns that are identifiers / targets / post-game only.
ID_COLS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "total",
    "result",
    "overtime",
    "old_game_id",
    "gsis",
    "nfl_detail_id",
    "pfr",
    "pff",
    "espn",
    "ftn",
    "stadium_id",
    "stadium",
    "location",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
    "referee",
    "home_win",
    "margin",
    "home_margin",
    "home_total",
    "target_margin",
    "target_total",
    "pred_margin",
    "pred_total",
    "pred_home_score",
    "pred_away_score",
    "pred_home_win_prob",
]

MARKET_COLS = [
    "spread_line",
    "total_line",
    "home_implied",
    "away_implied",
    "market_margin",
    "market_total",
    "market_home_win_prob",
]
