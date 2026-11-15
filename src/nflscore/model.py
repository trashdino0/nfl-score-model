"""Model training, prediction, and evaluation.

Design
------
The closing market is a strong baseline, so the primary model is a
market-anchored residual model: ridge regressions predict the *difference*
between the actual result and the closing line, and that residual is shrunk
toward zero before being added back to the market.  A pure stats-only model is
kept for comparison and for weeks where a market line is unavailable.

Margin and total are modeled separately, then converted to scores:
    home = total / 2 + margin / 2
    away = total / 2 - margin / 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TRAIN_START
from .distribution import simulate_scores

# Curated matchup labels: (feature stem, readable label).
MATCHUP_LABELS = {
    "epa": "Overall EPA",
    "pass_epa": "Pass EPA",
    "rush_epa": "Rush EPA",
    "qb_vs_pass": "QB vs pass defense",
    "qb_cpoe_vs_coverage": "QB CPOE vs coverage",
    "wr1_vs_coverage": "WR1 vs coverage",
    "wr_corps_vs_coverage": "WR corps vs coverage",
    "rb1_vs_run": "RB1 vs run defense",
    "pass_rush_vs_protection": "Pass rush vs protection",
    "turnovers": "Turnover margin",
    "red_zone": "Red-zone efficiency",
    "points_drive": "Points per drive",
    "pace": "Pace",
    "third_down": "Third down",
    "success": "Success rate",
    "explosive_pass": "Explosive pass",
    "explosive_rush": "Explosive rush",
    "pass_success": "Pass success",
    "rush_success": "Rush success",
    "pass_yards": "Pass yards",
    "rush_yards": "Rush yards",
    "plays_drive": "Plays per drive",
}


def _dedupe(cols):
    return list(dict.fromkeys([c for c in cols if c]))


def _matchup_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.endswith("_net")]
    for lab in MATCHUP_LABELS:
        for side in ["home_adv", "away_adv"]:
            c = f"{lab}_{side}"
            if c in df.columns:
                cols.append(c)
    cols += [c for c in df.columns if c.endswith(("_roll8_diff", "_prev_diff"))]
    return _dedupe(cols)


def _team_cols(df: pd.DataFrame) -> list[str]:
    return _dedupe(
        [
            c
            for c in df.columns
            if c.endswith("_diff") and not c.startswith(("inj_", "qb_", "start_qb_"))
        ]
    )


def _qb_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("start_qb_")]
    cols += [
        c
        for c in [
            "home_qb_out",
            "away_qb_out",
            "qb_out_diff",
            "qb_questionable_diff",
            "home_qb_change",
            "away_qb_change",
            "qb_change_diff",
            "home_start_qb_is_new",
            "away_start_qb_is_new",
            "start_qb_new_diff",
            "home_qb_projected",
            "away_qb_projected",
        ]
        if c in df.columns
    ]
    return _dedupe(cols)


def _injury_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("inj_") and c.endswith("_diff")]
    cols += [c for c in ["inj_lost_off_sum", "inj_lost_def_sum"] if c in df.columns]
    cols += [
        c
        for c in df.columns
        if c.endswith(("_out_diff", "_questionable_diff"))
        and c.split("_")[0] in {"wr", "rb", "te"}
    ]
    return _dedupe(cols)


def _situational_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in [
            "rest_diff",
            "short_week_diff",
            "is_dome",
            "is_turf",
            "home_altitude",
            "div_game",
            "wind",
            "temp",
        ]
        if c in df.columns
    ]


def margin_features(df: pd.DataFrame) -> list[str]:
    """Features for the margin-residual model (matchups + QB + injuries).

    Only directional injury features belong here; game-level injury *sums* are
    kept for the total model, where they describe the overall game environment.
    """
    inj = [c for c in _injury_cols(df) if not c.endswith("_sum")]
    return _dedupe(_matchup_cols(df) + _qb_cols(df) + inj)


def total_features(df: pd.DataFrame) -> list[str]:
    """Features for the total-residual model (adds team, pace, situation)."""
    return _dedupe(
        _matchup_cols(df)
        + _team_cols(df)
        + _qb_cols(df)
        + _injury_cols(df)
        + _situational_cols(df)
        + (["elo_diff"] if "elo_diff" in df.columns else [])
    )


def pure_features(df: pd.DataFrame) -> list[str]:
    """Stats-only feature set: no market columns."""
    return _dedupe(total_features(df) + [c for c in df.columns if c.endswith("_sum")])


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _usable_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].nunique(dropna=True) > 1]


def _safe_market(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise market columns without inventing a closing line.

    A missing spread is *not* filled with the naive baseline.  The caller must
    decide between the market anchor and the pure-model fallback, and label
    which one it used.  A no-vig probability may be derived from a real spread
    (that is still market information); it is never derived from a model.
    """
    out = df.copy()
    for col in ["market_margin", "market_total", "market_home_win_prob"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    missing = out["market_home_win_prob"].isna() & out["market_margin"].notna()
    if missing.any():
        sigma = 13.5
        out.loc[missing, "market_home_win_prob"] = [
            0.5 * (1 + erf((m / sigma) / sqrt(2)))
            for m in out.loc[missing, "market_margin"]
        ]
    return out


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def prob_from_margin(margin, sigma: float) -> np.ndarray:
    """Normal-approximation win probability (ties get no special credit)."""
    z = np.asarray(margin, dtype=float) / sigma
    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in z])



@dataclass
class ModelConfig:
    train_start: int = TRAIN_START
    margin_alpha: float = 100.0
    total_alpha: float = 100.0
    margin_shrink: float = 0.20
    # The total model has not beaten the closing total in any independent
    # sample.  The default is therefore no adjustment; set this above zero only
    # if a total-specific rebuild clears the evaluation harness out of sample.
    total_shrink: float = 0.0
    min_training_rows: int = 200
    # Out-of-fold residuals drive the score distribution, the win-probability
    # scale, and the market/model blend weight.
    cv_folds: int = 5
    # Games with a model edge smaller than this are not labelled as ATS plays.
    ats_min_edge: float = 1.0
    n_sim_draws: int = 4000
    sim_seed: int = 20261114


@dataclass
class NFLModel:
    """Market-anchored ridge model with a stats-only comparison model."""

    config: ModelConfig = field(default_factory=ModelConfig)
    margin_model_: Pipeline | None = None
    total_model_: Pipeline | None = None
    pure_margin_model_: Pipeline | None = None
    pure_total_model_: Pipeline | None = None
    margin_features_: list[str] = field(default_factory=list)
    total_features_: list[str] = field(default_factory=list)
    pure_features_: list[str] = field(default_factory=list)
    margin_sigma_: float = 13.5
    pure_margin_sigma_: float = 15.0
    train_rows_: int = 0
    resid_pool_: np.ndarray | None = None
    pure_resid_pool_: np.ndarray | None = None
    win_prob_model_weight_: float = 0.5
    blend_n_: int = 0
    blend_loss_: float = float("nan")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, games: pd.DataFrame, train_start: int | None = None) -> NFLModel:
        start = self.config.train_start if train_start is None else train_start
        train = games[
            (games["season"] >= start)
            & games["home_score"].notna()
            & games["margin"].notna()
            & games["total"].notna()
        ].copy()
        train = _safe_market(train)
        train = train.dropna(subset=["market_margin", "market_total"])
        if len(train) < self.config.min_training_rows:
            raise ValueError(f"Only {len(train)} training rows; need more data")

        mf = _usable_features(train, margin_features(train))
        tf = _usable_features(train, total_features(train))
        pf = _usable_features(train, pure_features(train))
        if not mf or not tf or not pf:
            raise ValueError("No usable model features")

        self.margin_features_ = mf
        self.total_features_ = tf
        self.pure_features_ = pf

        train["resid_margin"] = train["margin"] - train["market_margin"]
        train["resid_total"] = train["total"] - train["market_total"]

        self.margin_model_ = make_ridge(self.config.margin_alpha)
        self.margin_model_.fit(train[mf], train["resid_margin"])

        self.total_model_ = make_ridge(self.config.total_alpha)
        self.total_model_.fit(train[tf], train["resid_total"])

        # Stats-only comparison model (no market anchor).
        self.pure_margin_model_ = make_ridge(self.config.margin_alpha)
        self.pure_margin_model_.fit(train[pf], train["margin"])

        self.pure_total_model_ = make_ridge(self.config.total_alpha)
        self.pure_total_model_.fit(train[pf], train["total"])

        # Out-of-fold predictions.  In-sample residuals understate forecast
        # error, which would make score intervals too narrow and the win
        # probabilities overconfident.
        oof = self._out_of_fold(train, mf, tf, pf)

        err_margin = train["margin"].to_numpy(dtype=float) - oof["pred_margin"]
        err_total = train["total"].to_numpy(dtype=float) - oof["pred_total"]
        self.margin_sigma_ = float(max(10.0, np.std(err_margin, ddof=1)))
        self.pure_margin_sigma_ = float(
            max(
                10.0,
                np.std(
                    train["margin"].to_numpy(dtype=float) - oof["pure_margin"],
                    ddof=1,
                ),
            )
        )
        # Residuals are noise around the point forecast.  Keep any non-zero
        # sample mean out of the bootstrap, otherwise the simulation silently
        # re-introduces a bias the point forecast already carries.
        self.resid_pool_ = np.column_stack([err_margin, err_total])
        self.resid_pool_ -= self.resid_pool_.mean(axis=0)
        self.pure_resid_pool_ = np.column_stack(
            [
                train["margin"].to_numpy(dtype=float) - oof["pure_margin"],
                train["total"].to_numpy(dtype=float) - oof["pure_total"],
            ]
        )
        self.pure_resid_pool_ -= self.pure_resid_pool_.mean(axis=0)

        self._fit_win_prob_blend(train, oof)
        self.train_rows_ = len(train)
        return self

    def _out_of_fold(
        self, train: pd.DataFrame, mf: list[str], tf: list[str], pf: list[str]
    ) -> dict[str, np.ndarray]:
        """Cross-validated predictions for every training row."""
        n = len(train)
        folds = min(self.config.cv_folds, max(2, n // 100))
        kfold = KFold(n_splits=folds, shuffle=True, random_state=self.config.sim_seed)
        pred = {
            "pred_margin": np.full(n, np.nan),
            "pred_total": np.full(n, np.nan),
            "pure_margin": np.full(n, np.nan),
            "pure_total": np.full(n, np.nan),
        }
        for tr_idx, te_idx in kfold.split(train):
            tr = train.iloc[tr_idx]
            te = train.iloc[te_idx]

            margin_model = make_ridge(self.config.margin_alpha)
            margin_model.fit(tr[mf], tr["resid_margin"])
            total_model = make_ridge(self.config.total_alpha)
            total_model.fit(tr[tf], tr["resid_total"])
            pred["pred_margin"][te_idx] = (
                te["market_margin"].to_numpy(dtype=float)
                + self.config.margin_shrink * margin_model.predict(te[mf])
            )
            pred["pred_total"][te_idx] = (
                te["market_total"].to_numpy(dtype=float)
                + self.config.total_shrink * total_model.predict(te[tf])
            )

            pure_margin = make_ridge(self.config.margin_alpha)
            pure_margin.fit(tr[pf], tr["margin"])
            pred["pure_margin"][te_idx] = pure_margin.predict(te[pf])

            pure_total = make_ridge(self.config.total_alpha)
            pure_total.fit(tr[pf], tr["total"])
            pred["pure_total"][te_idx] = pure_total.predict(te[pf])
        return pred

    def _fit_win_prob_blend(
        self, train: pd.DataFrame, oof: dict[str, np.ndarray]
    ) -> None:
        """Choose the market/model weight in log-odds space on out-of-fold data.

        A 50/50 blend is a guess.  This picks the weight that minimises log
        loss on inner-fold predictions, bounded to [0, 1] so the model can
        never get more weight than a pure model or less than none.
        """
        margin = train["margin"].to_numpy(dtype=float)
        y = (margin > 0).astype(float)
        market = train["market_home_win_prob"].to_numpy(dtype=float)
        model = prob_from_margin(oof["pred_margin"], self.margin_sigma_)
        mask = (margin != 0) & np.isfinite(market) & np.isfinite(model)
        self.blend_n_ = int(mask.sum())
        if self.blend_n_ < 100:
            self.win_prob_model_weight_ = 0.5
            return

        market_logit = _logit(market[mask])
        model_logit = _logit(model[mask])
        y = y[mask]
        best_w, best_loss = 0.5, float("inf")
        for weight in np.linspace(0.0, 1.0, 101):
            p = np.clip(
                _expit((1 - weight) * market_logit + weight * model_logit),
                1e-9,
                1 - 1e-9,
            )
            loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if loss < best_loss:
                best_w, best_loss = float(weight), loss
        self.win_prob_model_weight_ = best_w
        self.blend_loss_ = best_loss

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        if self.margin_model_ is None or self.total_model_ is None:
            raise RuntimeError("Call fit() before predict()")
        if self.resid_pool_ is None or self.pure_resid_pool_ is None:
            raise RuntimeError("Call fit() before predict()")
        out = _safe_market(games).copy()

        out["pure_margin"] = self.pure_margin_model_.predict(out[self.pure_features_])
        out["pure_total"] = self.pure_total_model_.predict(out[self.pure_features_])
        out["pure_home_score"] = (out["pure_total"] + out["pure_margin"]) / 2
        out["pure_away_score"] = (out["pure_total"] - out["pure_margin"]) / 2

        has_line = out["market_margin"].notna() & out["market_total"].notna()
        out["line_source"] = np.where(has_line, "market", "pure_fallback")

        resid_m = self.margin_model_.predict(out[self.margin_features_])
        resid_t = self.total_model_.predict(out[self.total_features_])

        out["resid_margin_pred"] = resid_m
        out["resid_total_pred"] = resid_t

        # With a line, nudge the close by the residual model.  Without one, use
        # the pure model directly: a residual trained against market lines is
        # not calibrated for a different anchor.
        out["pred_margin"] = np.where(
            has_line,
            out["market_margin"] + self.config.margin_shrink * resid_m,
            out["pure_margin"],
        )
        out["pred_total"] = np.where(
            has_line,
            out["market_total"] + self.config.total_shrink * resid_t,
            out["pure_total"],
        )
        out["pred_home_score"] = (out["pred_total"] + out["pred_margin"]) / 2
        out["pred_away_score"] = (out["pred_total"] - out["pred_margin"]) / 2

        sim = simulate_scores(
            out["pred_margin"].to_numpy(dtype=float),
            out["pred_total"].to_numpy(dtype=float),
            self.resid_pool_,
            fallback_pool=self.pure_resid_pool_,
            use_fallback=(~has_line).to_numpy(),
            market_margin=out["market_margin"].to_numpy(dtype=float),
            market_total=out["market_total"].to_numpy(dtype=float),
            n_draws=self.config.n_sim_draws,
            seed=self.config.sim_seed,
        )
        out["sim_win_prob"] = sim.home_win_prob
        out["cover_prob"] = sim.cover_prob
        out["push_prob"] = sim.push_prob
        out["over_prob"] = sim.over_prob
        out["total_push_prob"] = sim.total_push_prob
        out["pred_home_score_low"] = sim.home_low
        out["pred_home_score_med"] = sim.home_med
        out["pred_home_score_high"] = sim.home_high
        out["pred_away_score_low"] = sim.away_low
        out["pred_away_score_med"] = sim.away_med
        out["pred_away_score_high"] = sim.away_high
        out["margin_sd"] = sim.margin_sd
        out["total_sd"] = sim.total_sd

        # Win probability: model scale from out-of-fold error, blended with the
        # no-vig market at the weight chosen on out-of-fold log loss.  A no-line
        # game uses the pure model's wider error scale, not the hybrid's.
        model_prob = np.where(
            has_line,
            prob_from_margin(out["pred_margin"], self.margin_sigma_),
            prob_from_margin(out["pred_margin"], self.pure_margin_sigma_),
        )
        out["model_win_prob"] = model_prob
        out["market_win_prob"] = out["market_home_win_prob"]
        weight = self.win_prob_model_weight_
        blended = _expit(
            (1 - weight) * _logit(out["market_win_prob"].to_numpy(dtype=float))
            + weight * _logit(out["model_win_prob"].to_numpy(dtype=float))
        )
        out["win_prob"] = np.where(
            out["market_win_prob"].notna(), blended, out["model_win_prob"]
        )
        out["win_prob_source"] = np.where(
            out["market_win_prob"].notna(), "market_blend", "model_only"
        )

        out["pred_home_win"] = (out["pred_margin"] > 0).astype(float)
        out["pred_winner"] = np.where(
            out["pred_home_win"] == 1, out["home_team"], out["away_team"]
        )
        out["pick_win_prob"] = np.where(
            out["pred_home_win"] == 1,
            out["model_win_prob"],
            1 - out["model_win_prob"],
        )
        out["model_confidence"] = np.maximum(
            out["pick_win_prob"], 1 - out["pick_win_prob"]
        )

        out["model_edge"] = out["pred_margin"] - out["market_margin"]
        out["ats_pick"] = np.where(
            out["model_edge"].isna(),
            np.nan,
            np.where(
                out["model_edge"] > 0, out["home_team"], out["away_team"]
            ),
        )
        out["ats_play"] = np.where(
            out["model_edge"].abs() >= self.config.ats_min_edge,
            out["ats_pick"],
            None,
        )
        out["pick_cover_prob"] = np.where(
            out["cover_prob"].isna(),
            np.nan,
            np.where(
                out["ats_pick"] == out["home_team"],
                out["cover_prob"],
                1 - out["cover_prob"],
            ),
        )
        # Deliberately no ou_pick: the total model has no demonstrated edge, so
        # the over probability is a diagnostic, not a recommendation.
        return out

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    def margin_coefficients(self) -> pd.Series:
        if self.margin_model_ is None:
            return pd.Series(dtype=float)
        coef = self.margin_model_.named_steps["ridge"].coef_
        return pd.Series(coef, index=self.margin_features_).sort_values()

    def total_coefficients(self) -> pd.Series:
        if self.total_model_ is None:
            return pd.Series(dtype=float)
        coef = self.total_model_.named_steps["ridge"].coef_
        return pd.Series(coef, index=self.total_features_).sort_values()

    def margin_contributions(self, games: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to the predicted margin, in points.

        Contribution = standardized feature value * ridge coefficient.  This is
        the model's additive adjustment to the market margin before shrinkage.
        """
        if self.margin_model_ is None:
            raise RuntimeError("Call fit() before margin_contributions()")
        out = _safe_market(games)
        X = out[self.margin_features_]
        impute = self.margin_model_.named_steps["impute"]
        scale = self.margin_model_.named_steps["scale"]
        coef = self.margin_model_.named_steps["ridge"].coef_
        Xs = scale.transform(impute.transform(X))
        return pd.DataFrame(Xs * coef, columns=self.margin_features_, index=out.index)


_PRETTY_STEMS = {
    "off_epa_play": "Offensive EPA/play",
    "def_epa_play": "Defensive EPA/play",
    "off_pass_epa_db": "Offensive pass EPA/DB",
    "def_pass_epa_db": "Defensive pass EPA/DB",
    "off_rush_epa_carry": "Offensive rush EPA/carry",
    "def_rush_epa_carry": "Defensive rush EPA/carry",
    "off_success_rate": "Offensive success rate",
    "def_success_rate": "Defensive success rate",
    "off_pass_success": "Offensive pass success",
    "def_pass_success": "Defensive pass success",
    "off_rush_success": "Offensive rush success",
    "def_rush_success": "Defensive rush success",
    "off_points_per_drive": "Offensive points/drive",
    "def_points_per_drive": "Defensive points/drive",
    "off_plays_per_drive": "Offensive plays/drive",
    "def_plays_per_drive": "Defensive plays/drive",
    "off_seconds_per_play": "Offensive seconds/play",
    "def_seconds_per_play": "Defensive seconds/play",
    "off_red_zone_td_rate": "Offensive red-zone TD rate",
    "def_red_zone_td_rate": "Defensive red-zone TD rate",
    "off_turnover_rate": "Offensive turnover rate",
    "def_takeaway_rate": "Defensive takeaway rate",
    "off_sack_rate": "Offensive sack rate",
    "def_sack_rate": "Defensive sack rate",
    "off_qb_hit_rate": "Offensive QB hit rate",
    "def_qb_hit_rate": "Defensive QB hit rate",
    "off_pass_yards_play": "Offensive pass yards/play",
    "def_pass_yards_play": "Defensive pass yards/play",
    "off_rush_yards_play": "Offensive rush yards/play",
    "def_rush_yards_play": "Defensive rush yards/play",
    "points_for": "Points scored",
    "points_allowed": "Points allowed",
    "pass_yards": "Pass yards",
    "rush_yards": "Rush yards",
    "pass_success": "Pass success",
    "rush_success": "Rush success",
    "success": "Success rate",
    "third_down": "Third down",
    "sacks": "Sacks",
    "pressure_rate": "Pressure rate",
    "coverage_passer_rating": "Coverage passer rating",
    "coverage_yards_per_target": "Coverage yards/target",
    "qb_epa_db": "QB EPA/DB",
    "qb_cpoe": "QB CPOE",
    "rb1_yards_vs_run": "RB1 yards vs run defense",
    "rb1_rush_epa_carry": "RB1 rush EPA/carry",
    "wr1_epa_target": "WR1 EPA/target",
    "wr_corps_epa_target": "WR corps EPA/target",
}


def _pretty_stem(stem: str) -> str:
    return _PRETTY_STEMS.get(stem, stem.replace("_", " ").title())


def feature_label(name: str) -> str:
    """Human-readable label for a model feature."""
    if name.endswith("_net"):
        stem = name[: -len("_net")]
        return MATCHUP_LABELS.get(stem, _pretty_stem(stem))
    if name.endswith(("_home_adv", "_away_adv")):
        for suffix, prefix in [("_home_adv", "Home"), ("_away_adv", "Away")]:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                label = MATCHUP_LABELS.get(stem, _pretty_stem(stem))
                opp = "away" if prefix == "Home" else "home"
                return f"{prefix} {label.lower()} vs {opp} defense"
    direct = {
        "start_qb_epa_diff": "Starting QB EPA edge",
        "start_qb_cpoe_diff": "Starting QB CPOE edge",
        "start_qb_int_diff": "Starting QB interception edge",
        "start_qb_sack_diff": "Starting QB sack edge",
        "start_qb_attempts_diff": "Starting QB workload edge",
        "start_qb_career_diff": "Starting QB experience edge",
        "start_qb_new_diff": "Starting QB inexperience edge",
        "qb_out_diff": "QB availability edge",
        "qb_change_diff": "QB continuity edge",
        "qb_questionable_diff": "QB questionable edge",
        "inj_lost_qb_diff": "QB injury impact edge",
        "inj_lost_wr_diff": "WR injury impact edge",
        "inj_lost_rb_diff": "RB injury impact edge",
        "inj_lost_te_diff": "TE injury impact edge",
        "inj_lost_ol_diff": "OL injury impact edge",
        "inj_lost_dl_diff": "DL injury impact edge",
        "inj_lost_lb_diff": "LB injury impact edge",
        "inj_lost_db_diff": "DB injury impact edge",
        "inj_out_count_diff": "Out-count edge",
        "inj_questionable_count_diff": "Questionable-count edge",
        "inj_doubtful_count_diff": "Doubtful-count edge",
        "home_qb_out": "Home QB out",
        "away_qb_out": "Away QB out",
        "home_qb_change": "Home QB change",
        "away_qb_change": "Away QB change",
        "home_qb_projected": "Home projected QB",
        "away_qb_projected": "Away projected QB",
        "inj_lost_off_diff": "Offensive injury edge",
        "inj_lost_def_diff": "Defensive injury edge",
        "inj_lost_skill_diff": "Skill-position injury edge",
        "inj_lost_off_sum": "Total offensive injuries",
        "inj_lost_def_sum": "Total defensive injuries",
        "wr_out_diff": "WR availability edge",
        "rb_out_diff": "RB availability edge",
        "te_out_diff": "TE availability edge",
        "wr_questionable_diff": "WR questionable edge",
        "rb_questionable_diff": "RB questionable edge",
        "te_questionable_diff": "TE questionable edge",
        "elo_diff": "Elo edge",
    }
    if name in direct:
        return direct[name]
    if name.endswith("_roll8_diff"):
        stem = name[: -len("_roll8_diff")]
        return f"{_pretty_stem(stem)} (8-game) edge"
    if name.endswith("_prev_diff"):
        stem = name[: -len("_prev_diff")]
        return f"{_pretty_stem(stem)} (prior season) edge"
    if name.endswith("_diff"):
        stem = name[: -len("_diff")]
        return f"{_pretty_stem(stem)} edge"
    return name.replace("_", " ").title()


def add_market_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Add market and naive baseline score predictions for comparison."""
    out = _safe_market(df).copy()
    out["market_home_score"] = (out["market_total"] + out["market_margin"]) / 2
    out["market_away_score"] = (out["market_total"] - out["market_margin"]) / 2
    out["naive_home_score"] = (
        out.get("naive_total", out["market_total"]) + out.get("naive_margin", 0)
    ) / 2
    out["naive_away_score"] = (
        out.get("naive_total", out["market_total"]) - out.get("naive_margin", 0)
    ) / 2
    return out
