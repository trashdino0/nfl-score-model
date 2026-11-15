"""Paired evaluation of the saved backtest outputs.

Run after ``python -m nflscore.backtest``:
    $env:PYTHONPATH="src"; python scripts/evaluate.py

Writes:
    outputs/evaluation.md    human-readable report
    outputs/evaluation.json  machine-readable summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflscore.config import OUTPUTS  # noqa: E402
from nflscore.evaluation import (  # noqa: E402
    ats_summary,
    brier,
    diebold_mariano,
    log_loss,
    mcnemar,
    over_summary,
    paired_bootstrap_diff,
    reliability_table,
)

MAE = lambda x, axis=None: np.mean(np.abs(x), axis=axis)  # noqa: E731
BOOT = 10_000

MODEL_COLS = {
    "market": dict(
        margin="market_margin",
        total="market_total",
        prob="market_win_prob",
        label="Market closing line",
    ),
    "pure": dict(
        margin="pure_margin",
        total="pure_total",
        prob="pure_win_prob",
        label="Pure stats ridge",
    ),
    "hybrid": dict(
        margin="pred_margin",
        total="pred_total",
        prob="win_prob",
        label="Hybrid market+stats",
    ),
}


def load_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    current = OUTPUTS / "backtest_2026.csv"
    if not current.exists():
        current = OUTPUTS / "backtest_2026_weeks1_8.csv"
    if current.exists():
        frames["2026"] = pd.read_csv(current)
    validation = OUTPUTS / "validation_2021_2025_predictions.csv"
    if validation.exists():
        frames["2021-2025"] = pd.read_csv(validation)
    if not frames:
        raise SystemExit("No backtest outputs found. Run nflscore.backtest first.")
    return frames


def _scores(df: pd.DataFrame, margin_col: str, total_col: str):
    total = df[total_col].astype(float)
    margin = df[margin_col].astype(float)
    return (total + margin) / 2, (total - margin) / 2


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["margin"].notna() & df["total"].notna() & df["home_score"].notna()
    ].copy()


def _accuracy(df: pd.DataFrame, margin_col: str) -> float:
    d = df[df["home_win"].notna()]
    return float(((d[margin_col] > 0).astype(float) == d["home_win"]).mean())


def _score_mae(df: pd.DataFrame, margin_col: str, total_col: str) -> np.ndarray:
    ph, pa = _scores(df, margin_col, total_col)
    actual = pd.concat([df["home_score"], df["away_score"]], ignore_index=True)
    pred = pd.concat([ph, pa], ignore_index=True)
    return (actual.astype(float) - pred.astype(float)).abs().to_numpy()


def headline(df: pd.DataFrame) -> list[dict]:
    rows = []
    for key, cols in MODEL_COLS.items():
        if cols["margin"] not in df.columns:
            continue
        d = _valid(df)
        prob = d[cols["prob"]].astype(float) if cols["prob"] in d.columns else None
        decisive = d["home_win"].notna()
        score_err = _score_mae(d, cols["margin"], cols["total"])
        rows.append(
            {
                "model": key,
                "label": cols["label"],
                "n": int(len(d)),
                "win_acc": _accuracy(d, cols["margin"]),
                "margin_mae": float((d["margin"] - d[cols["margin"]]).abs().mean()),
                "total_mae": float((d["total"] - d[cols["total"]]).abs().mean()),
                "score_mae": float(score_err.mean()),
                "brier": brier(d.loc[decisive, "home_win"], prob[decisive])
                if prob is not None
                else float("nan"),
                "logloss": log_loss(d.loc[decisive, "home_win"], prob[decisive])
                if prob is not None
                else float("nan"),
            }
        )
    return rows


def compare(df: pd.DataFrame, a: str, b: str) -> list[dict]:
    """Paired comparisons of model ``a`` against model ``b``."""
    ca, cb = MODEL_COLS[a], MODEL_COLS[b]
    d = _valid(df)
    rows = []

    for metric in ["margin", "total"]:
        diff = paired_bootstrap_diff(
            (d[metric] - d[ca[metric]]).abs().to_numpy(),
            (d[metric] - d[cb[metric]]).abs().to_numpy(),
            MAE,
            n_boot=BOOT,
        )
        dm = diebold_mariano(
            (d[metric] - d[ca[metric]]).abs().to_numpy(),
            (d[metric] - d[cb[metric]]).abs().to_numpy(),
        )
        rows.append(
            {
                "comparison": f"{a} vs {b}",
                "metric": f"{metric} MAE",
                "diff": diff["diff"],
                "lo": diff["lo"],
                "hi": diff["hi"],
                "p": diff["p"],
                "dm_p": dm["p"],
                "n": diff["n"],
            }
        )

    err_a = _score_mae(d, ca["margin"], ca["total"])
    err_b = _score_mae(d, cb["margin"], cb["total"])
    diff = paired_bootstrap_diff(err_a, err_b, MAE, n_boot=BOOT)
    rows.append(
        {
            "comparison": f"{a} vs {b}",
            "metric": "score MAE",
            "diff": diff["diff"],
            "lo": diff["lo"],
            "hi": diff["hi"],
            "p": diff["p"],
            "dm_p": diebold_mariano(err_a, err_b)["p"],
            "n": diff["n"],
        }
    )

    decisive = d["home_win"].notna()
    pred_a = (d.loc[decisive, ca["margin"]] > 0).astype(float)
    pred_b = (d.loc[decisive, cb["margin"]] > 0).astype(float)
    mc = mcnemar(d.loc[decisive, "home_win"], pred_a, pred_b)
    rows.append(
        {
            "comparison": f"{a} vs {b}",
            "metric": "winner accuracy (pp)",
            "diff": (_accuracy(d, ca["margin"]) - _accuracy(d, cb["margin"])) * 100,
            "lo": float("nan"),
            "hi": float("nan"),
            "p": mc["p"],
            "dm_p": float("nan"),
            "n": int(decisive.sum()),
            "note": f"{mc['a_only']} only-{a} wins, {mc['b_only']} only-{b} wins",
        }
    )

    if ca["prob"] in d.columns and cb["prob"] in d.columns:
        pa = d.loc[decisive, ca["prob"]].astype(float)
        pb = d.loc[decisive, cb["prob"]].astype(float)
        y = d.loc[decisive, "home_win"].astype(float)
        loss_a = (pa - y) ** 2
        loss_b = (pb - y) ** 2
        diff = paired_bootstrap_diff(loss_a, loss_b, np.mean, n_boot=BOOT)
        rows.append(
            {
                "comparison": f"{a} vs {b}",
                "metric": "Brier",
                "diff": diff["diff"],
                "lo": diff["lo"],
                "hi": diff["hi"],
                "p": diff["p"],
                "dm_p": diebold_mariano(loss_a, loss_b)["p"],
                "n": diff["n"],
            }
        )

    return rows


def edge_threshold_table(df: pd.DataFrame) -> list[dict]:
    if "model_edge" not in df.columns:
        return []
    d = df[df["market_margin"].notna()].copy()
    rows = []
    for threshold in [0.0, 0.5, 1.0, 1.5, 2.0]:
        tmp = d.copy()
        tmp["_pick"] = np.where(
            tmp["model_edge"].abs() >= threshold,
            np.where(tmp["model_edge"] > 0, tmp["home_team"], tmp["away_team"]),
            np.nan,
        )
        s = ats_summary(tmp, pick_col="_pick")
        if not s.get("n"):
            continue
        rows.append(
            {
                "threshold": threshold,
                "n": s["n"],
                "wins": s["wins"],
                "losses": s["losses"],
                "pushes": s["pushes"],
                "win_rate": s["win_rate"],
                "win_rate_lo": s["win_rate_lo"],
                "win_rate_hi": s["win_rate_hi"],
                "roi": s["roi"],
                "roi_lo": s["roi_lo"],
                "roi_hi": s["roi_hi"],
            }
        )
    return rows


def calibration_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for key, col in [
        ("blend", "win_prob"),
        ("model", "model_win_prob"),
        ("market", "market_win_prob"),
    ]:
        if col not in df.columns:
            continue
        table = reliability_table(df, col)
        table.insert(0, "model", key)
        rows.extend(table.to_dict("records"))
    return rows


def fmt(x, nd=2, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def pct(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.{nd}f}%"


def md_table(headers, rows, aligns=None):
    if aligns is None:
        aligns = ["---"] * len(headers)
    mapping = {"l": ":---", "c": ":---:", "r": "---:"}
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(mapping[a] for a in aligns) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def verdict(row: dict) -> str:
    if row.get("note"):
        return "detectable" if row["p"] < 0.05 else "no detectable difference"
    if row["diff"] == 0 and row["lo"] == 0 and row["hi"] == 0:
        return "identical by design"
    if not np.isfinite(row["lo"]) or not np.isfinite(row["hi"]):
        return "—"
    if row["p"] < 0.05 and (row["lo"] > 0 or row["hi"] < 0):
        return "detectable"
    return "no detectable difference"


def build_report() -> tuple[str, dict]:
    frames = load_frames()
    payload = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"), "samples": {}}
    lines = [
        "# Evaluation — paired, out-of-sample",
        "",
        f"*Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
        "Bootstrap intervals resample games, so paired predictions keep their dependence.*",
        "",
        "**How to read this:** a negative Δ for MAE/Brier means the first model is "
        "better. A result is only called detectable when the 95% interval excludes "
        "zero and the paired test agrees. Everything else is noise.",
        "",
    ]

    for sample, df in frames.items():
        d = _valid(df)
        payload["samples"][sample] = {
            "n": int(len(d)),
            "headline": headline(df),
            "comparisons": [],
        }
        lines += [f"## {sample} ({len(d)} games)", ""]

        head = headline(df)
        lines += [
            md_table(
                ["Model", "n", "Winner acc.", "Margin MAE", "Total MAE", "Score MAE", "Brier", "Log loss"],
                [
                    [
                        row["label"],
                        row["n"],
                        pct(row["win_acc"]),
                        fmt(row["margin_mae"]),
                        fmt(row["total_mae"]),
                        fmt(row["score_mae"]),
                        fmt(row["brier"], 3),
                        fmt(row["logloss"], 3),
                    ]
                    for row in head
                ],
                ["l", "r", "r", "r", "r", "r", "r", "r"],
            ),
            "",
        ]

        comp_rows = []
        for a, b in [("hybrid", "market"), ("hybrid", "pure"), ("pure", "market")]:
            if MODEL_COLS[a]["margin"] not in df.columns:
                continue
            if MODEL_COLS[b]["margin"] not in df.columns:
                continue
            for row in compare(df, a, b):
                payload["samples"][sample]["comparisons"].append(row)
                ci = (
                    "—"
                    if not np.isfinite(row["lo"])
                    else f"[{row['lo']:+.3f}, {row['hi']:+.3f}]"
                )
                comp_rows.append(
                    [
                        f"{a} vs {b}",
                        row["metric"],
                        fmt(row["diff"], 3, signed=True),
                        ci,
                        fmt(row["p"], 3),
                        verdict(row),
                    ]
                )
        lines += [
            "### Paired comparisons",
            "",
            md_table(
                ["Comparison", "Metric", "Δ (A−B)", "95% CI", "p", "Verdict"],
                comp_rows,
                ["l", "l", "r", "r", "r", "l"],
            ),
            "",
        ]

        if "ats_pick" in df.columns:
            ats = ats_summary(df)
            play = ats_summary(df, pick_col="ats_play") if "ats_play" in df.columns else None
            ou = over_summary(df)
            payload["samples"][sample]["betting"] = {"ats_lean": ats, "ats_play": play, "ou": ou}
            lines += [
                "### Betting diagnostics (flat 1-unit bets at −110)",
                "",
                md_table(
                    ["Strategy", "n", "W–L–P", "Win rate (95% CI)", "ROI (95% CI)"],
                    [
                        [
                            "ATS — every lean",
                            ats.get("n", 0),
                            f"{ats.get('wins', 0)}–{ats.get('losses', 0)}–{ats.get('pushes', 0)}",
                            f"{pct(ats.get('win_rate'))} "
                            f"[{pct(ats.get('win_rate_lo'))}, {pct(ats.get('win_rate_hi'))}]",
                            f"{pct(ats.get('roi'))} "
                            f"[{pct(ats.get('roi_lo'))}, {pct(ats.get('roi_hi'))}]",
                        ],
                        [
                            "ATS — plays only (edge ≥ 1)",
                            (play or {}).get("n", 0),
                            f"{(play or {}).get('wins', 0)}–{(play or {}).get('losses', 0)}–{(play or {}).get('pushes', 0)}",
                            f"{pct((play or {}).get('win_rate'))} "
                            f"[{pct((play or {}).get('win_rate_lo'))}, {pct((play or {}).get('win_rate_hi'))}]",
                            f"{pct((play or {}).get('roi'))} "
                            f"[{pct((play or {}).get('roi_lo'))}, {pct((play or {}).get('roi_hi'))}]",
                        ],
                    ],
                    ["l", "r", "r", "r", "r"],
                ),
                "",
            ]
            if ou.get("n"):
                lines += [
                    f"Over/under diagnostic (not published as a pick): "
                    f"{ou['wins']}–{ou['losses']}–{ou['pushes']} = "
                    f"{pct(ou['win_rate'])}, ROI {pct(ou['roi'])} "
                    f"[{pct(ou['roi_lo'])}, {pct(ou['roi_hi'])}] at −110.",
                    "",
                ]
            if "model_edge" in df.columns:
                lines += [
                    "**ATS by model edge:**",
                    "",
                    md_table(
                        ["Min |edge|", "n", "W–L–P", "Win rate (95% CI)", "ROI (95% CI)"],
                        [
                            [
                                f"{row['threshold']:g}",
                                row["n"],
                                f"{row['wins']}–{row['losses']}–{row['pushes']}",
                                f"{pct(row['win_rate'])} "
                                f"[{pct(row['win_rate_lo'])}, {pct(row['win_rate_hi'])}]",
                                f"{pct(row['roi'])} "
                                f"[{pct(row['roi_lo'])}, {pct(row['roi_hi'])}]",
                            ]
                            for row in edge_threshold_table(df)
                        ],
                        ["r", "r", "r", "r", "r"],
                    ),
                    "",
                ]

        cal = calibration_rows(df)
        payload["samples"][sample]["calibration"] = cal
        if cal:
            lines += [
                "### Calibration",
                "",
                md_table(
                    ["Probability", "Bin", "n", "Predicted", "Actual", "95% CI"],
                    [
                        [
                            row["model"],
                            row["bin"],
                            int(row["n"]),
                            pct(row["predicted"]),
                            pct(row["actual"]),
                            f"[{pct(row['lo'])}, {pct(row['hi'])}]",
                        ]
                        for row in cal
                    ],
                    ["l", "l", "r", "r", "r", "l"],
                ),
                "",
            ]

        if "model_edge" in df.columns:
            edge = df["model_edge"].dropna().abs()
            payload["samples"][sample]["edge"] = {
                "mean_abs": float(edge.mean()),
                "median_abs": float(edge.median()),
                "p90_abs": float(edge.quantile(0.90)),
                "share_over_1": float((edge > 1).mean()),
                "share_over_2": float((edge > 2).mean()),
            }
            lines += [
                "### Model edge over the market",
                "",
                f"Mean absolute edge **{edge.mean():.2f} pts** (median "
                f"{edge.median():.2f}, p90 {edge.quantile(0.90):.2f}). "
                f"{pct((edge > 1).mean())} of games have an edge above 1 point; "
                f"{pct((edge > 2).mean())} above 2.",
                "",
            ]

    lines += [
        "## What this means",
        "",
        "- The hybrid model is **not detectably better than the closing line** on "
        "margin, score, or Brier. Its total is the market total by construction "
        "(`total_shrink = 0`), so the total comparison is an identity, not a result.",
        "- Winner accuracy is the easiest metric to over-read. A two- or three-game "
        "difference is a McNemar coin flip, not evidence.",
        "- ATS/over ROI intervals are wide enough to contain any plausible edge. "
        "Do not treat a positive point estimate as a strategy.",
        "- The useful product is the calibrated score distribution and the honest "
        "market comparison, not a claim of a betting edge.",
        "",
    ]
    return "\n".join(lines), payload


def _clean(obj):
    """Replace numpy scalars and non-finite floats before JSON serialisation."""
    if isinstance(obj, dict):
        return {key: _clean(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def main() -> None:
    text, payload = build_report()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "evaluation.md").write_text(text, encoding="utf-8")
    (OUTPUTS / "evaluation.json").write_text(
        json.dumps(_clean(payload), indent=2), encoding="utf-8"
    )
    print(text)
    print(f"\nWrote {OUTPUTS / 'evaluation.md'}")
    print(f"Wrote {OUTPUTS / 'evaluation.json'}")


if __name__ == "__main__":
    main()
