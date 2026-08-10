"""Markdown report + equity-curve chart generator.

``generate_report`` takes a ``BacktestResult`` and writes:

1. A markdown file (``output_path``) with:
   - Executive summary (return, Sharpe, drawdown, win rate, trade count).
   - Per-analyst Brier scores table (sorted best to worst).
   - Per-category breakdown table.
   - Equity-curve PNG reference.
   - Weight-tuning recommendations (heuristic, derived from the metrics).

2. An equity-curve PNG alongside the markdown (same stem, ``.png`` extension),
   rendered with matplotlib in the Pythia dark theme:

   - Background: ``#0F172A`` (slate-950).
   - Equity line: ``#D4AF37`` (gold).
   - Grid: subtle white at 10% opacity.
   - Title + axis labels in off-white.

The PNG path is embedded in the markdown as ``![equity curve](<stem>.png)``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt

from .types import BacktestResult

logger = logging.getLogger(__name__)

# --- Pythia visual identity -------------------------------------------------
_BG_COLOR = "#0F172A"       # slate-950 — chart background
_PANEL_COLOR = "#1E293B"    # slate-800 — plot area
_EQUITY_COLOR = "#D4AF37"   # gold — equity curve line
_TEXT_COLOR = "#E2E8F0"     # slate-200 — labels
_GRID_COLOR = "#334155"     # slate-700 — grid lines
_LOSS_MARKER = "#EF4444"    # red-500 — losing trades
_WIN_MARKER = "#22C55E"     # green-500 — winning trades


def generate_report(result: BacktestResult, output_path: Path) -> None:
    """Write a markdown backtest report + equity-curve PNG.

    Parameters
    ----------
    result : BacktestResult
        The aggregated backtest output (from ``Backtester.run``).
    output_path : Path
        Where to write the markdown file. The PNG is written alongside
        with the same stem and a ``.png`` extension. Parent directories
        are created if they don't exist.

    The markdown contains:

    - **Executive Summary** — starting/ending capital, return %, Sharpe,
      max drawdown, win rate, total trades.
    - **Per-Analyst Brier Scores** — table sorted ascending (best first),
      with calibration interpretation.
    - **Per-Category Breakdown** — table with count, win rate, return %,
      Brier, total P&L.
    - **Equity Curve** — embedded PNG reference.
    - **Recommendations** — heuristic weight-tuning suggestions derived
      from the metrics (see ``_build_recommendations``).
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Equity-curve PNG -----------------------------------------------
    png_path = output_path.with_suffix(".png")
    try:
        plot_equity_curve(result.equity_curve, png_path)
    except Exception as exc:  # noqa: BLE001 — chart failure must not crash report
        logger.warning("equity-curve plot failed: %s", exc)
        png_path = output_path  # fall back; the markdown will note the missing chart

    # ---- Markdown body --------------------------------------------------
    lines: list[str] = []
    lines.append("# Pythia Forge — Backtest Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")

    # -- Executive Summary --
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Starting capital | ${result.starting_capital_usd:,.2f} |")
    lines.append(f"| Ending capital | ${result.ending_capital_usd:,.2f} |")
    lines.append(f"| Total return | **{result.total_return_pct:+.2f}%** |")
    lines.append(f"| Sharpe ratio (annualised) | {result.sharpe_ratio:.3f} |")
    lines.append(f"| Max drawdown | {result.max_drawdown_pct:.2f}% |")
    lines.append(f"| Win rate | {result.win_rate * 100:.1f}% |")
    lines.append(f"| Total trades | {result.total_trades} |")
    lines.append("")

    # -- Per-Analyst Brier Scores --
    lines.append("## Per-Analyst Brier Scores")
    lines.append("")
    lines.append(
        "_Lower is better. 0.00 = perfect, "
        "0.25 = uninformative (always 0.5), 1.00 = always wrong._"
    )
    lines.append("")
    lines.append("| Rank | Analyst | Brier score | Calibration |")
    lines.append("| --- | --- | --- | --- |")
    sorted_brier = sorted(result.brier_scores.items(), key=lambda kv: kv[1])
    for rank, (aid, score) in enumerate(sorted_brier, 1):
        cal = _brier_label(score)
        lines.append(f"| {rank} | `{aid}` | {score:.4f} | {cal} |")
    if not sorted_brier:
        lines.append("| — | _no analysts_ | — | — |")
    lines.append("")

    # -- Per-Category Breakdown --
    lines.append("## Per-Category Breakdown")
    lines.append("")
    lines.append("| Category | Trades | Win rate | Return % | Total P&L | Brier |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    # Sort by total P&L descending (best categories first).
    sorted_cats = sorted(
        result.per_category_stats.items(),
        key=lambda kv: kv[1].get("total_pnl_usd", 0.0),
        reverse=True,
    )
    for cat, stats in sorted_cats:
        lines.append(
            f"| {cat} | {stats.get('count', 0)} | "
            f"{stats.get('win_rate', 0.0) * 100:.1f}% | "
            f"{stats.get('return_pct', 0.0):+.2f}% | "
            f"${stats.get('total_pnl_usd', 0.0):+,.2f} | "
            f"{stats.get('brier', 0.0):.4f} |"
        )
    if not sorted_cats:
        lines.append("| _no trades_ | — | — | — | — | — |")
    lines.append("")

    # -- Equity Curve --
    lines.append("## Equity Curve")
    lines.append("")
    if png_path != output_path and png_path.exists():
        png_name = png_path.name
        lines.append(f"![equity curve]({png_name})")
    else:
        lines.append("_Equity-curve chart could not be rendered._")
    lines.append("")

    # -- Recommendations --
    lines.append("## Recommendations")
    lines.append("")
    recs = _build_recommendations(result)
    if recs:
        for r in recs:
            lines.append(f"- {r}")
    else:
        lines.append("- No actionable recommendations — metrics are within healthy ranges.")
    lines.append("")

    # -- Footer --
    lines.append("---")
    lines.append("")
    lines.append("_Generated by `pythia-forge`. See `pythia_forge/report.py` for the renderer._")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report written: %s (png: %s)", output_path, png_path)


def plot_equity_curve(
    equity_curve: list[tuple[datetime, float]],
    output_path: Path,
) -> None:
    """Render the equity curve as a dark-themed PNG with a gold equity line.

    Parameters
    ----------
    equity_curve : list[tuple[datetime, float]]
        From ``BacktestResult.equity_curve``. First entry is the starting
        capital; one entry per settled trade thereafter.
    output_path : Path
        Where to write the PNG. Parent directories are created.

    The chart uses the Pythia visual identity:
    - Background ``#0F172A`` (slate-950).
    - Equity line ``#D4AF37`` (gold), 2.0pt.
    - Filled area under the line at 15% gold opacity.
    - Winning-trade markers in green, losing-trade markers in red.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not equity_curve:
        # Render an empty chart with a "no data" label.
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG_COLOR)
        ax.set_facecolor(_PANEL_COLOR)
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center",
                color=_TEXT_COLOR, fontsize=14, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(output_path, dpi=120, facecolor=_BG_COLOR, bbox_inches="tight")
        plt.close(fig)
        return

    timestamps = [t for t, _ in equity_curve]
    values = [v for _, v in equity_curve]

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=_BG_COLOR)
    ax.set_facecolor(_PANEL_COLOR)

    # Equity line (gold).
    ax.plot(timestamps, values, color=_EQUITY_COLOR, linewidth=2.0,
            marker="o", markersize=4, markerfacecolor=_EQUITY_COLOR,
            markeredgecolor=_EQUITY_COLOR, zorder=3, label="Equity")

    # Filled area under the line (subtle gold tint).
    ax.fill_between(timestamps, values, alpha=0.15, color=_EQUITY_COLOR, zorder=2)

    # Colour-code trade markers by win/loss (skip the first point = starting capital).
    if len(values) >= 2:
        prev = values[0]
        for i in range(1, len(values)):
            color = _WIN_MARKER if values[i] >= prev else _LOSS_MARKER
            ax.scatter(timestamps[i], values[i], color=color, s=30, zorder=4,
                       edgecolors=_BG_COLOR, linewidths=0.5)
            prev = values[i]

    # Starting-capital reference line (dashed, dim).
    if values:
        ax.axhline(y=values[0], color=_TEXT_COLOR, linestyle="--", linewidth=0.8,
                   alpha=0.4, zorder=1, label="Starting capital")

    # Styling.
    ax.set_title("Pythia Forge — Backtest Equity Curve",
                 color=_TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Bankroll (USD)", color=_TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
    ax.grid(True, color=_GRID_COLOR, alpha=0.5, linestyle="-", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)

    # Format y-axis as dollars.
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Date formatting on x-axis.
    fig.autofmt_xdate(rotation=30, ha="right")

    legend = ax.legend(loc="best", facecolor=_PANEL_COLOR, edgecolor=_GRID_COLOR,
                       labelcolor=_TEXT_COLOR, fontsize=9)
    if legend:
        for text in legend.get_texts():
            text.set_color(_TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, facecolor=_BG_COLOR, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Recommendation engine — heuristic weight-tuning suggestions.
# ---------------------------------------------------------------------------


def _brier_label(score: float) -> str:
    """Human-readable calibration label for a Brier score."""
    if score <= 0.10:
        return "excellent"
    if score <= 0.18:
        return "good"
    if score <= 0.25:
        return "marginal (≈ uninformative)"
    if score <= 0.35:
        return "poor — consider down-weighting"
    return "very poor — strongly down-weight"


def _build_recommendations(result: BacktestResult) -> list[str]:
    """Derive heuristic weight-tuning recommendations from the metrics.

    These are starting points, not laws — always validate against
    out-of-sample data. The rules (in priority order):

    1. Per-analyst Brier > 0.30 → recommend down-weighting (weight < 0.5).
    2. Per-analyst Brier < 0.18 → recommend up-weighting (weight > 1.2).
    3. win_rate < 0.45 → raise agreement_threshold (mesh is trading on disagreement).
    4. max_drawdown_pct > 3.0 → halve kelly_fraction.
    5. sharpe_ratio < 0.5 → try consensus method = "trimmed-mean".
    6. total_return_pct < 0 → broad warning.
    7. total_trades < 5 → sample too small for confidence.
    """
    recs: list[str] = []

    # Per-analyst weight recommendations.
    for aid, score in sorted(result.brier_scores.items(), key=lambda kv: kv[1]):
        if score > 0.30:
            recs.append(
                f"**Down-weight `{aid}`** — Brier {score:.3f} > 0.30. "
                f"Set `[consensus.weights] {aid} < 0.5` (or drop it from the mesh)."
            )
        elif score < 0.18:
            recs.append(
                f"**Up-weight `{aid}`** — Brier {score:.3f} < 0.18 (well-calibrated). "
                f"Set `[consensus.weights] {aid} > 1.2`."
            )

    # Win-rate signal.
    if result.total_trades > 0 and result.win_rate < 0.45:
        recs.append(
            f"**Raise `agreement_threshold`** — win rate is "
            f"{result.win_rate * 100:.1f}% (< 45%), suggesting the mesh is "
            f"trading on disagreement. Try 0.70 or 0.75."
        )

    # Drawdown signal.
    if result.max_drawdown_pct > 3.0:
        recs.append(
            f"**Halve `kelly_fraction`** — max drawdown is "
            f"{result.max_drawdown_pct:.2f}% (> 3%), approaching the 5% circuit "
            f"breaker. Quarter-Kelly → eighth-Kelly."
        )

    # Sharpe signal.
    if result.total_trades >= 5 and result.sharpe_ratio < 0.5:
        recs.append(
            f"**Try `method = \"trimmed-mean\"`** — Sharpe is "
            f"{result.sharpe_ratio:.2f} (< 0.5), suggesting outlier analysts "
            f"are dragging the logit-mean. Trimmed-mean is more robust."
        )

    # Negative return warning.
    if result.total_return_pct < 0:
        recs.append(
            f"**Negative return** ({result.total_return_pct:+.2f}%) — the strategy "
            f"lost money on this market set. Re-check analyst Brier scores and "
            f"consider whether the MockLLM heuristic is flattering a specific analyst."
        )

    # Sample-size warning.
    if result.total_trades < 5:
        recs.append(
            f"**Small sample** — only {result.total_trades} trades. "
            f"Metrics are noisy; expand the markets fixture before drawing conclusions."
        )

    return recs


__all__ = ["generate_report", "plot_equity_curve"]
