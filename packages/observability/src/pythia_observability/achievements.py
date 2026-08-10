"""Achievements evaluator — wraps `icohangar-ops/achievements` config format.

The upstream `achievements` repo defines a TOML schema for milestone configs:

    [[achievement]]
    id = "first_trade"
    name = "First Blood"
    description = "Place the first trade."
    condition = { type = "trade_count", op = ">=", value = 1 }

We re-use that exact schema (no transformation needed) and add the *evaluation*
layer: given an `AuditLogReader`, walk every achievement, check its condition
against the current stats, and return a list of `Achievement` objects with
`unlocked_at` populated (or `None` if still locked).

Each condition type is a standalone function `eval_<type>(stats, condition)
-> tuple[bool, Any]` returning (unlocked?, unlocked_value). This makes it
trivial to add new condition types — drop in another function + dispatch entry.

# VERIFY: the upstream `icohangar-ops/achievements` API for *emitting* an
# unlocked achievement (badge / webhook / event bus) is not yet pinned. We
# expose `AchievementsEvaluator.evaluate()` which returns the unlocked list;
# the executor / orchestrator is responsible for forwarding to the upstream's
# emission point. See VENDOR_COMMIT.txt.
"""

from __future__ import annotations

# tomllib is stdlib on Python 3.11+; fallback to tomli on older Pythons.
import tomllib  # type: ignore[import-not-found,no-redef]
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audit_reader import AuditLogReader
from .types import Achievement, AchievementCondition

# --------------------------------------------------------------------------- #
# Per-condition evaluators.
#
# Each signature: (stats: dict, condition: AchievementCondition) -> (unlocked, value)
# `value` is the actual observed value that met (or didn't meet) the condition;
# it gets stored on the Achievement for the UI to display ("You reached $127 P&L").
# --------------------------------------------------------------------------- #


def eval_trade_count(stats: dict[str, Any], cond: AchievementCondition) -> tuple[bool, Any]:
    """Total number of audit entries (executed + skipped).

    We count all entries, not just executed ones, because the "Warming Up /
    10 trades" achievement is about *engagement*, not just execution.
    """
    actual = int(stats.get("total_trades", 0))
    return _compare_with_op(actual, cond.op, cond.value), actual


def eval_win_count(stats: dict[str, Any], cond: AchievementCondition) -> tuple[bool, Any]:
    """Number of settled winning trades (realized_pnl > 0)."""
    actual = int(stats.get("winning_trades", 0))
    return _compare_with_op(actual, cond.op, cond.value), actual


def eval_win_streak(stats: dict[str, Any], cond: AchievementCondition) -> tuple[bool, Any]:
    """Longest run of consecutive winning settled trades.

    `stats` doesn't carry this directly (it's order-sensitive) — we recompute
    it from `stats["_entries"]` if the evaluator injected them, else fall back
    to 0. The evaluator always injects `_entries` when calling these
    functions, so this is safe.
    """
    entries = stats.get("_entries") or []
    longest = 0
    current = 0
    for e in entries:
        # Only settled trades count toward the streak.
        pnl = getattr(e, "realized_pnl_usd", None)
        if pnl is None:
            continue
        if pnl > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return _compare_with_op(longest, cond.op, cond.value), longest


def eval_realized_pnl_usd(
    stats: dict[str, Any], cond: AchievementCondition
) -> tuple[bool, Any]:
    """Total realized P&L in USD (cumulative across all settled trades)."""
    actual = float(stats.get("total_realized_pnl_usd", 0.0))
    return _compare_with_op(actual, cond.op, cond.value), actual


def eval_brier_score(stats: dict[str, Any], cond: AchievementCondition) -> tuple[bool, Any]:
    """Best (lowest) per-analyst Brier score, gated by `min_trades`.

    If `min_trades` is set, the achievement only unlocks once at least that
    many trades have settled — a single-trade Brier of 0.0 isn't meaningful.
    """
    min_trades = cond.min_trades or 0
    settled = int(stats.get("settled_trades", 0))
    if settled < min_trades:
        return False, None

    scores: dict[str, float] = stats.get("per_analyst_brier", {}) or {}
    if not scores:
        return False, None

    # Best = lowest Brier (lower is better).
    best_analyst = min(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = float(scores[best_analyst])
    return _compare_with_op(best_score, cond.op, cond.value), best_score


def eval_drawdown_pct(stats: dict[str, Any], cond: AchievementCondition) -> tuple[bool, Any]:
    """Current drawdown from peak bankroll (percentage).

    The `days` field is currently advisory — we evaluate the *current*
    drawdown, not a rolling-window max. A future enhancement could check that
    the drawdown has stayed under threshold for the last `days` days; that
    requires timestamped bankroll snapshots which the audit log has but the
    stats dict aggregates away. Tracked as a # VERIFY: comment below.
    """
    # VERIFY: extend to time-windowed drawdown (cond.days) once the upstream
    # achievements repo publishes its evaluation reference. For now, current
    # drawdown only.
    actual = float(stats.get("current_drawdown_pct", 0.0))
    return _compare_with_op(actual, cond.op, cond.value), actual


def eval_wins_in_category(
    stats: dict[str, Any], cond: AchievementCondition
) -> tuple[bool, Any]:
    """Number of winning settled trades in a specific market category.

    Requires walking the entries (the stats dict doesn't break down wins by
    category), so like `eval_win_streak` this depends on the evaluator
    injecting `_entries`.
    """
    entries = stats.get("_entries") or []
    target_cat = (cond.category or "").strip().lower()
    if not target_cat:
        return False, 0

    count = 0
    for e in entries:
        pnl = getattr(e, "realized_pnl_usd", None)
        if pnl is None or pnl <= 0:
            continue
        cat = (getattr(e, "category", None) or "").strip().lower()
        if cat == target_cat:
            count += 1
    return _compare_with_op(count, cond.op, cond.value), count


# Dispatch table — string type -> evaluator function.
EVALUATORS: dict[str, Callable[[dict[str, Any], AchievementCondition], tuple[bool, Any]]] = {
    "trade_count": eval_trade_count,
    "win_count": eval_win_count,
    "win_streak": eval_win_streak,
    "realized_pnl_usd": eval_realized_pnl_usd,
    "brier_score": eval_brier_score,
    "drawdown_pct": eval_drawdown_pct,
    "wins_in_category": eval_wins_in_category,
}


def _compare_with_op(actual: float, op: str, threshold: Any) -> bool:
    """Apply comparison operator to two numeric values.

    Coerces `threshold` to float. Raises `ValueError` for unknown ops.
    """
    try:
        thr = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Achievement threshold {threshold!r} is not numeric"
        ) from exc

    if op == ">=":
        return actual >= thr
    if op == "<=":
        return actual <= thr
    if op == ">":
        return actual > thr
    if op == "<":
        return actual < thr
    if op == "==":
        return actual == thr
    raise ValueError(f"Unsupported operator: {op!r}")


class AchievementsEvaluator:
    """Load achievements.toml + evaluate every condition against the audit log.

    Parameters
    ----------
    config_path:
        Path to a TOML file in the upstream `achievements` format
        (top-level `[[achievement]]` table array).
    """

    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Achievements config not found: {self.config_path}"
            )
        self.achievements: list[Achievement] = self._load_config()

    def _load_config(self) -> list[Achievement]:
        """Parse the TOML config into a list of `Achievement` models."""
        with self.config_path.open("rb") as fh:
            data = tomllib.load(fh)

        raw_list = data.get("achievement", [])
        if not isinstance(raw_list, list):
            raise TypeError(
                f"Achievements config {self.config_path}: top-level "
                f"`achievement` must be an array of tables"
            )

        out: list[Achievement] = []
        for i, raw in enumerate(raw_list):
            if not isinstance(raw, dict):
                raise TypeError(
                    f"Achievements config {self.config_path}: "
                    f"achievement[{i}] is not a table"
                )
            # `condition` may be an inline table in the TOML — already a dict.
            cond_raw = raw.get("condition")
            if cond_raw is None:
                raise ValueError(
                    f"Achievements config {self.config_path}: "
                    f"achievement[{i}] missing `condition`"
                )
            cond = AchievementCondition.model_validate(cond_raw)
            out.append(
                Achievement(
                    id=str(raw.get("id", f"achievement_{i}")),
                    name=str(raw.get("name", raw.get("id", f"achievement_{i}"))),
                    description=str(raw.get("description", "")),
                    condition=cond,
                )
            )
        return out

    def evaluate(self, log: AuditLogReader) -> list[Achievement]:
        """Return the full achievement list with `unlocked_at` populated.

        Side-effect-free: returns a fresh list of `Achievement` objects (the
        evaluator's own `self.achievements` is not mutated, so the evaluator
        can be reused across logs / time).

        `unlocked_at` is set to "now" (UTC) when the condition is met. We
        don't try to reconstruct the *original* unlock timestamp from the
        audit log — that's the upstream `achievements` repo's job (it
        persists unlocks as they happen). This evaluator is the *read-side*
        check used by the replay UI.
        """
        stats = log.compute_stats()
        # Inject the raw entries so order-sensitive evaluators (win_streak,
        # wins_in_category) can walk them. Prefixed with `_` so it's clearly
        # not part of the public stats contract.
        stats["_entries"] = log.read_all()

        now = datetime.now(UTC)
        out: list[Achievement] = []
        for ach in self.achievements:
            evaluator = EVALUATORS.get(ach.condition.type)
            if evaluator is None:
                # Unknown condition type — leave locked, log a warning.
                out.append(ach.model_copy(deep=True))
                continue
            try:
                unlocked, value = evaluator(stats, ach.condition)
            except Exception:  # noqa: BLE001 - defensive: a failing evaluator must not crash the whole run
                # Evaluator raised — leave locked, don't crash the whole run.
                out.append(ach.model_copy(deep=True))
                continue

            unlocked_at = now if unlocked else None
            out.append(
                ach.model_copy(
                    update={"unlocked_at": unlocked_at, "unlocked_value": value}
                )
            )
        return out


__all__ = [
    "EVALUATORS",
    "AchievementsEvaluator",
    "eval_brier_score",
    "eval_drawdown_pct",
    "eval_realized_pnl_usd",
    "eval_trade_count",
    "eval_win_count",
    "eval_win_streak",
    "eval_wins_in_category",
]
