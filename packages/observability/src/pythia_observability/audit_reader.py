"""JSONL audit-log reader for the Pythia mesh.

`AuditLogReader` is the single entry point for everything that consumes the
audit log — the achievements evaluator, the replay server, and the CLI. It:

- streams JSONL lazily via `iter_entries()` (for big logs),
- eagerly loads via `read_all()` (for tests + small logs),
- slices by market / time range,
- computes derived analytics: cumulative P&L series, per-analyst Brier scores,
  aggregate stats (win rate, avg stake, drawdown).

The reader is **stateless across calls** — every method re-reads the file
unless `read_all()` has been called (in which case subsequent calls reuse the
cached list). This trades a little I/O for simplicity; the audit log is
append-only and typically small (a few hundred entries for a 2-week Delphi
production run).

# VERIFY: the upstream `icohangar-ops/agent-observability` may emit one JSONL
# record per pipeline *stage* (one line for each of estimates / consensus /
# plan / receipt) keyed by market_id + stage, rather than one line per market
# cycle. The current reader assumes one-line-per-cycle. If the upstream uses
# the stage-per-line shape, a `reconstruct_chains()` adapter would need to be
# added at the top of `iter_entries()` that joins consecutive lines with the
# same market_id. Marked here so the migration is a one-file change.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .types import AuditEntry, PnLMilestone

_log = logging.getLogger(__name__)

def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp into a timezone-aware UTC datetime.

    Handles ``Z`` suffix, naive timestamps (treated as UTC), and offsets.
    """
    # Python 3.11+ `datetime.fromisoformat` accepts `Z` natively.
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)

def _compare(actual: float, op: str, threshold: float) -> bool:
    """Apply a comparison operator: >=, <=, ==, >, <."""
    if op == ">=":
        return actual >= threshold
    if op == "<=":
        return actual <= threshold
    if op == ">":
        return actual > threshold
    if op == "<":
        return actual < threshold
    if op == "==":
        return actual == threshold
    raise ValueError(f"Unsupported operator: {op!r}")

class AuditLogReader:
    """Lazily- or eagerly-read JSONL audit log + derive analytics.

    Parameters
    ----------
    log_path:
        Path to the audit JSONL file. Must exist.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        if not self.log_path.exists():
            raise FileNotFoundError(f"Audit log not found: {self.log_path}")
        self._cache: list[AuditEntry] | None = None

    # ------------------------------------------------------------------ #
    # Core iteration / read.
    # ------------------------------------------------------------------ #

    def iter_entries(self) -> Iterator[AuditEntry]:
        """Stream audit entries lazily — one per non-blank JSONL line.

        Malformed lines are logged at WARNING and skipped (the audit log is
        the last line of defence — we never want one bad line to break the
        replay UI).
        """
        with self.log_path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    _log.warning(
                        "audit log %s:%d: skipping malformed JSON: %s",
                        self.log_path,
                        lineno,
                        exc,
                    )
                    continue
                try:
                    yield AuditEntry.model_validate(payload)
                except Exception as exc:  # noqa: BLE001 - defensive: a bad line must not break the whole stream
                    _log.warning(
                        "audit log %s:%d: skipping entry that failed validation: %s",
                        self.log_path,
                        lineno,
                        exc,
                    )
                    continue

    def read_all(self) -> list[AuditEntry]:
        """Eagerly read + cache all entries. Subsequent calls reuse the cache."""
        if self._cache is None:
            self._cache = list(self.iter_entries())
        return self._cache

    def invalidate_cache(self) -> None:
        """Drop the in-memory cache so the next read re-reads the file."""
        self._cache = None

    # ------------------------------------------------------------------ #
    # Slicing.
    # ------------------------------------------------------------------ #

    def get_by_market(self, market_id: str) -> list[AuditEntry]:
        """Return all entries for one market_id, oldest first."""
        return [e for e in self.read_all() if e.market_id == market_id]

    def filter_by_time(self, start: datetime, end: datetime) -> list[AuditEntry]:
        """Return entries with `start <= timestamp <= end` (inclusive)."""
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        out: list[AuditEntry] = []
        for e in self.read_all():
            try:
                ts = _parse_timestamp(e.timestamp)
            except (ValueError, TypeError):
                continue
            if start <= ts <= end:
                out.append(e)
        return out

    # ------------------------------------------------------------------ #
    # Derived analytics.
    # ------------------------------------------------------------------ #

    def compute_pnl_series(self) -> list[PnLMilestone]:
        """Walk executed + settled trades, emit a cumulative P&L curve.

        Each milestone is one settled trade. `realized_pnl_usd` is cumulative
        across all prior settled trades. `bankroll_usd` starts at the initial
        bankroll (read from the first entry's plan, or defaults to 1000 USD)
        and tracks the running total. `drawdown_pct` is the percentage drop
        from the peak bankroll so far.

        Entries with no realized P&L (skipped / unsettled) are skipped here
        — they don't move the bankroll.
        """
        entries = self.read_all()

        # Initial bankroll: read from the first entry's `plan.bankroll_before`
        # if present (pythia-risk annotates each TradePlan with current
        # bankroll); else fall back to 1000 USD (Pythia's MVP default).
        # VERIFY: pythia-risk doesn't currently stamp `bankroll_before` on
        # the plan — we may want to add that, or have the executor stamp it
        # on the receipt. Until then, default is 1000.
        initial_bankroll = 1000.0
        for e in entries:
            bb = e.plan.get("bankroll_before") if e.plan else None
            if bb is not None:
                try:
                    initial_bankroll = float(bb)
                    break
                except (TypeError, ValueError):
                    pass

        milestones: list[PnLMilestone] = []
        cumulative_pnl = 0.0
        bankroll = initial_bankroll
        peak = initial_bankroll

        for e in entries:
            pnl = e.realized_pnl_usd
            if pnl is None:
                continue
            cumulative_pnl += pnl
            bankroll += pnl
            peak = max(peak, bankroll)
            if peak > 0:
                drawdown = max(0.0, (peak - bankroll) / peak) * 100.0
            else:
                drawdown = 0.0
            milestones.append(
                PnLMilestone(
                    timestamp=e.timestamp,
                    realized_pnl_usd=round(cumulative_pnl, 4),
                    unrealized_pnl_usd=0.0,
                    bankroll_usd=round(bankroll, 4),
                    drawdown_pct=round(drawdown, 4),
                )
            )
        return milestones

    def compute_brier_scores(self) -> dict[str, float]:
        """Per-analyst Brier score from settled markets.

        Brier = mean((p_i - o_i)^2) where p_i is the analyst's probability
        estimate and o_i is the outcome (1 if YES settled, 0 if NO settled).
        Lower = better (0 = perfect, 0.25 = uninformative, 0.33 = always-0.5).

        Outcome is read from `receipt.settlement.outcome` ("YES" | "NO") on
        executed + settled trades. Analysts only get scored on the markets
        they actually produced an estimate for.
        """
        # analyst_id -> list of (prob, outcome_int)
        contributions: dict[str, list[tuple[float, int]]] = {}

        for e in self.read_all():
            if e.receipt is None:
                continue
            settlement = e.receipt.get("settlement")
            if not settlement:
                continue
            outcome_raw = settlement.get("outcome")
            if outcome_raw is None:
                continue
            outcome_str = str(outcome_raw).strip().upper()
            if outcome_str == "YES":
                outcome = 1
            elif outcome_str == "NO":
                outcome = 0
            else:
                # Some markets settle on a numeric value (e.g. 0.42 for a
                # "what % will X reach" market). If we get a number, treat
                # it directly as the outcome.
                try:
                    outcome = float(outcome_raw)  # type: ignore[arg-type]
                    if not (0.0 <= outcome <= 1.0):
                        continue
                except (TypeError, ValueError):
                    continue

            for est in e.estimates:
                analyst_id = est.get("analyst_id")
                prob = est.get("probability")
                if analyst_id is None or prob is None:
                    continue
                try:
                    prob_f = float(prob)
                except (TypeError, ValueError):
                    continue
                if not (0.0 <= prob_f <= 1.0):
                    continue
                contributions.setdefault(str(analyst_id), []).append(
                    (prob_f, outcome)
                )

        scores: dict[str, float] = {}
        for analyst_id, pairs in contributions.items():
            if not pairs:
                continue
            mse = sum((p - o) ** 2 for p, o in pairs) / len(pairs)
            scores[analyst_id] = round(mse, 4)
        return scores

    def compute_stats(self) -> dict[str, Any]:
        """Aggregate stats for the dashboard top-of-page panel.

        Returns
        -------
        dict with keys:
            total_trades: int              — all audit entries (executed + skipped)
            executed_trades: int           — entries that produced a receipt
            skipped_trades: int            — entries that were gated out
            paper_trades: int              — executed in paper mode
            settled_trades: int            — executed + settlement known
            winning_trades: int            — settled with realized_pnl > 0
            losing_trades: int             — settled with realized_pnl < 0
            win_rate: float                — winning_trades / settled_trades (0..1)
            avg_stake_usd: float           — mean plan.size_usd over executed
            total_realized_pnl_usd: float  — sum of realized P&L on settled
            current_bankroll_usd: float    — last known bankroll
            peak_bankroll_usd: float       — max bankroll across the P&L series
            current_drawdown_pct: float    — drop from peak (0..100)
            per_analyst_brier: dict        — output of compute_brier_scores()
            skipped_reasons: dict          — {reason: count}
            signature_stub_count: int      — entries whose sig starts with `stub:`
        """
        entries = self.read_all()
        milestones = self.compute_pnl_series()

        executed = [e for e in entries if e.is_executed]
        skipped = [e for e in entries if e.is_skipped]
        settled = [e for e in entries if e.realized_pnl_usd is not None]
        winning = [e for e in settled if (e.realized_pnl_usd or 0.0) > 0.0]
        losing = [e for e in settled if (e.realized_pnl_usd or 0.0) < 0.0]
        paper = [e for e in entries if e.is_paper]

        stakes = [
            float(e.plan.get("size_usd", 0.0))
            for e in executed
            if e.plan.get("size_usd") is not None
        ]
        avg_stake = sum(stakes) / len(stakes) if stakes else 0.0

        total_pnl = sum(e.realized_pnl_usd or 0.0 for e in settled)

        # Bankroll / drawdown from the P&L series (more accurate than recomputing).
        if milestones:
            current_bankroll = milestones[-1].bankroll_usd
            peak_bankroll = max(m.bankroll_usd for m in milestones)
            current_drawdown = milestones[-1].drawdown_pct
        else:
            # No settled trades yet — fall back to initial bankroll if present.
            current_bankroll = 1000.0
            for e in entries:
                bb = e.plan.get("bankroll_before") if e.plan else None
                if bb is not None:
                    try:
                        current_bankroll = float(bb)
                        break
                    except (TypeError, ValueError):
                        pass
            peak_bankroll = current_bankroll
            current_drawdown = 0.0

        # Skipped-reason breakdown.
        reasons: dict[str, int] = {}
        for e in skipped:
            r = e.skipped_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1

        stub_count = sum(1 for e in entries if (e.signature or "").startswith("stub:"))

        return {
            "total_trades": len(entries),
            "executed_trades": len(executed),
            "skipped_trades": len(skipped),
            "paper_trades": len(paper),
            "settled_trades": len(settled),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": (len(winning) / len(settled)) if settled else 0.0,
            "avg_stake_usd": round(avg_stake, 4),
            "total_realized_pnl_usd": round(total_pnl, 4),
            "current_bankroll_usd": round(current_bankroll, 4),
            "peak_bankroll_usd": round(peak_bankroll, 4),
            "current_drawdown_pct": round(current_drawdown, 4),
            "per_analyst_brier": self.compute_brier_scores(),
            "skipped_reasons": reasons,
            "signature_stub_count": stub_count,
        }

__all__ = ["AuditLogReader"]
