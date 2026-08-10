"""Pydantic v2 data models for the pythia-observability wrapper.

These are the **read-side** types — what the audit log reader parses, what the
achievements evaluator inspects, and what the FastAPI replay server serialises
back to the dashboard UI. The write side (signing each decision into the audit
log) lives in `pythia-consensus` / `pythia-executor`; this wrapper only
**replays** what they already wrote.

The audit log is a single append-only JSONL file. Each line is one
`AuditEntry` covering the full decision chain for one market cycle:

    market → estimates[] → consensus decision → risk plan → receipt (optional)

# VERIFY: the upstream `icohangar-ops/agent-observability` JSONL schema is not
# yet pinned (see VENDOR_COMMIT.txt). The shape below is the *contract* this
# wrapper assumes; if the upstream emits a different field layout, only this
# file needs updating. Two areas in particular need confirming once vendored:
#
#   1. Whether the audit record stores the full decision chain inline (one
#      record per market cycle) or one record per pipeline stage (multiple
#      records per market_id with a `stage` discriminator). We assume the
#      former — easier to replay — and the wrapper reconstructs chains by
#      market_id if the upstream uses the latter.
#
#   2. Whether `signature` is a single Ed25519 hex string or a structured
#      object (algorithm + key_fingerprint + sig). We model it as an opaque
#      string; the replay UI just shows "✓ signed" vs. "stub".
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

class AuditEntry(BaseModel):
    """One record in the audit log — the full decision chain for one market.

    Attributes
    ----------
    timestamp:
        ISO-8601 timestamp of when the record was written (UTC).
    market_id:
        Delphi market identifier.
    estimates:
        List of analyst `Estimate` dicts (probability, confidence, rationale,
        evidence, analyst_id). Stored as `list[dict]` rather than a typed
        Estimate model so the wrapper doesn't hard-depend on
        `pythia_analyst_mesh` being installed — the replay UI just renders the
        raw fields.
    decision:
        `ConsensusDecision` dict (consensus_prob, agreement_score, gate,
        contributor_ids, method).
    plan:
        `TradePlan` dict from pythia-risk (side, size_usd, limit_price,
        rationale, risk_flags, decision="APPROVE"|"REJECT"). Always present,
        even on skipped trades — the plan is "REJECT" with rationale.
    receipt:
        `TradeReceipt` dict from pythia-executor (att_order_id, fill_price,
        signed_by, audit_log_path). `None` for skipped / paper-not-executed
        decisions.
    skipped_reason:
        Human-readable reason this market cycle was skipped. `None` for
        executed trades. Common values: "agreement_below_threshold",
        "drawdown_breaker", "exposure_cap_hit", "min_analysts_not_met",
        "market_type_blocked".
    signature:
        Ed25519 signature hex over the canonical JSON of the record, prefixed
        with the algorithm name. `stub:sha256:<hex>` when the upstream signer
        is not vendored (matches the convention in pythia-consensus/audit.py).
    """

    model_config = ConfigDict(extra="allow")

    timestamp: str = Field(..., description="ISO-8601 UTC timestamp.")
    market_id: str = Field(..., description="Delphi market identifier.")
    estimates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Analyst Estimate dicts (see pythia-analyst-mesh).",
    )
    decision: dict[str, Any] = Field(
        default_factory=dict,
        description="ConsensusDecision dict (see pythia-consensus).",
    )
    plan: dict[str, Any] = Field(
        default_factory=dict,
        description="TradePlan dict (see pythia-risk).",
    )
    receipt: dict[str, Any] | None = Field(
        default=None,
        description="TradeReceipt dict from pythia-executor; None if skipped.",
    )
    skipped_reason: str | None = Field(
        default=None,
        description="Why this market cycle was skipped, if applicable.",
    )
    signature: str | None = Field(
        default=None,
        description="Ed25519 sig hex (or stub:sha256:<hex> fallback).",
    )

    # ----- Convenience accessors used by the reader / evaluator -----

    @property
    def is_executed(self) -> bool:
        """True iff this entry resulted in an actual trade receipt."""
        return self.receipt is not None

    @property
    def is_skipped(self) -> bool:
        """True iff this market cycle was gated out before execution."""
        return self.skipped_reason is not None

    @property
    def is_paper(self) -> bool:
        """True iff this entry was executed in paper mode.

        # VERIFY: how the upstream tags paper vs. live mode in the audit log.
        # Most likely the receipt carries a `mode` field, or the signature is
        # prefixed with `paper:` rather than just `stub:`. We check both.
        """
        if self.receipt is None:
            return False
        mode = str(self.receipt.get("mode", "")).lower()
        if mode == "paper":
            return True
        sig = self.signature or ""
        return sig.startswith("paper:")

    @property
    def realized_pnl_usd(self) -> float | None:
        """Realized P&L for this entry, if the market is settled.

        Lives in `receipt.settlement.realized_pnl_usd` per the
        ARCHITECTURE.md `Settlement` contract. Returns None for unsettled
        or skipped trades.
        """
        if self.receipt is None:
            return None
        settlement = self.receipt.get("settlement")
        if settlement is None:
            return None
        pnl = settlement.get("realized_pnl_usd")
        if pnl is None:
            return None
        try:
            return float(pnl)
        except (TypeError, ValueError):
            return None

    @property
    def won(self) -> bool | None:
        """True iff this entry's market settled in Pythia's favour.

        `realized_pnl_usd > 0`. Returns None for unsettled trades.
        """
        pnl = self.realized_pnl_usd
        if pnl is None:
            return None
        return pnl > 0.0

    @property
    def category(self) -> str | None:
        """Market category (politics / crypto / sports / subjective / ...).

        Lives in `decision.market_category` if the consensus layer annotates
        it (Pythia's pipeline does), else in the first estimate's metadata.
        """
        cat = self.decision.get("market_category")
        if cat:
            return str(cat)
        for est in self.estimates:
            meta = est.get("market_metadata") or {}
            if meta.get("category"):
                return str(meta["category"])
        return None

class PnLMilestone(BaseModel):
    """One point on the cumulative P&L curve — one per executed trade.

    The replay UI charts these in the P&L panel.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(..., description="ISO-8601 UTC timestamp.")
    realized_pnl_usd: float = Field(
        ..., description="Cumulative realized P&L up to and including this trade."
    )
    unrealized_pnl_usd: float = Field(
        default=0.0,
        description="Optional unrealized P&L (open positions) at this timestamp.",
    )
    bankroll_usd: float = Field(
        ..., description="Bankroll after this trade settled."
    )
    drawdown_pct: float = Field(
        ...,
        ge=0.0,
        description="Current drawdown from peak bankroll, in percent (0..100).",
    )

class AchievementCondition(BaseModel):
    """One condition inside an `Achievement`.

    The achievements TOML stores these as inline tables:

        condition = { type = "trade_count", op = ">=", value = 10 }

    Supported `type` values are dispatched by `AchievementsEvaluator`:
    trade_count, win_count, win_streak, realized_pnl_usd, brier_score,
    drawdown_pct, wins_in_category. The optional `min_trades`, `days`, and
    `category` fields are used by specific condition types (e.g. the Brier
    achievement requires `min_trades=20` resolved markets before the score is
    meaningful).
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Condition type (see module docstring).")
    op: str = Field(
        ...,
        description="Comparison operator: >=, <=, ==, >, <.",
    )
    value: Any = Field(..., description="Threshold value (numeric or string).")
    min_trades: int | None = Field(
        default=None,
        ge=0,
        description="Min settled trades required before evaluating (Brier etc.).",
    )
    days: int | None = Field(
        default=None,
        ge=0,
        description="Window in days for time-bounded conditions (drawdown).",
    )
    category: str | None = Field(
        default=None,
        description="Market category filter (wins_in_category).",
    )

class Achievement(BaseModel):
    """One milestone that Pythia can unlock.

    Mirrors the upstream `icohangar-ops/achievements` record shape so the
    wrapper can hand unlocked achievements back to the upstream for badge
    emission if desired. The `unlocked_at` and `unlocked_value` fields are
    populated by `AchievementsEvaluator.evaluate()`.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Stable slug, used as the achievement key.")
    name: str = Field(..., description="Display name (shown in the UI grid).")
    description: str = Field(..., description="One-line description.")
    condition: AchievementCondition = Field(
        ..., description="Evaluable condition for unlock."
    )
    unlocked_at: datetime | None = Field(
        default=None,
        description="When unlocked, else None. Set by the evaluator.",
    )
    unlocked_value: Any = Field(
        default=None,
        description="Actual value that met the condition (for UI display).",
    )

__all__ = [
    "Achievement",
    "AchievementCondition",
    "AuditEntry",
    "PnLMilestone",
]
