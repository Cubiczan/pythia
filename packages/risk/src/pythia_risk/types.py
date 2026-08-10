"""Type definitions for pythia-risk.

This module owns:
    - The pydantic models for RiskConfig, TradePlan, BankrollState, MarketTypeRules,
      Market, TradeReceipt.
    - A re-export of ConsensusDecision from pythia_consensus (with a local
      fallback if the consensus package is not installed yet, so the risk engine
      can be developed / tested in isolation).

The fallback ConsensusDecision is intentionally minimal — it mirrors the
contract documented in the top-level ARCHITECTURE.md but is *not* the source of
truth. Once pythia_consensus is vendored, the try/except import below resolves
to the real type.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# --- ConsensusDecision re-export ----------------------------------------------

# We try the real pythia_consensus package first. If it isn't installed (e.g.
# during isolated risk-engine development), fall back to a local dataclass-
# equivalent pydantic model that matches the documented contract.
try:  # pragma: no cover - exercised only when the consensus wrapper is present
    from pythia_consensus.types import ConsensusDecision  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 - we want to swallow any import failure

    class ConsensusDecision(BaseModel):
        """Fallback ConsensusDecision.

        Fused analyst output. Source of truth lives in pythia_consensus.types;
        this is a faithful but minimal mirror so pythia-risk can be developed
        in isolation before the consensus wrapper is vendored.
        """

        model_config = ConfigDict(extra="allow")

        market_id: str
        consensus_prob: float = Field(..., ge=0.0, le=1.0)
        agreement_score: float = Field(..., ge=0.0, le=1.0)
        gate: Literal["trade", "skip", "wait"]
        contributor_ids: list[str] = Field(default_factory=list)
        method: str = "logit-mean"
        timestamp: str = ""


# --- Sizing & rule types -------------------------------------------------------

SizingMethod = Literal["kelly-fractional", "fixed"]


class MarketTypeRules(BaseModel):
    """Per-category risk rules (politics / crypto / sports / niche / subjective)."""

    model_config = ConfigDict(extra="forbid")

    max_stake_usd: float = Field(..., gt=0.0, description="Hard cap on stake for this category.")
    allowed: bool = Field(True, description="Whether the bot may trade this category at all.")


# --- Core config & state -------------------------------------------------------

class RiskConfig(BaseModel):
    """Risk configuration.

    Mirrors the ``[risk]`` section of ``configs/live-mvp.toml``. Field names
    match the TOML keys exactly so the config can be loaded with ``tomllib`` +
    ``RiskConfig.model_validate(data["risk"])``.
    """

    model_config = ConfigDict(extra="forbid")

    sizing: SizingMethod = "kelly-fractional"
    kelly_fraction: float = Field(0.25, gt=0.0, le=1.0, description="Fractional Kelly multiplier (0.25 = quarter-Kelly).")
    max_stake_per_market_usd: float = Field(..., gt=0.0, description="Global per-market stake cap.")
    max_total_exposure_usd: float = Field(..., gt=0.0, description="Cap on sum of open positions + proposed stake.")
    max_drawdown_pct: float = Field(..., ge=0.0, le=100.0, description="Drawdown circuit breaker threshold.")
    cool_down_min_after_loss: float = Field(..., ge=0.0, description="Wall-clock minutes to pause trading after a loss.")
    market_type_rules: dict[str, MarketTypeRules] = Field(default_factory=dict)


class BankrollState(BaseModel):
    """Snapshot of the bot's capital position.

    Passed into ``RiskEngine.evaluate``. The engine treats it as read-only;
    only ``update_state`` / ``record_loss`` / ``record_win`` mutate the copy
    held on the engine instance.

    ``drawdown_pct`` is computed off ``peak_bankroll_usd``, not starting
    capital: a new high-water mark resets the breaker's denominator.
    """

    model_config = ConfigDict(extra="forbid")

    cash_usd: float = Field(..., ge=0.0)
    open_positions_usd: float = Field(0.0, ge=0.0)
    peak_bankroll_usd: float = Field(..., ge=0.0)
    current_bankroll_usd: float = Field(..., ge=0.0)
    drawdown_pct: float = Field(0.0, ge=0.0, le=100.0)
    last_loss_at: Optional[datetime] = None


# --- Market & trade outputs ----------------------------------------------------

class Market(BaseModel):
    """Market metadata needed by the risk engine.

    NOTE: the canonical ``Market`` type lives in ``pythia-delphi-adapter``.
    This is a local mirror for type-checking and isolated testing. When the
    adapter is vendored, replace this with a re-export (same try/except
    pattern as ConsensusDecision above).
    """

    model_config = ConfigDict(extra="allow")

    market_id: str
    yes_price: float = Field(..., ge=0.0, le=1.0, description="Current YES share price (probability quote).")
    category: str = Field("niche", description="Market category key — must match a key in market_type_rules.")
    question: str = ""
    close_date: Optional[datetime] = None


class TradePlan(BaseModel):
    """Output of ``RiskEngine.evaluate``.

    Consumed by ``pythia-executor``. If ``decision == "REJECT"`` the executor
    must refuse to submit, regardless of ``size_usd``.
    """

    model_config = ConfigDict(extra="forbid")

    market_id: str
    side: Literal["YES", "NO"]
    size_usd: float = Field(..., ge=0.0)
    limit_price: Optional[float] = Field(None, ge=0.0, le=1.0, description="None = market order.")
    rationale: str
    risk_flags: list[str] = Field(default_factory=list)
    decision: Literal["APPROVE", "REJECT"]
    timestamp: str = ""


class TradeReceipt(BaseModel):
    """Confirmation of a filled trade (produced by pythia-executor).

    The risk engine consumes this via ``update_state`` to keep its internal
    BankrollState in sync with the live ledger. The shape mirrors the
    ``TradeReceipt`` contract in the top-level ARCHITECTURE.md.
    """

    model_config = ConfigDict(extra="allow")

    market_id: str
    side: str
    size_usd: float
    fill_price: float
    att_order_id: str = ""
    signed_by: str = ""
    timestamp: str = ""
    audit_log_path: str = ""
