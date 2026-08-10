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

try:  # pragma: no cover - exercised only when the consensus wrapper is present
    from pythia_consensus.types import ConsensusDecision  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001

    class ConsensusDecision(BaseModel):
        """Fallback ConsensusDecision."""

        model_config = ConfigDict(extra="allow")

        market_id: str
        consensus_prob: float = Field(..., ge=0.0, le=1.0)
        agreement_score: float = Field(..., ge=0.0, le=1.0)
        gate: Literal["trade", "skip", "wait"]
        contributor_ids: list[str] = Field(default_factory=list)
        method: str = "logit-mean"
        weights_used: dict[str, float] = Field(default_factory=dict)
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
    no_edge_tolerance: float = Field(0.02, ge=0.0, le=1.0, description="Minimum |edge| to consider a trade.")
    market_type_rules: dict[str, MarketTypeRules] = Field(default_factory=dict)

class BankrollState(BaseModel):
    """Snapshot of the bot's capital position."""

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

    Supports multi-outcome LMSR markets. A market has:
      - ``outcomes``: list of outcome labels (e.g. ["YES", "NO"] or
        ["Bitcoin", "Ethereum", "Solana"]).
      - ``spot_prices``: per-outcome spot prices (implied probabilities),
        same length as ``outcomes``.

    For backward compatibility with binary-market test code, a ``yes_price``
    field is accepted as a convenience: if ``spot_prices`` is not provided
    but ``yes_price`` is, the market is treated as binary with
    ``outcomes = ["YES", "NO"]`` and ``spot_prices = [yes_price, 1 - yes_price]``.
    """

    model_config = ConfigDict(extra="allow")

    market_id: str
    category: str = Field("niche", description="Market category key — must match a key in market_type_rules.")
    question: str = ""
    outcomes: list[str] = Field(
        default_factory=lambda: ["YES", "NO"],
        description="Outcome labels. Defaults to binary YES/NO for backward compat.",
    )
    spot_prices: list[float] = Field(
        default_factory=list,
        description="Per-outcome spot prices (implied probabilities). If empty, derived from yes_price.",
    )
    # Legacy field — accepted for backward compat, used to derive spot_prices
    # when spot_prices is not provided.
    yes_price: float | None = Field(
        None, ge=0.0, le=1.0, description="Legacy: current YES price. Used to derive spot_prices if not given."
    )
    close_date: Optional[datetime] = None

    def model_post_init(self, __context: object) -> None:
        """After validation, derive spot_prices from yes_price if not provided."""
        if not self.spot_prices and self.yes_price is not None:
            self.spot_prices = [self.yes_price, 1.0 - self.yes_price]
            if len(self.outcomes) == 0:
                self.outcomes = ["YES", "NO"]
        elif not self.spot_prices and self.outcomes:
            # No prices at all — default to uniform distribution.
            n = len(self.outcomes)
            self.spot_prices = [1.0 / n] * n

class TradePlan(BaseModel):
    """Output of ``RiskEngine.evaluate``.

    For multi-outcome markets, ``outcome_idx`` (0-based) identifies which
    outcome the plan bets on, and ``side`` is the outcome label.

    For backward compatibility with binary markets, ``side`` is kept as a
    string ("YES" or "NO" for binary, or the outcome label for multi-outcome).
    """

    model_config = ConfigDict(extra="forbid")

    market_id: str
    side: str = Field(..., description="Outcome label the plan bets on (e.g. 'YES', 'Bitcoin').")
    outcome_idx: int = Field(0, ge=0, description="0-based index of the chosen outcome.")
    size_usd: float = Field(..., ge=0.0)
    limit_price: Optional[float] = Field(None, ge=0.0, le=1.0, description="None = market order.")
    rationale: str
    risk_flags: list[str] = Field(default_factory=list)
    decision: Literal["APPROVE", "REJECT"]
    timestamp: str = ""

class TradeReceipt(BaseModel):
    """Confirmation of a filled trade (produced by pythia-executor)."""

    model_config = ConfigDict(extra="allow")

    market_id: str
    side: str
    outcome_idx: int = 0
    size_usd: float
    fill_price: float
    att_order_id: str = ""
    signed_by: str = ""
    timestamp: str = ""
    audit_log_path: str = ""
