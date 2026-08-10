"""pythia-risk: Delphi-specific risk gating (wrapper around icohangar-ops/meshcfo).

Public API:
    RiskEngine      — runs the 6-gate evaluation pipeline (market-type, drawdown,
                      cool-down, exposure, no-edge, per-market cap) and returns a
                      TradePlan with APPROVE / REJECT + risk_flags.
    TradePlan       — output of RiskEngine.evaluate; consumed by pythia-executor.
    RiskConfig      — config schema (sizing method, Kelly fraction, caps, rules).
    size_trade      — convenience: size a trade from a ConsensusDecision + Market
                      without standing up a full RiskEngine.

Submodules:
    types           — pydantic models + ConsensusDecision re-export.
    sizing          — Kelly criterion for multi-outcome LMSR markets.
    engine          — RiskEngine implementation.
    cli             — `pythia-risk` command-line entry point.
"""

from pythia_risk.engine import RiskEngine
from pythia_risk.sizing import (
    kelly_fraction,
    size_trade_fixed,
    size_trade_kelly,
    size_trade_kelly_multi,
)
from pythia_risk.types import (
    BankrollState,
    ConsensusDecision,
    Market,
    MarketTypeRules,
    RiskConfig,
    SizingMethod,
    TradePlan,
    TradeReceipt,
)

__version__ = "0.1.0"

__all__ = [
    "RiskEngine",
    "TradePlan",
    "RiskConfig",
    "BankrollState",
    "MarketTypeRules",
    "Market",
    "TradeReceipt",
    "ConsensusDecision",
    "SizingMethod",
    "kelly_fraction",
    "size_trade_kelly",
    "size_trade_kelly_multi",
    "size_trade_fixed",
    "size_trade",
    "__version__",
]

def size_trade(
    consensus_prob: float,
    market_price: float,
    bankroll_usd: float,
    *,
    kelly_fraction_multiplier: float = 0.25,
    max_stake_usd: float = 50.0,
) -> float:
    """Convenience wrapper around :func:`sizing.size_trade_kelly`.

    Returns the dollar stake for a binary-outcome market given the consensus
    probability, the current YES price, the current bankroll, and a Kelly
    fraction multiplier (default quarter-Kelly = 0.25). Capped by
    ``max_stake_usd``.
    """
    return size_trade_kelly(
        p_consensus=consensus_prob,
        market_price=market_price,
        bankroll_usd=bankroll_usd,
        kelly_fraction=kelly_fraction_multiplier,
        max_stake_usd=max_stake_usd,
    )
