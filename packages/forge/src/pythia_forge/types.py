"""Pydantic models for pythia-forge.

These are the data contracts that flow in and out of the backtest harness:

- ``HistoricalMarket``  — one row of the resolved-markets JSON file.
- ``BacktestConfig``    — what to backtest (strategy + markets + capital + filter).
- ``BacktestResult``    — the aggregated output (P&L + calibration metrics).

The strategy TOML is parsed into the sibling packages' own config types
(``ConsensusConfig`` from pythia-consensus, ``RiskConfig`` from pythia-risk,
and the mesh's ``LLMConfig`` + analyst list from pythia-analyst-mesh) 
``BacktestConfig`` only holds the *path* to the strategy TOML plus the
backtest-specific knobs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Historical market — one row of the resolved-markets JSON fixture.
# ---------------------------------------------------------------------------

class HistoricalMarket(BaseModel):
    """A resolved Delphi market, as recorded by the ATT archive.

    This is the input to the backtester: the market as it existed at open
    (question, category, opening YES price, volume) plus the ground-truth
    outcome it eventually settled to.

    Attributes
    ----------
    market_id:
        Delphi market identifier (e.g. ``"dphi_01J"``).
    question:
        The market's resolution question, verbatim.
    category:
        One of ``"politics"``, ``"crypto"``, ``"sports"``, ``"niche"``,
        ``"subjective"``, ``"economics"`` — must match a key in the strategy
        TOML's ``[risk.market_type_rules]`` for the risk engine to size it.
    opened_at:
        ISO-8601 timestamp when the market opened for trading.
    closed_at:
        ISO-8601 timestamp when the market closed for new orders.
    settled_at:
        ISO-8601 timestamp when the AI arbiter resolved the market.
    yes_price_at_open:
        The YES share price at market open, in [0, 1]. The backtester uses
        this as the ``current_yes_price`` fed to the mesh (it is the price the
        bot *would have* seen).
    final_outcome:
        ``"YES"`` or ``"NO"`` — the arbiter's resolution. Ground truth for
        P&L and Brier-score computation.
    volume_usd:
        Total USD volume the market saw over its lifetime. Used by the
        ``min_volume_usd`` markets filter.
    arbiter_model:
        The LLM model the Delphi arbiter used to settle this market
        (e.g. ``"gpt-4o"``). Informational only — not used by the backtester
        logic, but useful for filtering "easy" vs "hard" arbiters.

    # VERIFY: the ATT archive schema (field names, timestamp format, arbiter
    # model field) is inferred from the gensyn-delphi-skills repo and the
    # docs at https://docs.gensyn.ai/tech/agentic-trading. Confirm against
    # a live archive dump once available.
    """

    model_config = ConfigDict(extra="allow")

    market_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    opened_at: datetime
    closed_at: datetime
    settled_at: datetime
    yes_price_at_open: float = Field(..., ge=0.0, le=1.0)
    final_outcome: Literal["YES", "NO"]
    volume_usd: float = Field(..., ge=0.0)
    arbiter_model: str = Field(default="unknown")

    @field_validator("opened_at", "closed_at", "settled_at")
    @classmethod
    def _coerce_utc(cls, v: datetime) -> datetime:
        """Ensure all timestamps are timezone-aware UTC.

        Naive datetimes (no tz info) are treated as UTC — the ATT archive
        is expected to emit ISO-8601 with ``Z`` or ``+00:00``, but we guard
        against naive strings leaking through.
        """
        if v.tzinfo is None:
            from datetime import UTC

            return v.replace(tzinfo=UTC)
        return v

# ---------------------------------------------------------------------------
# Backtest config — what to backtest.
# ---------------------------------------------------------------------------

class BacktestConfig(BaseModel):
    """Configuration for a single backtest run.

    Attributes
    ----------
    strategy_path:
        Path to the strategy TOML (mesh + consensus + risk config).
        See ``configs/strategies/ensemble-v1.toml`` for the format.
    markets_path:
        Path to a JSON file containing a list of ``HistoricalMarket`` objects.
    starting_capital_usd:
        Initial bankroll for the backtest. The risk engine sizes trades as a
        fraction of this.
    markets_filter:
        Optional filter dict applied to the loaded markets. Recognised keys:

        - ``categories``: list[str] — only keep markets whose ``category`` is
          in this list.
        - ``min_volume_usd``: float — drop markets below this volume.
        - ``min_market_lifetime_sec``: float — drop markets whose
          ``settled_at - opened_at`` is shorter than this (trivial markets).
        - ``arbiter_models``: list[str] — only keep markets settled by one of
          these arbiter models (e.g. ``["gpt-4o"]``).
        - ``track_bankroll``: bool — if True, thread the running bankroll
          through markets (exercises drawdown + cool-down gates). Default
          False (each market evaluated against starting capital — additive,
          independent P&L).
        - ``use_real_llm``: bool — if True, call the real LLM provider
          configured in the strategy TOML. Default False (use MockLLM).
    """

    model_config = ConfigDict(extra="forbid")

    strategy_path: Path
    markets_path: Path
    starting_capital_usd: float = Field(..., gt=0.0)
    markets_filter: dict[str, Any] = Field(default_factory=dict)

    @field_validator("strategy_path")
    @classmethod
    def _strategy_must_exist(cls, v: Path) -> Path:
        # We don't hard-fail at construction time (the path may be set before
        # the file is written in test fixtures), but we normalise to absolute.
        return v.resolve() if v.is_absolute() else v

    @field_validator("markets_path")
    @classmethod
    def _markets_must_exist(cls, v: Path) -> Path:
        return v.resolve() if v.is_absolute() else v

# ---------------------------------------------------------------------------
# Backtest result — aggregated output.
# ---------------------------------------------------------------------------

class BacktestResult(BaseModel):
    """Aggregated output of a backtest run.

    All the metrics the report generator needs, frozen into a single pydantic
    model so it serialises cleanly to JSON for archival / comparison runs.

    Attributes
    ----------
    starting_capital_usd:
        Initial bankroll (from config).
    ending_capital_usd:
        Starting + sum of settled trade P&L.
    total_return_pct:
        ``(ending - starting) / starting * 100``.
    sharpe_ratio:
        Annualised Sharpe of daily returns on the equity curve.
        Assumes 1 trade/day → ``sqrt(252)`` annualisation factor.
        Returns 0.0 if fewer than 2 trades.
    max_drawdown_pct:
        Peak-to-trough drawdown on the equity curve, as a percentage.
    total_trades:
        Count of APPROVE'd (and thus settled) trades. Markets where the
        consensus gate returned ``"skip"`` or ``"wait"``, or the risk engine
        returned ``REJECT``, are NOT counted.
    win_rate:
        Fraction of settled trades that were profitable (P&L > 0).
        Returns 0.0 if ``total_trades == 0``.
    brier_scores:
        Per-analyst mean Brier score. ``{analyst_id: mean_brier}``.
        Lower is better; 0.25 = uninformative (always 0.5); 0.0 = perfect.
    per_category_stats:
        Per-category breakdown. ``{category: {count, win_rate, return_pct, brier}}``.
    equity_curve:
        List of ``(datetime, bankroll_usd)`` tuples. First entry is at the
        starting capital (timestamp = first market's opened_at); one entry
        per settled trade thereafter.
    """

    model_config = ConfigDict(extra="forbid")

    starting_capital_usd: float
    ending_capital_usd: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float = Field(..., ge=0.0)
    total_trades: int = Field(..., ge=0)
    win_rate: float = Field(..., ge=0.0, le=1.0)
    brier_scores: dict[str, float]
    per_category_stats: dict[str, dict[str, Any]]
    equity_curve: list[tuple[datetime, float]]

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "HistoricalMarket",
]
