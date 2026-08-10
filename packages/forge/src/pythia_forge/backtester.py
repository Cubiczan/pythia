"""The core backtest harness.

``Backtester`` replays resolved historical Delphi markets through the full
Pythia mesh → consensus → risk pipeline, "places" every APPROVE'd trade, and
"settles" it against the known final outcome to compute realised P&L.

The harness is **deterministic by default** — it uses ``MockLLM`` to produce
analyst estimates without burning real LLM credits, so backtest results are
reproducible and a tune sweep over thousands of parameter combinations
finishes in seconds. Pass ``use_real_llm=True`` in the markets filter (or
``--use-real-llm`` on the CLI) to call the actual LLM provider configured in
the strategy TOML — slow and costly, but the only way to validate a final
config against real analyst behaviour.

Pipeline per market::

    HistoricalMarket
        │
        ▼
    MarketContext (yes_price_at_open → current_yes_price)
        │
        ▼
    run_mesh  ──▶  list[Estimate]   (4 analysts, concurrent, with MockLLM by default)
        │
        ▼
    fuse  ──▶  ConsensusDecision   (consensus_prob, agreement_score, gate)
        │
        ▼  (skip if gate != "trade")
    RiskEngine.evaluate  ──▶  TradePlan  (side, size_usd, APPROVE/REJECT)
        │
        ▼  (skip if REJECT)
    _settle  ──▶  P&L  (compare plan.side to market.final_outcome)
        │
        ▼
    equity_curve.append((settled_at, running_capital + pnl))

Aggregation at the end produces the ``BacktestResult``: total return, Sharpe,
max drawdown, win rate, per-analyst Brier scores, per-category breakdown.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# tomllib is stdlib in 3.11+; fall back to tomli for older Pythons.
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - py < 3.11 only
    import tomli as tomllib  # type: ignore[no-redef]

from pythia_analyst_mesh import (
    AnalystRegistry,
    BaseAnalyst,
    LLMConfig,
    MarketContext,
    run_mesh,
)
from pythia_consensus import ConsensusConfig, fuse
from pythia_risk import BankrollState, MarketTypeRules, RiskConfig, TradeReceipt
from pythia_risk.engine import RiskEngine
from pythia_risk.types import Market as RiskMarket

from .mock_llm import MockLLM
from .types import BacktestConfig, BacktestResult, HistoricalMarket

logger = logging.getLogger(__name__)

# Annualisation factor for the Sharpe ratio. The spec says "assume 1 trade/day
# for simplicity" → 252 trading days/year is the standard equity-market convention.
_TRADING_DAYS_PER_YEAR = 252

# Guard against division-by-zero in the settle math (price clamped to [eps, 1-eps]).
_PRICE_EPS = 0.01


# ---------------------------------------------------------------------------
# Internal trade record — accumulates everything needed for aggregation.
# ---------------------------------------------------------------------------


@dataclass
class _TradeRecord:
    """One settled trade, with the estimates that produced it.

    Stored per trade so we can compute per-analyst Brier scores and
    per-category breakdowns after the full run completes.
    """

    market_id: str
    category: str
    question: str
    side: str  # "YES" | "NO"
    stake_usd: float
    pnl_usd: float
    final_outcome: str  # "YES" | "NO"
    # analyst_id → probability that analyst gave for this market.
    # Only includes analysts that actually returned an estimate.
    estimates: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """Replay resolved Delphi markets through the full Pythia pipeline.

    Parameters
    ----------
    config : BacktestConfig
        Strategy TOML path, markets JSON path, starting capital, filter.

    Example
    -------
    ::

        import asyncio
        from pathlib import Path
        from pythia_forge import Backtester, BacktestConfig

        cfg = BacktestConfig(
            strategy_path=Path("configs/strategies/ensemble-v1.toml"),
            markets_path=Path("resolved-2025-Q4.json"),
            starting_capital_usd=1000.0,
        )
        result = asyncio.run(Backtester(cfg).run())
        print(f"Return: {result.total_return_pct:.2f}%")
    """

    def __init__(
        self,
        config: BacktestConfig,
        *,
        strategy_override: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the backtester.

        Parameters
        ----------
        config : BacktestConfig
            Strategy TOML path, markets JSON path, starting capital, filter.
        strategy_override : dict, optional
            If provided, used in place of loading ``config.strategy_path``.
            This is the injection point used by the ``tune`` CLI subcommand
            to grid-search parameter combinations without writing temp files.
        """
        self.config: BacktestConfig = config
        self._mock_llm: MockLLM = MockLLM()
        self._strategy_override: dict[str, Any] | None = strategy_override

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self) -> BacktestResult:
        """Run the full backtest and return aggregated results.

        Loads the strategy + markets, applies the filter, builds the mesh +
        consensus + risk engines, iterates over every market, and aggregates
        the settled trades into a ``BacktestResult``.
        """
        # ---- Load & parse inputs ----------------------------------------
        strategy = self._load_strategy(self.config.strategy_path)
        markets = self._load_markets(self.config.markets_path)
        markets = self._apply_filter(markets)

        use_real_llm = bool(self.config.markets_filter.get("use_real_llm", False))
        track_bankroll = bool(self.config.markets_filter.get("track_bankroll", False))
        analyst_timeout = float(
            self.config.markets_filter.get("analyst_timeout_sec", 30.0)
        )

        if not markets:
            logger.warning("backtest: no markets survived the filter; returning empty result")
            return self._empty_result()

        # ---- Build the pipeline components ------------------------------
        mesh = self._build_mesh(strategy, use_real_llm)
        consensus_config = self._build_consensus_config(strategy)
        risk_engine = self._build_risk_engine(strategy)

        # ---- Iterate over markets ---------------------------------------
        trades: list[_TradeRecord] = []
        equity_curve: list[tuple[datetime, float]] = []

        starting = self.config.starting_capital_usd
        running_capital = starting

        # Seed the equity curve at the first market's open time.
        first_ts = markets[0].opened_at
        equity_curve.append((first_ts, running_capital))

        # Bankroll state for the risk engine.
        # In track_bankroll mode, this is mutated as trades fill + settle.
        # In default mode, we pass a fresh-ish snapshot each market so the
        # drawdown / cool-down gates never trip (each market is independent).
        bankroll = BankrollState(
            cash_usd=starting,
            open_positions_usd=0.0,
            peak_bankroll_usd=starting,
            current_bankroll_usd=starting,
            drawdown_pct=0.0,
            last_loss_at=None,
        )

        for idx, market in enumerate(markets):
            ctx = self._to_market_context(market)

            # ---- Mesh: produce estimates --------------------------------
            try:
                estimates = await run_mesh(ctx, mesh, timeout_sec=analyst_timeout)
            except Exception as exc:  # noqa: BLE001 — mesh must not crash backtest
                logger.warning(
                    "market=%s (%d/%d): mesh raised %s: %s; skipping",
                    market.market_id, idx + 1, len(markets), type(exc).__name__, exc,
                )
                continue

            if not estimates:
                logger.info(
                    "market=%s (%d/%d): no estimates returned; skipping",
                    market.market_id, idx + 1, len(markets),
                )
                continue

            # ---- Consensus: fuse estimates → decision -------------------
            try:
                decision = fuse(estimates, consensus_config)
            except ValueError as exc:
                logger.warning(
                    "market=%s: fuse() failed: %s; skipping", market.market_id, exc
                )
                continue

            if decision.gate != "trade":
                logger.info(
                    "market=%s: consensus gate=%s (agreement=%.3f, n=%d); no trade",
                    market.market_id, decision.gate, decision.agreement_score,
                    len(estimates),
                )
                continue

            # ---- Risk: evaluate → trade plan ----------------------------
            risk_market = RiskMarket(
                market_id=market.market_id,
                yes_price=market.yes_price_at_open,
                category=market.category,
                question=market.question,
            )

            # In track_bankroll mode, thread the live bankroll through.
            # In default mode, always pass the starting-capital snapshot so
            # per-market P&L is independent and additive.
            current_bankroll = bankroll if track_bankroll else bankroll.model_copy(
                update={
                    "current_bankroll_usd": starting,
                    "cash_usd": starting,
                    "open_positions_usd": 0.0,
                    "drawdown_pct": 0.0,
                    "last_loss_at": None,
                }
            )

            try:
                plan = risk_engine.evaluate(
                    decision=decision,
                    market=risk_market,
                    current_bankroll=current_bankroll,
                )
            except Exception as exc:  # noqa: BLE001 — risk engine must not crash backtest
                logger.warning(
                    "market=%s: risk.evaluate raised %s: %s; skipping",
                    market.market_id, type(exc).__name__, exc,
                )
                continue

            # ---- Record Brier contributions for every analyst -----------
            # (even if the trade was REJECTed — the analyst still gave an
            # estimate, which is what Brier measures. The outcome is
            # resolved inside _compute_brier from final_outcome.)
            est_map: dict[str, float] = {
                est.analyst_id: est.probability for est in estimates
            }

            if plan.decision != "APPROVE":
                logger.info(
                    "market=%s: risk REJECT (%s); no trade placed",
                    market.market_id, ", ".join(plan.risk_flags) or "no flags",
                )
                continue

            # ---- Settle the trade ---------------------------------------
            pnl = self._settle(plan.side, plan.size_usd, market.yes_price_at_open,
                               market.final_outcome)
            running_capital += pnl

            trades.append(
                _TradeRecord(
                    market_id=market.market_id,
                    category=market.category,
                    question=market.question,
                    side=plan.side,
                    stake_usd=plan.size_usd,
                    pnl_usd=pnl,
                    final_outcome=market.final_outcome,
                    estimates=est_map,
                )
            )

            # ---- Update bankroll state (track_bankroll mode only) -------
            if track_bankroll:
                receipt = TradeReceipt(
                    market_id=market.market_id,
                    side=plan.side,
                    size_usd=plan.size_usd,
                    fill_price=market.yes_price_at_open,
                    timestamp=datetime.now(UTC).isoformat(),
                )
                risk_engine.update_state(receipt)
                if pnl >= 0:
                    risk_engine.record_win(market.market_id, pnl)
                else:
                    risk_engine.record_loss(market.market_id, abs(pnl))
                bankroll = risk_engine.state
                running_capital = bankroll.current_bankroll_usd

            equity_curve.append((market.settled_at, running_capital))

            logger.info(
                "market=%s: %s $%.2f @ %.3f → %s | P&L=%+.2f | bankroll=$%.2f",
                market.market_id, plan.side, plan.size_usd,
                market.yes_price_at_open, market.final_outcome, pnl, running_capital,
            )

        # ---- Aggregate --------------------------------------------------
        return self._aggregate(
            trades=trades,
            equity_curve=equity_curve,
            starting_capital=starting,
            ending_capital=running_capital,
        )

    # ------------------------------------------------------------------ #
    # Strategy + markets loading
    # ------------------------------------------------------------------ #

    def _load_strategy(self, path: Path) -> dict[str, Any]:
        """Parse the strategy TOML into a raw dict.

        The dict is then consumed by ``_build_mesh`` / ``_build_consensus_config``
        / ``_build_risk_engine`` to construct the sibling-package config objects.

        If ``strategy_override`` was passed to ``__init__``, it is returned
        directly and ``path`` is ignored (used by the ``tune`` CLI).
        """
        if self._strategy_override is not None:
            return self._strategy_override
        if not path.exists():
            raise FileNotFoundError(f"strategy TOML not found: {path}")
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        # The strategy TOML wraps everything under [strategy].
        # Allow either top-level [strategy] or flat (mesh/consensus/risk at root).
        if "strategy" in data:
            return data["strategy"]
        return data

    def _load_markets(self, path: Path) -> list[HistoricalMarket]:
        """Load the resolved-markets JSON file into a list of HistoricalMarket."""
        if not path.exists():
            raise FileNotFoundError(f"markets JSON not found: {path}")
        import json

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise ValueError(
                f"markets JSON must be a list of objects; got {type(raw).__name__}"
            )
        return [HistoricalMarket.model_validate(m) for m in raw]

    def _apply_filter(self, markets: list[HistoricalMarket]) -> list[HistoricalMarket]:
        """Apply the markets_filter from BacktestConfig.

        Recognised filter keys:
        - categories: list[str]
        - min_volume_usd: float
        - min_market_lifetime_sec: float
        - arbiter_models: list[str]
        """
        flt = self.config.markets_filter
        if not flt:
            return markets

        result = markets

        cats = flt.get("categories")
        if cats:
            cats_set = {str(c).lower() for c in cats}
            result = [m for m in result if m.category.lower() in cats_set]

        min_vol = flt.get("min_volume_usd")
        if min_vol is not None:
            min_vol_f = float(min_vol)
            result = [m for m in result if m.volume_usd >= min_vol_f]

        min_life = flt.get("min_market_lifetime_sec")
        if min_life is not None:
            min_life_f = float(min_life)
            result = [
                m for m in result
                if (m.settled_at - m.opened_at).total_seconds() >= min_life_f
            ]

        arb = flt.get("arbiter_models")
        if arb:
            arb_set = {str(a) for a in arb}
            result = [m for m in result if m.arbiter_model in arb_set]

        logger.info(
            "markets filter: %d → %d markets (filter=%s)",
            len(markets), len(result), flt,
        )
        return result

    # ------------------------------------------------------------------ #
    # Pipeline construction
    # ------------------------------------------------------------------ #

    def _build_mesh(self, strategy: dict[str, Any], use_real_llm: bool) -> list[BaseAnalyst]:
        """Build the analyst mesh from the strategy TOML's [mesh] section.

        If ``use_real_llm`` is False (default), the MockLLM is installed on
        each analyst by monkey-patching ``_call_llm``.
        """
        mesh_cfg = strategy.get("mesh", {})
        analyst_names: list[str] = list(mesh_cfg.get("analysts", []))
        if not analyst_names:
            # Fall back to the 4 built-ins if the TOML doesn't specify.
            analyst_names = ["politics", "crypto", "sports", "niche"]

        # Build the LLMConfig. When use_real_llm is False, the config is
        # irrelevant (MockLLM replaces _call_llm), but we still need a valid
        # LLMConfig to instantiate the analysts.
        import os

        llm_config = LLMConfig(
            provider=str(mesh_cfg.get("llm_provider", "openai")),
            model=str(mesh_cfg.get("llm_model", "gpt-4o-mini")),
            api_key=os.environ.get("LLM_API_KEY") if use_real_llm else "mock-not-used",
            temperature=float(mesh_cfg.get("llm_temperature", 0.2)),
            max_tokens=int(mesh_cfg.get("llm_max_tokens", 800)),
        )

        registry = AnalystRegistry()
        mesh = registry.build_mesh(analyst_names, llm_config)

        if not use_real_llm:
            self._install_mock_llm(mesh)
            logger.info(
                "mesh: %d analysts (%s) with MockLLM (deterministic, no LLM calls)",
                len(mesh), ", ".join(a.analyst_id for a in mesh),
            )
        else:
            logger.warning(
                "mesh: %d analysts with REAL LLM (%s/%s) — this will be slow and cost tokens",
                len(mesh), llm_config.provider, llm_config.model,
            )

        return mesh

    def _install_mock_llm(self, mesh: list[BaseAnalyst]) -> None:
        """Replace each analyst's ``_call_llm`` with a MockLLM-backed closure.

        The mesh's ``BaseAnalyst.estimate`` calls ``self._call_llm(messages, config)``
        and expects a string back (the assistant's text content). We produce a
        JSON string that ``_parse_llm_response`` will turn into a valid ``Estimate``.

        This is the cleanest injection point: it leaves the entire downstream
        pipeline (parse → Estimate → fuse → evaluate) unmodified, so the
        backtest exercises the *real* consensus + risk logic, just with
        mocked LLM inputs.
        """
        mock = self._mock_llm

        for analyst in mesh:
            analyst_id = analyst.analyst_id

            # Define an async closure that captures the analyst_id.
            # We bind it as an instance attribute so it shadows the class
            # method on this instance only (other instances / classes are
            # unaffected).
            async def _mock_call(
                messages: list[dict[str, str]],
                config: Any,  # noqa: ARG001 — config unused, MockLLM is deterministic
                _aid: str = analyst_id,
            ) -> str:
                return mock.respond(messages, analyst_id=_aid)

            # mypy: assigning a function to an instance attribute that shadows
            # a decorated method is intentional here.
            analyst._call_llm = _mock_call  # type: ignore[method-assign]

    def _build_consensus_config(self, strategy: dict[str, Any]) -> ConsensusConfig:
        """Build ConsensusConfig from the strategy TOML's [consensus] section."""
        cons_cfg = strategy.get("consensus", {})
        weights = cons_cfg.get("weights")
        # Pydantic v2 ConsensusConfig uses extra="forbid", so we must only
        # pass recognised keys.
        kwargs: dict[str, Any] = {
            "method": cons_cfg.get("method", "logit-mean"),
            "agreement_threshold": float(cons_cfg.get("agreement_threshold", 0.65)),
            "min_analysts": int(cons_cfg.get("min_analysts", 2)),
        }
        if weights is not None:
            kwargs["weights"] = {str(k): float(v) for k, v in weights.items()}
        return ConsensusConfig(**kwargs)

    def _build_risk_engine(self, strategy: dict[str, Any]) -> RiskEngine:
        """Build a RiskEngine from the strategy TOML's [risk] section.

        Fills in defaults for ``cool_down_min_after_loss`` and
        ``market_type_rules`` if the TOML omits them (the strategy TOML is
        minimal; the live-mvp.toml has the full set).
        """
        risk_cfg = strategy.get("risk", {})

        # Default market-type rules — mirror configs/live-mvp.toml.
        default_rules: dict[str, dict[str, Any]] = {
            "politics": {"max_stake_usd": 50.0, "allowed": True},
            "crypto": {"max_stake_usd": 40.0, "allowed": True},
            "sports": {"max_stake_usd": 20.0, "allowed": True},
            "niche": {"max_stake_usd": 30.0, "allowed": True},
            "subjective": {"max_stake_usd": 25.0, "allowed": True},
            "economics": {"max_stake_usd": 40.0, "allowed": True},
        }
        rules_raw = risk_cfg.get("market_type_rules", default_rules)
        market_type_rules = {
            str(k): MarketTypeRules(**v) for k, v in rules_raw.items()
        }

        config = RiskConfig(
            sizing=risk_cfg.get("sizing", "kelly-fractional"),
            kelly_fraction=float(risk_cfg.get("kelly_fraction", 0.25)),
            max_stake_per_market_usd=float(risk_cfg.get("max_stake_per_market_usd", 50.0)),
            max_total_exposure_usd=float(risk_cfg.get("max_total_exposure_usd", 500.0)),
            max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 5.0)),
            cool_down_min_after_loss=float(risk_cfg.get("cool_down_min_after_loss", 30.0)),
            market_type_rules=market_type_rules,
        )
        return RiskEngine(config)

    # ------------------------------------------------------------------ #
    # Per-market helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_market_context(market: HistoricalMarket) -> MarketContext:
        """Reconstruct the MarketContext the mesh would have seen at open.

        Uses ``yes_price_at_open`` as the current YES price (the price the bot
        *would have* seen), and ``closed_at`` as the ``closes_at`` timestamp.
        """
        yes_price = max(_PRICE_EPS, min(1.0 - _PRICE_EPS, market.yes_price_at_open))
        return MarketContext(
            market_id=market.market_id,
            question=market.question,
            category=market.category,
            current_yes_price=yes_price,
            current_no_price=round(1.0 - yes_price, 4),
            volume_usd=market.volume_usd,
            closes_at=market.closed_at.isoformat(),
            metadata={
                "opened_at": market.opened_at.isoformat(),
                "settled_at": market.settled_at.isoformat(),
                "arbiter_model": market.arbiter_model,
            },
        )

    @staticmethod
    def _settle(
        side: str,
        stake_usd: float,
        yes_price: float,
        final_outcome: str,
    ) -> float:
        """Compute realised P&L for a settled binary-outcome trade.

        Math (see pythia-risk sizing.py for the share-price derivation):

        - YES bet of $S at YES price ``m``:
            - Buy ``S / m`` YES shares.
            - If YES wins: payout = ``S / m``, profit = ``S * (1 - m) / m``.
            - If NO wins: payout = 0, profit = ``-S``.

        - NO bet of $S at YES price ``m`` (NO price = ``1 - m``):
            - Buy ``S / (1 - m)`` NO shares.
            - If NO wins: payout = ``S / (1 - m)``, profit = ``S * m / (1 - m)``.
            - If YES wins: payout = 0, profit = ``-S``.

        Prices are clamped to ``[_PRICE_EPS, 1 - _PRICE_EPS]`` to avoid
        division-by-zero on degenerate markets.
        """
        m = max(_PRICE_EPS, min(1.0 - _PRICE_EPS, yes_price))

        if side == "YES":
            if final_outcome == "YES":
                return stake_usd * (1.0 - m) / m
            return -stake_usd
        elif side == "NO":
            if final_outcome == "NO":
                return stake_usd * m / (1.0 - m)
            return -stake_usd
        else:  # pragma: no cover — side is always YES/NO from the risk engine
            raise ValueError(f"invalid side: {side!r}")

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    def _aggregate(
        self,
        *,
        trades: list[_TradeRecord],
        equity_curve: list[tuple[datetime, float]],
        starting_capital: float,
        ending_capital: float,
    ) -> BacktestResult:
        """Turn the per-market trade list into a BacktestResult."""
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl_usd > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        total_return_pct = (
            (ending_capital - starting_capital) / starting_capital * 100.0
            if starting_capital > 0 else 0.0
        )

        # Per-analyst Brier scores.
        brier_scores: dict[str, float] = {}
        all_analysts: set[str] = set()
        for t in trades:
            all_analysts.update(t.estimates.keys())
        for aid in all_analysts:
            brier_scores[aid] = self._compute_brier(aid, trades)

        # Per-category breakdown.
        per_category: dict[str, dict[str, Any]] = {}
        cats: dict[str, list[_TradeRecord]] = defaultdict(list)
        for t in trades:
            cats[t.category].append(t)
        for cat, cat_trades in cats.items():
            cat_wins = sum(1 for t in cat_trades if t.pnl_usd > 0)
            cat_pnl = sum(t.pnl_usd for t in cat_trades)
            # Category Brier = mean over all analysts across all markets in this category.
            cat_brier_scores: list[float] = []
            for t in cat_trades:
                outcome = 1.0 if t.final_outcome == "YES" else 0.0
                for p in t.estimates.values():
                    cat_brier_scores.append((p - outcome) ** 2)
            cat_brier = (
                float(np.mean(cat_brier_scores)) if cat_brier_scores else 0.0
            )
            per_category[cat] = {
                "count": len(cat_trades),
                "win_rate": cat_wins / len(cat_trades) if cat_trades else 0.0,
                "return_pct": cat_pnl / starting_capital * 100.0 if starting_capital > 0 else 0.0,
                "total_pnl_usd": round(cat_pnl, 2),
                "brier": round(cat_brier, 4),
            }

        sharpe = self._compute_sharpe(equity_curve)
        max_dd = self._compute_max_drawdown(equity_curve)

        # Round equity-curve values to 2 decimals (cents) for consistency with
        # the rounded ending_capital_usd. Without this, the last equity point
        # can differ from ending_capital_usd by sub-cent floating-point noise.
        rounded_equity: list[tuple[datetime, float]] = [
            (ts, round(val, 2)) for ts, val in equity_curve
        ]

        return BacktestResult(
            starting_capital_usd=round(starting_capital, 2),
            ending_capital_usd=round(ending_capital, 2),
            total_return_pct=round(total_return_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(max_dd, 4),
            total_trades=total_trades,
            win_rate=round(win_rate, 4),
            brier_scores={k: round(v, 4) for k, v in brier_scores.items()},
            per_category_stats=per_category,
            equity_curve=rounded_equity,
        )

    def _empty_result(self) -> BacktestResult:
        """Return a well-typed zero-trade result for empty market sets."""
        now = datetime.now(UTC)
        return BacktestResult(
            starting_capital_usd=round(self.config.starting_capital_usd, 2),
            ending_capital_usd=round(self.config.starting_capital_usd, 2),
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            total_trades=0,
            win_rate=0.0,
            brier_scores={},
            per_category_stats={},
            equity_curve=[(now, self.config.starting_capital_usd)],
        )

    # ------------------------------------------------------------------ #
    # Metric computations
    # ------------------------------------------------------------------ #

    def _compute_brier(self, analyst_id: str, trades: list[_TradeRecord]) -> float:
        """Mean Brier score for one analyst across all trades it estimated.

        For each trade where ``analyst_id`` gave an estimate::

            outcome = 1.0 if final_outcome == "YES" else 0.0
            brier_i = (p_estimated - outcome)^2

        Returns the mean over all such trades. Returns 0.0 if the analyst
        estimated none of the trades (should not happen in practice — the
        caller only invokes this for analysts seen in at least one trade).
        """
        sq_errors: list[float] = []
        for t in trades:
            if analyst_id not in t.estimates:
                continue
            p = t.estimates[analyst_id]
            outcome = 1.0 if t.final_outcome == "YES" else 0.0
            sq_errors.append((p - outcome) ** 2)
        if not sq_errors:
            return 0.0
        return float(np.mean(sq_errors))

    def _compute_sharpe(self, equity_curve: list[tuple[datetime, float]]) -> float:
        """Annualised Sharpe ratio of the equity curve.

        Steps:
        1. Extract the equity values (the float component of each tuple).
        2. Compute per-step returns: ``r_i = (v_i - v_{i-1}) / v_{i-1}``.
        3. ``sharpe = mean(r) / std(r) * sqrt(252)``.

        Assumes 1 trade/day for annualisation (per the spec). Returns 0.0 if
        fewer than 2 points or std == 0 (no variance → undefined Sharpe).
        """
        if len(equity_curve) < 2:
            return 0.0
        values = np.asarray([v for _, v in equity_curve], dtype=np.float64)
        # Guard against zero / negative equity (would produce inf/nan returns).
        prev = values[:-1]
        curr = values[1:]
        # Avoid div-by-zero: only compute returns where prev > 0.
        mask = prev > 0.0
        if not mask.any():
            return 0.0
        returns = (curr[mask] - prev[mask]) / prev[mask]
        if returns.size < 1:
            return 0.0
        std = float(np.std(returns, ddof=1)) if returns.size >= 2 else 0.0
        if std == 0.0 or not math.isfinite(std):
            return 0.0
        mean = float(np.mean(returns))
        return mean / std * math.sqrt(_TRADING_DAYS_PER_YEAR)

    def _compute_max_drawdown(
        self, equity_curve: list[tuple[datetime, float]]
    ) -> float:
        """Peak-to-trough drawdown percentage on the equity curve.

        ``max_dd = max over t of (peak_t - equity_t) / peak_t * 100``

        where ``peak_t`` is the running max up to and including ``t``.
        Returns 0.0 if the curve is empty or monotonically non-decreasing.
        """
        if not equity_curve:
            return 0.0
        values = [v for _, v in equity_curve]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
        return float(max_dd)


# ---------------------------------------------------------------------------
# Convenience function — one-call backtest.
# ---------------------------------------------------------------------------


async def run_backtest(config: BacktestConfig) -> BacktestResult:
    """Run a backtest in one call.

    Equivalent to::

        await Backtester(config).run()

    Provided as a top-level export so callers don't need to import the
    ``Backtester`` class for the common case.
    """
    return await Backtester(config).run()


__all__ = ["Backtester", "run_backtest"]
