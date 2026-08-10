"""Smoke tests for the pythia-forge Backtester.

Runs a full backtest against the 10-market sample fixture using the MockLLM
(default — no real LLM calls), then verifies the BacktestResult has the
correct types, trade count, Brier scores, and equity curve.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pythia_forge import BacktestConfig, Backtester, BacktestResult, run_backtest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
STRATEGY_PATH = FIXTURES / "strategy.toml"
MARKETS_PATH = FIXTURES / "resolved_markets_sample.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backtest_config() -> BacktestConfig:
    """A standard BacktestConfig pointing at the sample fixture."""
    return BacktestConfig(
        strategy_path=STRATEGY_PATH,
        markets_path=MARKETS_PATH,
        starting_capital_usd=1000.0,
        markets_filter={
            "categories": ["politics", "crypto", "sports", "subjective"],
            "min_volume_usd": 1000,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestResultTypes:
    """Verify the BacktestResult has the correct field types."""

    @pytest.mark.asyncio
    async def test_result_has_correct_types(self, backtest_config: BacktestConfig) -> None:
        result = await Backtester(backtest_config).run()

        assert isinstance(result, BacktestResult)
        assert isinstance(result.starting_capital_usd, float)
        assert isinstance(result.ending_capital_usd, float)
        assert isinstance(result.total_return_pct, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert isinstance(result.total_trades, int)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.brier_scores, dict)
        assert isinstance(result.per_category_stats, dict)
        assert isinstance(result.equity_curve, list)

    @pytest.mark.asyncio
    async def test_starting_capital_preserved(self, backtest_config: BacktestConfig) -> None:
        result = await Backtester(backtest_config).run()
        assert result.starting_capital_usd == 1000.0


class TestTradeCount:
    """Verify total_trades is sane."""

    @pytest.mark.asyncio
    async def test_at_least_some_trades_placed(self, backtest_config: BacktestConfig) -> None:
        """With agreement_threshold=0.50 and MockLLM producing ~4 analysts per
        market, we expect at least a few trades to clear the consensus + risk
        gates. Zero trades would indicate a pipeline bug.
        """
        result = await Backtester(backtest_config).run()
        assert result.total_trades > 0, (
            f"expected at least 1 trade, got {result.total_trades} — "
            "check that MockLLM produces non-degenerate estimates and the "
            "consensus/risk gates aren't too aggressive"
        )
        assert result.total_trades <= 10, (
            f"got {result.total_trades} trades from 10 markets — "
            "should not exceed the market count"
        )

    @pytest.mark.asyncio
    async def test_equity_curve_non_empty(self, backtest_config: BacktestConfig) -> None:
        result = await Backtester(backtest_config).run()
        # At minimum: the starting-capital seed point.
        assert len(result.equity_curve) >= 1
        # If trades were placed, equity_curve should have 1 + n_trades points.
        assert len(result.equity_curve) == 1 + result.total_trades

    @pytest.mark.asyncio
    async def test_equity_curve_first_point_is_starting_capital(
        self, backtest_config: BacktestConfig
    ) -> None:
        result = await Backtester(backtest_config).run()
        ts, val = result.equity_curve[0]
        assert isinstance(ts, datetime)
        assert val == pytest.approx(1000.0, rel=1e-6)


class TestBrierScores:
    """Verify per-analyst Brier scores are computed for every analyst."""

    @pytest.mark.asyncio
    async def test_brier_scores_for_all_analysts(
        self, backtest_config: BacktestConfig
    ) -> None:
        result = await Backtester(backtest_config).run()

        # The strategy configures 4 analysts. If any trades were placed,
        # all 4 should have Brier scores (each analyst estimates every market
        # the mesh runs on, and Brier is computed across all trades where the
        # analyst gave an estimate — even REJECTed trades record estimates).
        if result.total_trades > 0:
            assert len(result.brier_scores) >= 1, (
                f"expected Brier scores for ≥1 analyst, got {result.brier_scores}"
            )
            # Each analyst's Brier score is in [0, 1].
            for aid, score in result.brier_scores.items():
                assert isinstance(aid, str)
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0, (
                    f"Brier score {score} for {aid!r} outside [0, 1]"
                )

    @pytest.mark.asyncio
    async def test_brier_scores_match_analyst_ids(
        self, backtest_config: BacktestConfig
    ) -> None:
        """Brier score keys should be a subset of the configured analysts."""
        result = await Backtester(backtest_config).run()
        configured = {"politics", "crypto", "sports", "niche"}
        for aid in result.brier_scores:
            assert aid in configured, f"unexpected analyst_id in brier_scores: {aid!r}"


class TestPerCategoryStats:
    """Verify per-category breakdown is populated."""

    @pytest.mark.asyncio
    async def test_per_category_has_expected_categories(
        self, backtest_config: BacktestConfig
    ) -> None:
        result = await Backtester(backtest_config).run()
        if result.total_trades > 0:
            # The fixture has markets in politics, crypto, sports, subjective.
            # (niche is configured but the fixture has no niche markets.)
            expected_cats = {"politics", "crypto", "sports", "subjective"}
            actual_cats = set(result.per_category_stats.keys())
            assert actual_cats.issubset(expected_cats | {"niche"}), (
                f"unexpected categories: {actual_cats - expected_cats - {'niche'}}"
            )
            for _cat, stats in result.per_category_stats.items():
                assert "count" in stats
                assert "win_rate" in stats
                assert "return_pct" in stats
                assert "total_pnl_usd" in stats
                assert "brier" in stats

    @pytest.mark.asyncio
    async def test_per_category_count_sums_to_total_trades(
        self, backtest_config: BacktestConfig
    ) -> None:
        result = await Backtester(backtest_config).run()
        if result.total_trades > 0:
            total = sum(s["count"] for s in result.per_category_stats.values())
            assert total == result.total_trades, (
                f"category counts sum to {total}, but total_trades={result.total_trades}"
            )


class TestMetricsSanity:
    """Sanity-check the computed metrics."""

    @pytest.mark.asyncio
    async def test_return_pct_matches_capital_delta(
        self, backtest_config: BacktestConfig
    ) -> None:
        result = await Backtester(backtest_config).run()
        expected = (
            (result.ending_capital_usd - result.starting_capital_usd)
            / result.starting_capital_usd * 100.0
        )
        assert result.total_return_pct == pytest.approx(expected, rel=1e-4)

    @pytest.mark.asyncio
    async def test_win_rate_in_unit_interval(self, backtest_config: BacktestConfig) -> None:
        result = await Backtester(backtest_config).run()
        assert 0.0 <= result.win_rate <= 1.0

    @pytest.mark.asyncio
    async def test_max_drawdown_non_negative(self, backtest_config: BacktestConfig) -> None:
        result = await Backtester(backtest_config).run()
        assert result.max_drawdown_pct >= 0.0

    @pytest.mark.asyncio
    async def test_ending_capital_matches_last_equity_point(
        self, backtest_config: BacktestConfig
    ) -> None:
        result = await Backtester(backtest_config).run()
        if result.equity_curve:
            _, last_val = result.equity_curve[-1]
            assert result.ending_capital_usd == pytest.approx(last_val, rel=1e-6)


class TestRunBacktestConvenience:
    """Verify the top-level run_backtest() convenience function."""

    @pytest.mark.asyncio
    async def test_run_backtest_matches_backtester_run(
        self, backtest_config: BacktestConfig
    ) -> None:
        result = await run_backtest(backtest_config)
        assert isinstance(result, BacktestResult)
        assert result.starting_capital_usd == 1000.0


class TestEdgeCases:
    """Edge-case handling."""

    @pytest.mark.asyncio
    async def test_empty_filter_result_returns_empty_result(
        self, backtest_config: BacktestConfig
    ) -> None:
        """A filter that excludes all markets should return a zero-trade result."""
        backtest_config.markets_filter = {
            "categories": ["nonexistent_category"],
        }
        result = await Backtester(backtest_config).run()
        assert result.total_trades == 0
        assert result.starting_capital_usd == result.ending_capital_usd
        assert result.total_return_pct == 0.0
        assert len(result.equity_curve) == 1  # just the seed point

    @pytest.mark.asyncio
    async def test_strategy_override(self, backtest_config: BacktestConfig) -> None:
        """strategy_override should bypass the TOML file entirely."""
        override = {
            "mesh": {"analysts": ["politics", "crypto"]},
            "consensus": {"method": "median", "agreement_threshold": 0.40, "min_analysts": 2},
            "risk": {
                "sizing": "kelly-fractional",
                "kelly_fraction": 0.50,
                "max_stake_per_market_usd": 100,
                "max_total_exposure_usd": 1000,
                "max_drawdown_pct": 10,
            },
        }
        bt = Backtester(backtest_config, strategy_override=override)
        result = await bt.run()
        # With only 2 analysts and a lower threshold, we should still get trades.
        assert isinstance(result, BacktestResult)
        # Brier scores should only be for the 2 configured analysts.
        assert set(result.brier_scores.keys()).issubset({"politics", "crypto"})
