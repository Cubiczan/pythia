"""Tests for the pythia-forge report generator.

Builds a mock BacktestResult, generates a report, and verifies the markdown
file and equity-curve PNG are both written and contain the expected sections.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from pythia_forge.report import generate_report, plot_equity_curve
from pythia_forge.types import BacktestResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_result() -> BacktestResult:
    """A realistic-looking BacktestResult for report testing."""
    base_ts = datetime(2025, 10, 1, tzinfo=UTC)
    return BacktestResult(
        starting_capital_usd=1000.0,
        ending_capital_usd=1187.50,
        total_return_pct=18.75,
        sharpe_ratio=1.42,
        max_drawdown_pct=3.20,
        total_trades=8,
        win_rate=0.625,
        brier_scores={
            "politics": 0.1542,
            "crypto": 0.2103,
            "sports": 0.3310,
            "niche": 0.1875,
        },
        per_category_stats={
            "politics": {
                "count": 3,
                "win_rate": 0.667,
                "return_pct": 9.50,
                "total_pnl_usd": 95.00,
                "brier": 0.1542,
            },
            "crypto": {
                "count": 2,
                "win_rate": 0.500,
                "return_pct": 4.20,
                "total_pnl_usd": 42.00,
                "brier": 0.2103,
            },
            "sports": {
                "count": 2,
                "win_rate": 0.500,
                "return_pct": 1.80,
                "total_pnl_usd": 18.00,
                "brier": 0.3310,
            },
            "subjective": {
                "count": 1,
                "win_rate": 1.0,
                "return_pct": 3.25,
                "total_pnl_usd": 32.50,
                "brier": 0.1875,
            },
        },
        equity_curve=[
            (base_ts, 1000.0),
            (datetime(2025, 12, 16, tzinfo=UTC), 1045.0),
            (datetime(2026, 1, 1, tzinfo=UTC), 1010.0),
            (datetime(2026, 1, 1, tzinfo=UTC), 1052.0),
            (datetime(2026, 2, 9, tzinfo=UTC), 1020.0),
            (datetime(2026, 6, 30, tzinfo=UTC), 1090.0),
            (datetime(2026, 1, 1, tzinfo=UTC), 1075.0),
            (datetime(2026, 6, 23, tzinfo=UTC), 1110.0),
            (datetime(2026, 8, 1, tzinfo=UTC), 1155.0),
        ],
    )


@pytest.fixture
def empty_result() -> BacktestResult:
    """A zero-trade result for testing the empty case."""
    now = datetime.now(UTC)
    return BacktestResult(
        starting_capital_usd=1000.0,
        ending_capital_usd=1000.0,
        total_return_pct=0.0,
        sharpe_ratio=0.0,
        max_drawdown_pct=0.0,
        total_trades=0,
        win_rate=0.0,
        brier_scores={},
        per_category_stats={},
        equity_curve=[(now, 1000.0)],
    )


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Verify generate_report writes the expected files."""

    def test_writes_markdown_file(self, tmp_path: Path, mock_result: BacktestResult) -> None:
        report_path = tmp_path / "backtest.md"
        generate_report(mock_result, report_path)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_writes_equity_curve_png(self, tmp_path: Path, mock_result: BacktestResult) -> None:
        report_path = tmp_path / "backtest.md"
        generate_report(mock_result, report_path)
        png_path = report_path.with_suffix(".png")
        assert png_path.exists()
        assert png_path.stat().st_size > 1000  # non-trivial PNG

    def test_creates_parent_directories(
        self, tmp_path: Path, mock_result: BacktestResult
    ) -> None:
        report_path = tmp_path / "nested" / "deep" / "report.md"
        generate_report(mock_result, report_path)
        assert report_path.exists()

    def test_markdown_contains_executive_summary(
        self, tmp_path: Path, mock_result: BacktestResult
    ) -> None:
        report_path = tmp_path / "backtest.md"
        generate_report(mock_result, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "## Executive Summary" in content
        assert "Starting capital" in content
        assert "Total return" in content
        assert "Sharpe ratio" in content
        assert "Max drawdown" in content
        assert "Win rate" in content

    def test_markdown_contains_brier_table(
        self, tmp_path: Path, mock_result: BacktestResult
    ) -> None:
        report_path = tmp_path / "backtest.md"
        generate_report(mock_result, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "## Per-Analyst Brier Scores" in content
        assert "politics" in content
        assert "crypto" in content
        assert "sports" in content
        assert "niche" in content
        # Brier table should be sorted ascending (politics has lowest = 0.1542).
        politics_idx = content.index("politics")
        sports_idx = content.index("sports")
        assert politics_idx < sports_idx, "politics (lowest Brier) should appear before sports"

    def test_markdown_contains_category_table(
        self, tmp_path: Path, mock_result: BacktestResult
    ) -> None:
        report_path = tmp_path / "backtest.md"
        generate_report(mock_result, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "## Per-Category Breakdown" in content
        for cat in ["politics", "crypto", "sports", "subjective"]:
            assert cat in content

    def test_markdown_contains_equity_curve_reference(
        self, tmp_path: Path, mock_result: BacktestResult
    ) -> None:
        report_path = tmp_path / "backtest.md"
        generate_report(mock_result, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "## Equity Curve" in content
        assert "![equity curve]" in content
        assert ".png" in content

    def test_markdown_contains_recommendations(
        self, tmp_path: Path, mock_result: BacktestResult
    ) -> None:
        report_path = tmp_path / "backtest.md"
        generate_report(mock_result, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "## Recommendations" in content
        # sports has Brier 0.331 > 0.30 → should trigger a down-weight recommendation.
        assert "sports" in content
        assert "Down-weight" in content or "Up-weight" in content

    def test_empty_result_renders_without_error(
        self, tmp_path: Path, empty_result: BacktestResult
    ) -> None:
        report_path = tmp_path / "empty.md"
        generate_report(empty_result, report_path)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "## Executive Summary" in content
        # The Brier table should note "no analysts".
        assert "no analysts" in content or "—" in content


# ---------------------------------------------------------------------------
# Equity-curve plotting tests
# ---------------------------------------------------------------------------


class TestPlotEquityCurve:
    """Verify the equity-curve PNG renderer."""

    def test_writes_png_file(self, tmp_path: Path, mock_result: BacktestResult) -> None:
        png_path = tmp_path / "equity.png"
        plot_equity_curve(mock_result.equity_curve, png_path)
        assert png_path.exists()
        assert png_path.stat().st_size > 1000

    def test_creates_parent_directories(
        self, tmp_path: Path, mock_result: BacktestResult
    ) -> None:
        png_path = tmp_path / "charts" / "equity.png"
        plot_equity_curve(mock_result.equity_curve, png_path)
        assert png_path.exists()

    def test_empty_curve_renders_placeholder(
        self, tmp_path: Path
    ) -> None:
        """An empty equity_curve should render a 'No equity data' placeholder,
        not crash.
        """
        png_path = tmp_path / "empty.png"
        plot_equity_curve([], png_path)
        assert png_path.exists()
        assert png_path.stat().st_size > 500

    def test_single_point_curve_renders(
        self, tmp_path: Path
    ) -> None:
        """A single-point curve (starting capital only) should render."""
        png_path = tmp_path / "single.png"
        plot_equity_curve(
            [(datetime.now(UTC), 1000.0)],
            png_path,
        )
        assert png_path.exists()


# ---------------------------------------------------------------------------
# BacktestResult model validation
# ---------------------------------------------------------------------------


class TestBacktestResultModel:
    """Verify the BacktestResult pydantic model enforces its contract."""

    def test_valid_result_parses(self, mock_result: BacktestResult) -> None:
        # Round-trip through model_dump / model_validate to confirm serialisation.
        data = mock_result.model_dump()
        restored = BacktestResult.model_validate(data)
        assert restored.total_trades == mock_result.total_trades
        assert restored.sharpe_ratio == mock_result.sharpe_ratio

    def test_negative_total_trades_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BacktestResult(
                starting_capital_usd=1000.0,
                ending_capital_usd=1000.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                total_trades=-1,  # invalid
                win_rate=0.0,
                brier_scores={},
                per_category_stats={},
                equity_curve=[],
            )
