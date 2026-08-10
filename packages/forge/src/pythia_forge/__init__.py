"""pythia-forge — backtest harness + CI deploy pipeline for the Pythia mesh.

Wraps `icohangar-ops/forge` and adds:
    - Backtester     — replays resolved Delphi markets through mesh → consensus → risk.
    - BacktestResult — aggregated P&L + calibration metrics from a backtest run.
    - BacktestConfig — strategy path + markets path + starting capital + filter.
    - run_backtest   — convenience: config → result in one call.

See `pythia_forge.backtester` for the harness, `pythia_forge.report` for the
markdown + PNG report generator, and `pythia_forge.cli` for the CLI entry point.
"""

from __future__ import annotations

from .backtester import Backtester, run_backtest
from .types import BacktestConfig, BacktestResult, HistoricalMarket

__version__ = "0.1.0"

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "HistoricalMarket",
    "run_backtest",
    "__version__",
]
