"""Tests for pythia_risk.sizing — Kelly criterion math for binary markets.

Covers:
    - Positive edge → positive stake
    - No edge (p == m) → stake 0
    - Negative edge (p < m) → stake 0 (clamped, no shorting)
    - Stake capped at max_stake_usd
    - Quarter-Kelly (fraction=0.25) reduces stake vs full Kelly (fraction=1.0)
"""

from __future__ import annotations

import pytest

from pythia_risk.sizing import (
    _compute_kelly_fraction,
    kelly_fraction,
    size_trade_fixed,
    size_trade_kelly,
)

# --- kelly_fraction -----------------------------------------------------------

def test_kelly_basic() -> None:
    """p=0.7, m=0.5 → positive full-Kelly fraction; quarter-Kelly is 1/4 of it."""
    full = _compute_kelly_fraction(0.7, 0.5, fraction=1.0)
    # f* = (p - m) / (1 - m) = (0.7 - 0.5) / 0.5 = 0.4
    assert full == pytest.approx(0.4, abs=1e-9)

    quarter = kelly_fraction(0.7, 0.5, fraction=0.25)
    assert quarter == pytest.approx(0.1, abs=1e-9)
    assert quarter > 0.0

def test_kelly_no_edge() -> None:
    """p == m → full-Kelly = 0, fractional = 0."""
    assert _compute_kelly_fraction(0.5, 0.5, fraction=1.0) == 0.0
    assert kelly_fraction(0.5, 0.5, fraction=0.25) == 0.0

def test_kelly_negative_edge() -> None:
    """p < m → full-Kelly negative, but clamped to 0 (no shorting)."""
    # f* = (0.3 - 0.5) / 0.5 = -0.4, clamped to 0
    full = _compute_kelly_fraction(0.3, 0.5, fraction=1.0)
    assert full == 0.0
    assert kelly_fraction(0.3, 0.5, fraction=0.25) == 0.0

def test_kelly_at_extremes() -> None:
    """Very high edge is clamped to fraction * 1.0 (no leverage)."""
    # p=0.99, m=0.01 → f* = 0.98/0.99 ≈ 0.989898..., clamped to 1.0
    full = _compute_kelly_fraction(0.99, 0.01, fraction=1.0)
    assert full == pytest.approx(0.989898, abs=1e-4)
    assert full <= 1.0

    # m=1.0 (denominator would be 0) → guarded, returns fraction * (p-1)/eps
    # For p=0.99, m=1.0: f* = (0.99-1.0)/1e-6 = -100000, clamped to 0
    guarded = _compute_kelly_fraction(0.99, 1.0, fraction=0.25)
    assert guarded == 0.0

def test_kelly_validates_inputs() -> None:
    """Out-of-range inputs raise ValueError."""
    with pytest.raises(ValueError):
        kelly_fraction(1.5, 0.5)
    with pytest.raises(ValueError):
        kelly_fraction(0.5, -0.1)
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 0.5, fraction=0.0)
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 0.5, fraction=1.5)

# --- size_trade_kelly ---------------------------------------------------------

def test_size_trade_kelly_basic() -> None:
    """p=0.7, m=0.5, bankroll=$1000, quarter-Kelly → stake = 0.1 * 1000 = $100."""
    stake = size_trade_kelly(
        p_consensus=0.7,
        market_price=0.5,
        bankroll_usd=1000.0,
        kelly_fraction=0.25,
        max_stake_usd=50.0,
    )
    # 0.1 * 1000 = 100, but capped at max_stake_usd=50
    assert stake == 50.0

def test_kelly_capped_at_max_stake() -> None:
    """Even with a huge bankroll, stake never exceeds max_stake_usd."""
    stake = size_trade_kelly(
        p_consensus=0.9,
        market_price=0.1,
        bankroll_usd=1_000_000.0,
        kelly_fraction=0.25,
        max_stake_usd=50.0,
    )
    assert stake == 50.0

def test_quarter_kelly_reduces_stake() -> None:
    """Quarter-Kelly (0.25) gives exactly 1/4 of full-Kelly (1.0) stake."""
    full = size_trade_kelly(
        p_consensus=0.7,
        market_price=0.5,
        bankroll_usd=1000.0,
        kelly_fraction=1.0,
        max_stake_usd=10_000.0,  # high cap so the cap doesn't interfere
    )
    quarter = size_trade_kelly(
        p_consensus=0.7,
        market_price=0.5,
        bankroll_usd=1000.0,
        kelly_fraction=0.25,
        max_stake_usd=10_000.0,
    )
    # full: f* = 0.4 → stake = 0.4 * 1000 = 400
    assert full == pytest.approx(400.0, abs=0.01)
    # quarter: f = 0.1 → stake = 0.1 * 1000 = 100
    assert quarter == pytest.approx(100.0, abs=0.01)
    assert quarter == pytest.approx(full / 4.0, abs=0.01)

def test_size_trade_kelly_no_edge_returns_zero() -> None:
    """No edge → 0 stake."""
    stake = size_trade_kelly(
        p_consensus=0.5,
        market_price=0.5,
        bankroll_usd=1000.0,
        kelly_fraction=0.25,
        max_stake_usd=50.0,
    )
    assert stake == 0.0

# --- size_trade_fixed ---------------------------------------------------------

def test_size_trade_fixed_capped() -> None:
    """Fixed stake is capped at max_stake_usd."""
    assert size_trade_fixed(fixed_stake_usd=100.0, max_stake_usd=50.0) == 50.0
    assert size_trade_fixed(fixed_stake_usd=30.0, max_stake_usd=50.0) == 30.0
    assert size_trade_fixed(fixed_stake_usd=0.0, max_stake_usd=50.0) == 0.0
