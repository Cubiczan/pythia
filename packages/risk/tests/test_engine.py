"""Tests for pythia_risk.engine.RiskEngine — the 6-gate evaluation pipeline.

Covers each gate's APPROVE / REJECT behaviour:
    - test_approve_when_consensus_exceeds_price  (happy path, side=YES)
    - test_reject_no_edge_when_consensus_equals_price
    - test_reject_drawdown_breaker
    - test_reject_cool_down
    - test_reject_market_type_not_allowed
    - test_exposure_cap_reduces_size

Also covers side-flip to NO when consensus_prob < market.yes_price, and the
state-update helpers (record_loss / record_win / update_state).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pythia_risk.engine import RiskEngine
from pythia_risk.types import (
    BankrollState,
    ConsensusDecision,
    Market,
    MarketTypeRules,
    RiskConfig,
    TradeReceipt,
)


# --- Fixtures -----------------------------------------------------------------

def _default_config() -> RiskConfig:
    return RiskConfig(
        sizing="kelly-fractional",
        kelly_fraction=0.25,
        max_stake_per_market_usd=50.0,
        max_total_exposure_usd=500.0,
        max_drawdown_pct=5.0,
        cool_down_min_after_loss=30.0,
        market_type_rules={
            "politics": MarketTypeRules(max_stake_usd=50.0, allowed=True),
            "crypto": MarketTypeRules(max_stake_usd=40.0, allowed=True),
            "sports": MarketTypeRules(max_stake_usd=20.0, allowed=True),
            "niche": MarketTypeRules(max_stake_usd=30.0, allowed=True),
            "blocked": MarketTypeRules(max_stake_usd=50.0, allowed=False),
        },
    )


def _default_bankroll(current: float = 1000.0) -> BankrollState:
    return BankrollState(
        cash_usd=current,
        open_positions_usd=0.0,
        peak_bankroll_usd=current,
        current_bankroll_usd=current,
        drawdown_pct=0.0,
        last_loss_at=None,
    )


def _decision(prob: float = 0.72, market_id: str = "mkt-1") -> ConsensusDecision:
    return ConsensusDecision(
        market_id=market_id,
        consensus_prob=prob,
        agreement_score=0.81,
        gate="trade",
        contributor_ids=["politics", "crypto", "niche"],
        method="logit-mean",
        timestamp="2026-08-10T12:00:00Z",
    )


def _market(yes_price: float = 0.55, category: str = "politics", market_id: str = "mkt-1") -> Market:
    return Market(
        market_id=market_id,
        yes_price=yes_price,
        category=category,
        question="Will the Fed cut rates in September?",
    )


# --- Happy path ---------------------------------------------------------------

def test_approve_when_consensus_exceeds_price() -> None:
    """consensus_prob > yes_price (beyond tolerance) → APPROVE, side=YES, size>0."""
    engine = RiskEngine(_default_config())
    plan = engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=_default_bankroll(1000.0),
    )
    assert plan.decision == "APPROVE"
    assert plan.side == "YES"
    assert plan.size_usd > 0.0
    assert plan.size_usd <= 50.0  # max_stake_per_market_usd cap
    assert plan.risk_flags == []
    assert "APPROVE" in plan.rationale or plan.decision == "APPROVE"


def test_approve_flips_to_no_when_consensus_below_price() -> None:
    """consensus_prob < yes_price (beyond tolerance) → APPROVE, side=NO."""
    engine = RiskEngine(_default_config())
    plan = engine.evaluate(
        decision=_decision(prob=0.30),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=_default_bankroll(1000.0),
    )
    assert plan.decision == "APPROVE"
    assert plan.side == "NO"
    assert plan.size_usd > 0.0


# --- Gate: no_edge ------------------------------------------------------------

def test_reject_no_edge_when_consensus_equals_price() -> None:
    """|consensus_prob - yes_price| < 0.02 → REJECT, flag=no_edge."""
    engine = RiskEngine(_default_config())
    plan = engine.evaluate(
        decision=_decision(prob=0.55),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=_default_bankroll(1000.0),
    )
    assert plan.decision == "REJECT"
    assert plan.size_usd == 0.0
    assert "no_edge" in plan.risk_flags


def test_reject_no_edge_within_tolerance() -> None:
    """1pp difference is still within the 2pp tolerance → no_edge."""
    engine = RiskEngine(_default_config())
    plan = engine.evaluate(
        decision=_decision(prob=0.56),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=_default_bankroll(1000.0),
    )
    assert plan.decision == "REJECT"
    assert "no_edge" in plan.risk_flags


# --- Gate: drawdown_breaker ---------------------------------------------------

def test_reject_drawdown_breaker() -> None:
    """drawdown_pct >= max_drawdown_pct → REJECT, flag=drawdown_breaker."""
    engine = RiskEngine(_default_config())
    bankroll = _default_bankroll(1000.0)
    # 6% drawdown > 5% threshold
    bankroll.drawdown_pct = 6.0
    plan = engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=bankroll,
    )
    assert plan.decision == "REJECT"
    assert "drawdown_breaker" in plan.risk_flags
    assert plan.size_usd == 0.0


# --- Gate: cool_down_active ---------------------------------------------------

def test_reject_cool_down() -> None:
    """last_loss_at < cool_down_min ago → REJECT, flag=cool_down_active."""
    engine = RiskEngine(_default_config())
    bankroll = _default_bankroll(1000.0)
    bankroll.last_loss_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    # cool_down_min_after_loss=30, so 10 min ago is still in cool-down.
    plan = engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=bankroll,
    )
    assert plan.decision == "REJECT"
    assert "cool_down_active" in plan.risk_flags


def test_cool_down_clears_after_threshold() -> None:
    """After cool_down_min_after_loss has elapsed, trading is allowed again."""
    engine = RiskEngine(_default_config())
    bankroll = _default_bankroll(1000.0)
    bankroll.last_loss_at = datetime.now(timezone.utc) - timedelta(minutes=45)
    plan = engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=bankroll,
    )
    assert plan.decision == "APPROVE"
    assert "cool_down_active" not in plan.risk_flags


# --- Gate: market_type_not_allowed --------------------------------------------

def test_reject_market_type_not_allowed() -> None:
    """category has allowed=False → REJECT, flag=market_type_not_allowed."""
    engine = RiskEngine(_default_config())
    plan = engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="blocked"),
        current_bankroll=_default_bankroll(1000.0),
    )
    assert plan.decision == "REJECT"
    assert "market_type_not_allowed" in plan.risk_flags


def test_reject_unknown_category() -> None:
    """category not in market_type_rules → REJECT (treated as not allowed)."""
    engine = RiskEngine(_default_config())
    plan = engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="nonexistent"),
        current_bankroll=_default_bankroll(1000.0),
    )
    assert plan.decision == "REJECT"
    assert "market_type_not_allowed" in plan.risk_flags


# --- Gate: exposure_cap -------------------------------------------------------

def test_exposure_cap_reduces_size() -> None:
    """When open_positions + proposed_stake > max_total_exposure_usd,
    size is reduced to available headroom and flag=exposure_cap is set."""
    engine = RiskEngine(_default_config())
    # max_total_exposure_usd = 500, open_positions = 490 → headroom = 10.
    bankroll = _default_bankroll(1000.0)
    bankroll.open_positions_usd = 490.0
    # consensus 0.72, price 0.55 → quarter-Kelly f = 0.25 * (0.72-0.55)/0.45
    #                                 = 0.25 * 0.3778 = 0.0944
    # stake = 0.0944 * 1000 = 94.4, but cap = 50. After exposure gate: 10.
    plan = engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=bankroll,
    )
    assert plan.decision == "APPROVE"
    assert "exposure_cap" in plan.risk_flags
    assert plan.size_usd == pytest.approx(10.0, abs=0.01)


def test_exposure_cap_rejects_when_exhausted() -> None:
    """When headroom == 0, REJECT with flag=exposure_cap."""
    engine = RiskEngine(_default_config())
    bankroll = _default_bankroll(1000.0)
    bankroll.open_positions_usd = 500.0  # == max_total_exposure_usd
    plan = engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=bankroll,
    )
    assert plan.decision == "REJECT"
    assert "exposure_cap" in plan.risk_flags
    assert plan.size_usd == 0.0


# --- Per-market cap (silent reduce) -------------------------------------------

def test_per_market_cap_reduces_size() -> None:
    """Stake is capped at market_type_rules[category].max_stake_usd."""
    engine = RiskEngine(_default_config())
    # sports cap is 20. consensus 0.9, price 0.1, bankroll 100k → uncapped
    # stake would be huge, but the cap should clamp it to 20.
    bankroll = _default_bankroll(100_000.0)
    plan = engine.evaluate(
        decision=_decision(prob=0.90),
        market=_market(yes_price=0.10, category="sports"),
        current_bankroll=bankroll,
    )
    assert plan.decision == "APPROVE"
    assert plan.size_usd <= 20.0  # sports category cap


# --- State mutations ----------------------------------------------------------

def test_record_loss_arms_cool_down() -> None:
    """record_loss sets last_loss_at, so the next evaluate() hits cool_down."""
    engine = RiskEngine(_default_config())
    # Prime the state to a known value.
    engine.state = _default_bankroll(1000.0)
    engine.record_loss(market_id="mkt-X", loss_usd=50.0)
    assert engine.state.last_loss_at is not None
    # current_bankroll drops by 50
    assert engine.state.current_bankroll_usd == pytest.approx(950.0, abs=0.01)


def test_record_win_updates_peak() -> None:
    """record_win moves peak_bankroll_usd if current exceeds it."""
    engine = RiskEngine(_default_config())
    engine.state = _default_bankroll(1000.0)
    engine.record_win(market_id="mkt-Y", win_usd=200.0)
    assert engine.state.current_bankroll_usd == pytest.approx(1200.0, abs=0.01)
    assert engine.state.peak_bankroll_usd == pytest.approx(1200.0, abs=0.01)
    assert engine.state.drawdown_pct == 0.0


def test_update_state_moves_cash_to_positions() -> None:
    """update_state moves size from cash to open_positions."""
    engine = RiskEngine(_default_config())
    engine.state = _default_bankroll(1000.0)
    receipt = TradeReceipt(
        market_id="mkt-Z",
        side="YES",
        size_usd=100.0,
        fill_price=0.55,
        att_order_id="ord-1",
        signed_by="key-1",
        timestamp="2026-08-10T12:00:00Z",
        audit_log_path="./logs/audit.jsonl",
    )
    engine.update_state(receipt)
    assert engine.state.cash_usd == pytest.approx(900.0, abs=0.01)
    assert engine.state.open_positions_usd == pytest.approx(100.0, abs=0.01)
    assert engine.state.current_bankroll_usd == pytest.approx(1000.0, abs=0.01)


def test_drawdown_recompute_after_loss() -> None:
    """After a loss, drawdown_pct = (peak - current) / peak * 100."""
    engine = RiskEngine(_default_config())
    engine.state = BankrollState(
        cash_usd=1000.0,
        open_positions_usd=0.0,
        peak_bankroll_usd=1000.0,
        current_bankroll_usd=1000.0,
        drawdown_pct=0.0,
        last_loss_at=None,
    )
    engine.record_loss(market_id="mkt-X", loss_usd=100.0)
    # (1000 - 900) / 1000 * 100 = 10
    assert engine.state.drawdown_pct == pytest.approx(10.0, abs=0.01)


# --- Engine.evaluate is pure w.r.t. bankroll parameter ------------------------

def test_evaluate_does_not_mutate_input_bankroll() -> None:
    """evaluate must not mutate the bankroll passed in (purity for replay)."""
    engine = RiskEngine(_default_config())
    bankroll = _default_bankroll(1000.0)
    original_cash = bankroll.cash_usd
    original_positions = bankroll.open_positions_usd
    original_last_loss = bankroll.last_loss_at
    engine.evaluate(
        decision=_decision(prob=0.72),
        market=_market(yes_price=0.55, category="politics"),
        current_bankroll=bankroll,
    )
    assert bankroll.cash_usd == original_cash
    assert bankroll.open_positions_usd == original_positions
    assert bankroll.last_loss_at == original_last_loss
