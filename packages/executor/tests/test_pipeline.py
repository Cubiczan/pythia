"""Pipeline tests for PythiaExecutor.

All four sibling components (DelphiClient, BaseAnalyst, ConsensusEngine,
RiskEngine) are replaced with in-test stubs so no real ATT / LLM calls
are made. Each test exercises one branch of the 12-step pipeline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pythia_analyst_mesh import Estimate, MarketContext
from pythia_consensus import ConsensusConfig, ConsensusDecision
from pythia_delphi_adapter import Market as AdapterMarket
from pythia_delphi_adapter import (
    MarketStatus,
    OrderSide,
    TradeReceipt,
)
from pythia_delphi_adapter.models import MarketCategory, OrderStatus
from pythia_risk import BankrollState, TradePlan
from pythia_risk import Market as RiskMarket
from pythia_risk import TradeReceipt as RiskTradeReceipt

from pythia_executor import ExecutorConfig, PythiaExecutor

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubAnalyst:
    """Minimal stand-in for BaseAnalyst that returns a pre-configured Estimate."""

    def __init__(self, analyst_id: str, estimate: Estimate | None = None) -> None:
        self.analyst_id = analyst_id
        self._estimate = estimate

    async def estimate(self, market: MarketContext) -> Estimate:
        if self._estimate is None:
            return Estimate(
                market_id=market.market_id,
                probability=0.6,
                confidence=0.7,
                rationale=f"stub estimate from {self.analyst_id}",
                evidence=[],
                analyst_id=self.analyst_id,
            )
        return self._estimate


class StubConsensusEngine:
    """Stand-in for ConsensusEngine that returns a pre-configured decision."""

    def __init__(self, decision: ConsensusDecision, config: ConsensusConfig) -> None:
        self._decision = decision
        self._config = config

    @property
    def config(self) -> ConsensusConfig:
        return self._config

    def decide(self, estimates: Any) -> ConsensusDecision:
        return self._decision


class StubRiskEngine:
    """Stand-in for RiskEngine that returns a pre-configured TradePlan."""

    def __init__(self, plan: TradePlan) -> None:
        self._plan = plan
        self.state = BankrollState(
            cash_usd=500.0,
            open_positions_usd=0.0,
            peak_bankroll_usd=500.0,
            current_bankroll_usd=500.0,
            drawdown_pct=0.0,
            last_loss_at=None,
        )
        self.update_state_calls: list[RiskTradeReceipt] = []

    def evaluate(self, decision: Any, market: RiskMarket, bankroll: Any) -> TradePlan:
        return self._plan

    def update_state(self, receipt: RiskTradeReceipt) -> None:
        self.update_state_calls.append(receipt)


class StubDelphiClient:
    """Stand-in for DelphiClient that tracks place_order calls."""

    def __init__(self, market: AdapterMarket) -> None:
        self._market = market
        self.place_order_calls: list[dict[str, Any]] = []
        self._next_receipt = TradeReceipt(
            market_id=market.market_id,
            side=OrderSide.YES,
            size_usd=10.0,
            fill_price=0.55,
            att_order_id="att-order-123",
            status=OrderStatus.PENDING,
            signed_by="test-key-fingerprint",
            timestamp=datetime.now(UTC),
        )

    async def get_market(self, market_id: str) -> AdapterMarket:
        return self._market

    async def place_order(
        self,
        market_id: str,
        side: OrderSide,
        size_usd: float,
        limit_price: float | None = None,
        correlation_id: str | None = None,
    ) -> TradeReceipt:
        self.place_order_calls.append(
            {
                "market_id": market_id,
                "side": side,
                "size_usd": size_usd,
                "limit_price": limit_price,
                "correlation_id": correlation_id,
            }
        )
        return self._next_receipt.model_copy(deep=True)

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_market() -> AdapterMarket:
    return AdapterMarket(
        market_id="dphi_test_001",
        question="Will the test pass by Friday?",
        category=MarketCategory.SUBJECTIVE,
        status=MarketStatus.OPEN,
        yes_price=0.55,
        no_price=0.45,
        volume_usd=1000.0,
        liquidity_usd=500.0,
        created_at=datetime.now(UTC),
        closes_at=None,
        settlement_at=None,
        arbiter_model="gpt-4o-mini",
    )


def _make_estimate(analyst_id: str, prob: float = 0.7) -> Estimate:
    return Estimate(
        market_id="dphi_test_001",
        probability=prob,
        confidence=0.8,
        rationale=f"stub estimate from {analyst_id}",
        evidence=[],
        analyst_id=analyst_id,
    )


def _make_decision(gate: str = "trade", prob: float = 0.7) -> ConsensusDecision:
    return ConsensusDecision(
        market_id="dphi_test_001",
        consensus_prob=prob,
        agreement_score=0.85,
        gate=gate,  # type: ignore[arg-type]
        contributor_ids=["politics", "crypto"],
        method="logit-mean",
        weights_used={"politics": 0.5, "crypto": 0.5},
        timestamp=datetime.now(UTC).isoformat(),
    )


def _make_plan(decision: str = "APPROVE") -> TradePlan:
    return TradePlan(
        market_id="dphi_test_001",
        side="YES",
        size_usd=25.0,
        limit_price=0.55,
        rationale="stub plan",
        risk_flags=[] if decision == "APPROVE" else ["no_edge"],
        decision=decision,  # type: ignore[arg-type]
        timestamp=datetime.now(UTC).isoformat(),
    )


def _make_consensus_config(min_analysts: int = 2) -> ConsensusConfig:
    return ConsensusConfig(
        method="logit-mean",
        agreement_threshold=0.65,
        min_analysts=min_analysts,
    )


def _make_executor(
    *,
    mode: str = "paper",
    market: AdapterMarket | None = None,
    mesh: list[StubAnalyst] | None = None,
    decision: ConsensusDecision | None = None,
    plan: TradePlan | None = None,
    consensus_config: ConsensusConfig | None = None,
    audit_log_path: Path | None = None,
) -> tuple[PythiaExecutor, StubDelphiClient, StubRiskEngine]:
    """Build a PythiaExecutor wired to stubs. Returns (executor, client, risk)."""
    market = market or _make_market()
    client = StubDelphiClient(market)
    mesh = mesh or [
        StubAnalyst("politics", _make_estimate("politics")),
        StubAnalyst("crypto", _make_estimate("crypto")),
    ]
    decision = decision or _make_decision()
    plan = plan or _make_plan()
    consensus_config = consensus_config or _make_consensus_config()

    consensus_engine = StubConsensusEngine(decision, consensus_config)
    risk_engine = StubRiskEngine(plan)

    executor = PythiaExecutor(
        delphi_client=client,  # type: ignore[arg-type]
        mesh=mesh,  # type: ignore[arg-type]
        consensus_engine=consensus_engine,  # type: ignore[arg-type]
        risk_engine=risk_engine,  # type: ignore[arg-type]
        config=ExecutorConfig(
            mode=mode,  # type: ignore[arg-type]
            signing_key_env="DELPHI_SIGNING_KEY",
            idempotency_enabled=True,
            retry_max=3,
            retry_backoff_sec=5,
        ),
        audit_log_path=audit_log_path or Path("/tmp/pythia-test-audit.jsonl"),
    )
    return executor, client, risk_engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_mode_does_not_submit(tmp_path: Path) -> None:
    """Paper mode synthesises a PAPER receipt; place_order is never called."""
    audit_log = tmp_path / "audit.jsonl"
    executor, client, _ = _make_executor(mode="paper", audit_log_path=audit_log)

    result = await executor.run_for_market("dphi_test_001")

    assert client.place_order_calls == []
    assert result.skipped_reason is None
    assert result.receipt is not None
    # Paper receipts carry status="PAPER" (set via model_construct).
    assert result.receipt.status == "PAPER"
    assert result.receipt.signed_by == "paper-mode"
    assert result.receipt.att_order_id.startswith("paper-")


@pytest.mark.asyncio
async def test_live_mode_submits_with_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mode signs the order and calls delphi_client.place_order."""
    # Set a non-32-byte signing key so HMAC-SHA256 path is used (no
    # dependency on the cryptography package being installed).
    monkeypatch.setenv("DELPHI_SIGNING_KEY", "test-signing-key-not-32-bytes")

    audit_log = tmp_path / "audit.jsonl"
    executor, client, _ = _make_executor(mode="live", audit_log_path=audit_log)

    result = await executor.run_for_market("dphi_test_001")

    assert len(client.place_order_calls) == 1
    call = client.place_order_calls[0]
    assert call["market_id"] == "dphi_test_001"
    assert call["side"] == OrderSide.YES
    assert call["size_usd"] == 25.0
    assert call["limit_price"] == 0.55
    assert call["correlation_id"] is not None

    assert result.skipped_reason is None
    assert result.receipt is not None
    # The live receipt came from the stub DelphiClient.
    assert result.receipt.att_order_id == "att-order-123"
    # _submit_live_order attached the signature as an extra field.
    sig = getattr(result.receipt, "receipt_signature", None)
    assert sig is not None
    assert isinstance(sig, str)
    assert len(sig) > 0


@pytest.mark.asyncio
async def test_skips_when_insufficient_analysts(tmp_path: Path) -> None:
    """When the mesh returns < min_analysts estimates, we skip early."""
    audit_log = tmp_path / "audit.jsonl"
    # Only 1 analyst, but min_analysts=2.
    mesh = [StubAnalyst("politics", _make_estimate("politics"))]
    executor, client, _ = _make_executor(
        mode="paper",
        mesh=mesh,
        consensus_config=_make_consensus_config(min_analysts=2),
        audit_log_path=audit_log,
    )

    result = await executor.run_for_market("dphi_test_001")

    assert result.skipped_reason == "insufficient_analysts"
    assert result.decision is None
    assert result.plan is None
    assert result.receipt is None
    assert len(result.estimates) == 1
    assert client.place_order_calls == []


@pytest.mark.asyncio
async def test_skips_when_gate_is_skip(tmp_path: Path) -> None:
    """When consensus returns gate='skip', we skip before risk evaluation."""
    audit_log = tmp_path / "audit.jsonl"
    skip_decision = _make_decision(gate="skip", prob=0.5)
    executor, client, _ = _make_executor(
        mode="paper",
        decision=skip_decision,
        audit_log_path=audit_log,
    )

    result = await executor.run_for_market("dphi_test_001")

    assert result.skipped_reason == "gate_skip"
    assert result.decision is not None
    assert result.decision.gate == "skip"
    assert result.plan is None
    assert result.receipt is None
    assert client.place_order_calls == []


@pytest.mark.asyncio
async def test_skips_when_risk_rejects(tmp_path: Path) -> None:
    """When risk returns REJECT, we skip submission with risk_rejected:<flags>."""
    audit_log = tmp_path / "audit.jsonl"
    reject_plan = _make_plan(decision="REJECT")
    # Override risk_flags for a deterministic assertion.
    reject_plan = reject_plan.model_copy(
        update={"risk_flags": ["no_edge", "drawdown_breaker"]}
    )
    executor, client, _ = _make_executor(
        mode="paper",
        plan=reject_plan,
        audit_log_path=audit_log,
    )

    result = await executor.run_for_market("dphi_test_001")

    assert result.skipped_reason == "risk_rejected:no_edge,drawdown_breaker"
    assert result.plan is not None
    assert result.plan.decision == "REJECT"
    assert result.receipt is None
    assert client.place_order_calls == []


@pytest.mark.asyncio
async def test_audit_log_written(tmp_path: Path) -> None:
    """After a paper-trade run, the audit log has exactly one JSONL line
    with the full decision chain."""
    audit_log = tmp_path / "audit.jsonl"
    assert not audit_log.exists()

    executor, _, risk_engine = _make_executor(
        mode="paper", audit_log_path=audit_log
    )

    await executor.run_for_market("dphi_test_001")

    assert audit_log.exists()
    lines = audit_log.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    assert len(lines) == 1

    payload = json.loads(lines[0])
    # Top-level fields.
    assert payload["market_id"] == "dphi_test_001"
    assert payload["skipped_reason"] is None
    assert "timestamp" in payload
    # Signature fields.
    assert "signature" in payload
    assert isinstance(payload["signature"], str)
    assert len(payload["signature"]) > 0
    assert payload["signature_algo"] in {"ed25519", "hmac-sha256"}
    assert "signed_by" in payload
    # Decision chain.
    assert payload["estimates"] is not None
    assert len(payload["estimates"]) == 2
    assert payload["decision"] is not None
    assert payload["decision"]["gate"] == "trade"
    assert payload["plan"] is not None
    assert payload["plan"]["decision"] == "APPROVE"
    assert payload["receipt"] is not None
    assert payload["receipt"]["status"] == "PAPER"

    # Risk engine state was updated.
    assert len(risk_engine.update_state_calls) == 1
    risk_receipt = risk_engine.update_state_calls[0]
    assert risk_receipt.market_id == "dphi_test_001"
    assert risk_receipt.side == "YES"
