"""Pipeline tests for PythiaExecutor.

All four sibling components (DelphiClient, BaseAnalyst, ConsensusEngine,
RiskEngine) are replaced with in-test stubs so no real SDK / LLM calls
are made. Each test exercises one branch of the pipeline.
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
from pythia_delphi_adapter import MarketStatus, TradeReceipt
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
    """Stand-in for DelphiClient that tracks buy_shares calls."""

    def __init__(self, market: AdapterMarket) -> None:
        self._market = market
        self.buy_shares_calls: list[dict[str, Any]] = []
        self.ensure_token_approval_calls: list[dict[str, Any]] = []
        self._next_receipt = TradeReceipt(
            market_address=market.market_address,
            outcome_idx=0,
            side="buy",
            shares="1000000000000000000",
            transaction_hash="0xdeadbeef12345678",
            max_tokens_in="1000000000000000000",
            timestamp=datetime.now(UTC),
        )

    async def get_market(
        self, market_id: str, *, prices_and_implied_probabilities: bool = False
    ) -> AdapterMarket:
        return self._market

    async def ensure_token_approval(
        self, *, market_address: str, minimum_amount: str, approve_amount: str | None = None
    ) -> Any:
        self.ensure_token_approval_calls.append({
            "market_address": market_address,
            "minimum_amount": minimum_amount,
            "approve_amount": approve_amount,
        })
        return {"approval_needed": False, "allowance": minimum_amount}

    async def buy_shares(
        self,
        *,
        market_address: str,
        outcome_idx: int,
        shares_out: str,
        max_tokens_in: str,
    ) -> TradeReceipt:
        self.buy_shares_calls.append({
            "market_address": market_address,
            "outcome_idx": outcome_idx,
            "shares_out": shares_out,
            "max_tokens_in": max_tokens_in,
        })
        return self._next_receipt.model_copy(deep=True)

    async def stop(self) -> None:
        pass

# ---------------------------------------------------------------------------
# Fixtures — build Market using the new adapter SDK schema
# ---------------------------------------------------------------------------

def _make_market() -> AdapterMarket:
    """Build an adapter Market using the new SDK-compatible schema."""
    return AdapterMarket.model_validate({
        "id": "0xtestmarket001",
        "appMarketId": "dphi_test_001",
        "marketUrl": "https://testnet.delphi.fyi/m/dphi_test_001",
        "status": "open",
        "category": "miscellaneous",
        "deployer": "0xfeed0000000000000000000000000000000000fd",
        "createdAt": datetime.now(UTC).isoformat(),
        "settlesAt": None,
        "metadata": {
            "question": "Will the test pass by Friday?",
            "outcomes": ["YES", "NO"],
        },
        "spotPrices": [0.55, 0.45],
        "spotImpliedProbabilities": [0.55, 0.45],
    })

def _make_estimate(analyst_id: str, prob: float = 0.7) -> Estimate:
    return Estimate(
        market_id="0xtestmarket001",
        probability=prob,
        confidence=0.8,
        rationale=f"stub estimate from {analyst_id}",
        evidence=[],
        analyst_id=analyst_id,
    )

def _make_decision(gate: str = "trade", prob: float = 0.7) -> ConsensusDecision:
    return ConsensusDecision(
        market_id="0xtestmarket001",
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
        market_id="0xtestmarket001",
        side="YES",
        outcome_idx=0,
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
    """Paper mode synthesises a PAPER receipt; buy_shares is never called."""
    audit_log = tmp_path / "audit.jsonl"
    executor, client, _ = _make_executor(mode="paper", audit_log_path=audit_log)

    result = await executor.run_for_market("0xtestmarket001")

    assert client.buy_shares_calls == []
    assert result.skipped_reason is None
    assert result.receipt is not None
    # Paper receipt has a fake transaction hash starting with 0xpaper-.
    assert result.receipt.transaction_hash.startswith("0xpaper-")
    assert result.receipt.market_address == "0xtestmarket001"
    assert result.receipt.outcome_idx == 0

@pytest.mark.asyncio
async def test_live_mode_submits_via_buy_shares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mode ensures token approval and calls buy_shares."""
    monkeypatch.setenv("DELPHI_SIGNING_KEY", "test-signing-key-not-32-bytes")

    audit_log = tmp_path / "audit.jsonl"
    executor, client, _ = _make_executor(mode="live", audit_log_path=audit_log)

    result = await executor.run_for_market("0xtestmarket001")

    # Token approval was called first.
    assert len(client.ensure_token_approval_calls) == 1
    assert client.ensure_token_approval_calls[0]["market_address"] == "0xtestmarket001"

    # Then buy_shares was called.
    assert len(client.buy_shares_calls) == 1
    call = client.buy_shares_calls[0]
    assert call["market_address"] == "0xtestmarket001"
    assert call["outcome_idx"] == 0
    assert int(call["shares_out"]) > 0
    assert int(call["max_tokens_in"]) >= int(call["shares_out"])

    assert result.skipped_reason is None
    assert result.receipt is not None
    assert result.receipt.transaction_hash == "0xdeadbeef12345678"

@pytest.mark.asyncio
async def test_skips_when_insufficient_analysts(tmp_path: Path) -> None:
    """When the mesh returns < min_analysts estimates, we skip early."""
    audit_log = tmp_path / "audit.jsonl"
    mesh = [StubAnalyst("politics", _make_estimate("politics"))]
    executor, client, _ = _make_executor(
        mode="paper",
        mesh=mesh,
        consensus_config=_make_consensus_config(min_analysts=2),
        audit_log_path=audit_log,
    )

    result = await executor.run_for_market("0xtestmarket001")

    assert result.skipped_reason == "insufficient_analysts"
    assert result.decision is None
    assert result.plan is None
    assert result.receipt is None
    assert len(result.estimates) == 1
    assert client.buy_shares_calls == []

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

    result = await executor.run_for_market("0xtestmarket001")

    assert result.skipped_reason == "gate_skip"
    assert result.decision is not None
    assert result.decision.gate == "skip"
    assert result.plan is None
    assert result.receipt is None
    assert client.buy_shares_calls == []

@pytest.mark.asyncio
async def test_skips_when_risk_rejects(tmp_path: Path) -> None:
    """When risk returns REJECT, we skip submission with risk_rejected:<flags>."""
    audit_log = tmp_path / "audit.jsonl"
    reject_plan = _make_plan(decision="REJECT")
    reject_plan = reject_plan.model_copy(
        update={"risk_flags": ["no_edge", "drawdown_breaker"]}
    )
    executor, client, _ = _make_executor(
        mode="paper",
        plan=reject_plan,
        audit_log_path=audit_log,
    )

    result = await executor.run_for_market("0xtestmarket001")

    assert result.skipped_reason == "risk_rejected:no_edge,drawdown_breaker"
    assert result.plan is not None
    assert result.plan.decision == "REJECT"
    assert result.receipt is None
    assert client.buy_shares_calls == []

@pytest.mark.asyncio
async def test_audit_log_written(tmp_path: Path) -> None:
    """After a paper-trade run, the audit log has exactly one JSONL line."""
    audit_log = tmp_path / "audit.jsonl"
    assert not audit_log.exists()

    executor, _, risk_engine = _make_executor(
        mode="paper", audit_log_path=audit_log
    )

    await executor.run_for_market("0xtestmarket001")

    assert audit_log.exists()
    lines = audit_log.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["market_id"] == "0xtestmarket001"
    assert payload["skipped_reason"] is None
    assert "timestamp" in payload
    assert "signature" in payload
    assert isinstance(payload["signature"], str)
    assert len(payload["signature"]) > 0
    assert payload["signature_algo"] in {"ed25519", "hmac-sha256"}
    assert "signed_by" in payload
    assert payload["estimates"] is not None
    assert len(payload["estimates"]) == 2
    assert payload["decision"] is not None
    assert payload["decision"]["gate"] == "trade"
    assert payload["plan"] is not None
    assert payload["plan"]["decision"] == "APPROVE"
    assert payload["receipt"] is not None
    # Paper receipt has a fake transaction hash.
    assert payload["receipt"]["transaction_hash"].startswith("0xpaper-")

    # Risk engine state was updated.
    assert len(risk_engine.update_state_calls) == 1
    risk_receipt = risk_engine.update_state_calls[0]
    assert risk_receipt.market_id == "0xtestmarket001"
    assert risk_receipt.side == "YES"
