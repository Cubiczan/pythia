"""Tests for the 4 built-in analyst specialists.

We mock LLM calls so no network is required. Tests cover:
- Each analyst's _build_prompt produces the expected chat structure.
- Each analyst's system prompt mentions calibration.
- The full estimate() pipeline (with mocked _call_llm) returns a sane Estimate.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from pythia_analyst_mesh import (
    CryptoAnalyst,
    LLMConfig,
    MarketContext,
    NicheAnalyst,
    PoliticsAnalyst,
    SportsAnalyst,
)
from pythia_analyst_mesh.base import BaseAnalyst

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(
        provider="ollama",
        model="llama3",
        api_key=None,
        temperature=0.2,
        max_tokens=64,
    )


@pytest.fixture
def politics_market() -> MarketContext:
    return MarketContext(
        market_id="mkt-pol-1",
        question="Will the incumbent win the 2026 midterm in District 4?",
        category="politics",
        metadata={"news_context": "Recent polls show a 3-point lead."},
        current_yes_price=0.55,
        current_no_price=0.45,
        volume_usd=25_000.0,
        closes_at="2026-11-04T20:00:00Z",
    )


@pytest.fixture
def crypto_market() -> MarketContext:
    return MarketContext(
        market_id="mkt-crypto-1",
        question="Will ETH/USD close above $5,000 on or before Dec 31, 2026?",
        category="crypto",
        metadata={"news_context": "ETF inflows accelerating; spot up 18% MoM."},
        current_yes_price=0.42,
        current_no_price=0.58,
        volume_usd=124_000.0,
        closes_at="2026-12-31T23:59:59Z",
    )


@pytest.fixture
def sports_market() -> MarketContext:
    return MarketContext(
        market_id="mkt-spt-1",
        question="Will the Lakers make the 2026 NBA playoffs?",
        category="sports",
        metadata={"news_context": "Star player listed as day-to-day."},
        current_yes_price=0.61,
        current_no_price=0.39,
        volume_usd=8_000.0,
        closes_at="2026-04-15T23:59:59Z",
    )


@pytest.fixture
def niche_market() -> MarketContext:
    return MarketContext(
        market_id="mkt-niche-1",
        question="Will 'Album X' win Best Alternative Grammy at the 2027 ceremony?",
        category="niche",
        metadata={"news_context": "Critics' picks favor Album X; insider buzz mixed."},
        current_yes_price=0.30,
        current_no_price=0.70,
        volume_usd=4_500.0,
        closes_at="2027-02-05T03:00:00Z",
    )


# --------------------------------------------------------------------- #
# Shared structural assertions
# --------------------------------------------------------------------- #


def _assert_prompt_structure(messages, analyst, market):
    """Common assertions for _build_prompt output."""
    assert isinstance(messages, list), "_build_prompt must return a list"
    assert len(messages) >= 2, "must have at least system + user message"
    assert all(isinstance(m, dict) for m in messages), "messages must be dicts"
    assert all("role" in m and "content" in m for m in messages), \
        "each message must have role + content"

    # System message is first, and uses the class's SYSTEM_PROMPT.
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == analyst.SYSTEM_PROMPT

    # User message is second.
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]

    # User message MUST contain the market question.
    assert market.question in user_content, \
        f"user prompt missing market question: {user_content!r}"

    # User message MUST mention the current YES price (implied prob).
    if market.current_yes_price is not None:
        # Format is "%.3f" — check first 3 decimals appear.
        price_str = f"{market.current_yes_price:.3f}"
        assert price_str in user_content, \
            f"user prompt missing current YES price {price_str}: {user_content!r}"

    # User message MUST mention the close date if provided.
    if market.closes_at:
        assert market.closes_at in user_content, \
            f"user prompt missing close date: {user_content!r}"

    # User message MUST contain the JSON output contract.
    assert "JSON" in user_content, "user prompt must specify JSON output contract"
    assert "probability" in user_content
    assert "confidence" in user_content
    assert "rationale" in user_content
    assert "evidence" in user_content


def _assert_system_prompt_calibration(analyst):
    """System prompt must mention calibration (per task spec)."""
    s = analyst.SYSTEM_PROMPT.lower()
    assert "calibrat" in s, f"system prompt must mention calibration: {analyst.SYSTEM_PROMPT!r}"
    assert "confidence" in s, f"system prompt must mention confidence: {analyst.SYSTEM_PROMPT!r}"
    assert "0.6" in s or "< 0.6" in s, \
        f"system prompt must reference the 0.6 confidence threshold: {analyst.SYSTEM_PROMPT!r}"


def _assert_system_prompt_length(analyst):
    """System prompt should be 2-4 sentences (per task spec)."""
    # Count sentences by terminal punctuation. Allow some slack.
    n = max(1, len([s for s in analyst.SYSTEM_PROMPT.split(".") if s.strip()]))
    assert 2 <= n <= 6, \
        f"system prompt should be 2-4 sentences, got ~{n}: {analyst.SYSTEM_PROMPT!r}"


# --------------------------------------------------------------------- #
# Per-analyst prompt tests
# --------------------------------------------------------------------- #


def test_politics_prompt_structure(llm_config, politics_market):
    a = PoliticsAnalyst(llm_config)
    msgs = a._build_prompt(politics_market)
    _assert_prompt_structure(msgs, a, politics_market)
    _assert_system_prompt_calibration(a)
    _assert_system_prompt_length(a)
    # Politics-specific guidance should appear.
    assert "polling" in msgs[1]["content"].lower()


def test_crypto_prompt_structure(llm_config, crypto_market):
    a = CryptoAnalyst(llm_config)
    msgs = a._build_prompt(crypto_market)
    _assert_prompt_structure(msgs, a, crypto_market)
    _assert_system_prompt_calibration(a)
    _assert_system_prompt_length(a)
    # Crypto-specific guidance should appear.
    content = msgs[1]["content"].lower()
    assert "on-chain" in content or "defillama" in content or "etherscan" in content


def test_sports_prompt_structure(llm_config, sports_market):
    a = SportsAnalyst(llm_config)
    msgs = a._build_prompt(sports_market)
    _assert_prompt_structure(msgs, a, sports_market)
    _assert_system_prompt_calibration(a)
    _assert_system_prompt_length(a)
    # Sports-specific guidance should appear.
    content = msgs[1]["content"].lower()
    assert "elo" in content or "injury" in content or "rating" in content


def test_niche_prompt_structure(llm_config, niche_market):
    a = NicheAnalyst(llm_config)
    msgs = a._build_prompt(niche_market)
    _assert_prompt_structure(msgs, a, niche_market)
    _assert_system_prompt_calibration(a)
    _assert_system_prompt_length(a)
    # Niche-specific guidance should appear.
    content = msgs[1]["content"].lower()
    assert "anchor" in content or "subjective" in content or "award" in content


# --------------------------------------------------------------------- #
# Full estimate() pipeline with mocked LLM
# --------------------------------------------------------------------- #


def _mock_llm_response(*, prob, conf, rationale="Mocked.", evidence=None):
    return json.dumps({
        "probability": prob,
        "confidence": conf,
        "rationale": rationale,
        "evidence": evidence or [],
    })


async def test_politics_estimate_with_mocked_llm(llm_config, politics_market):
    a = PoliticsAnalyst(llm_config)
    fake = _mock_llm_response(prob=0.62, conf=0.7, evidence=["https://polls.example.com"])
    with patch.object(
        BaseAnalyst, "_call_llm", new=AsyncMock(return_value=fake)
    ):
        est = await a.estimate(politics_market)
    assert est.analyst_id == "politics"
    assert est.market_id == "mkt-pol-1"
    assert est.probability == pytest.approx(0.62)
    assert est.confidence == pytest.approx(0.7)
    assert "https://polls.example.com" in est.evidence


async def test_crypto_estimate_with_mocked_llm(llm_config, crypto_market):
    a = CryptoAnalyst(llm_config)
    fake = _mock_llm_response(prob=0.35, conf=0.55, evidence=["https://etherscan.io"])
    with patch.object(
        BaseAnalyst, "_call_llm", new=AsyncMock(return_value=fake)
    ):
        est = await a.estimate(crypto_market)
    assert est.analyst_id == "crypto"
    assert est.probability == pytest.approx(0.35)
    assert est.confidence == pytest.approx(0.55)


async def test_sports_estimate_with_mocked_llm(llm_config, sports_market):
    a = SportsAnalyst(llm_config)
    fake = _mock_llm_response(prob=0.58, conf=0.65)
    with patch.object(
        BaseAnalyst, "_call_llm", new=AsyncMock(return_value=fake)
    ):
        est = await a.estimate(sports_market)
    assert est.analyst_id == "sports"
    assert est.probability == pytest.approx(0.58)


async def test_niche_estimate_with_mocked_llm(llm_config, niche_market):
    a = NicheAnalyst(llm_config)
    fake = _mock_llm_response(prob=0.28, conf=0.4)  # low conf per niche guidance
    with patch.object(
        BaseAnalyst, "_call_llm", new=AsyncMock(return_value=fake)
    ):
        est = await a.estimate(niche_market)
    assert est.analyst_id == "niche"
    assert est.probability == pytest.approx(0.28)
    assert est.confidence == pytest.approx(0.4)


# --------------------------------------------------------------------- #
# run_mesh integration (mocked LLM)
# --------------------------------------------------------------------- #


async def test_run_mesh_with_mocked_llm(llm_config, politics_market):
    from pythia_analyst_mesh import AnalystRegistry, run_mesh

    reg = AnalystRegistry()
    mesh = reg.build_mesh(["politics", "crypto", "niche"], llm_config)

    # Each analyst gets a slightly different mocked response.
    responses = iter([
        _mock_llm_response(prob=0.6, conf=0.7),
        _mock_llm_response(prob=0.55, conf=0.6),
        _mock_llm_response(prob=0.45, conf=0.4),
    ])

    async def fake_call(self, messages, config):
        return next(responses)

    with patch.object(BaseAnalyst, "_call_llm", new=fake_call):
        estimates = await run_mesh(politics_market, mesh, timeout_sec=5.0)

    assert len(estimates) == 3
    assert {e.analyst_id for e in estimates} == {"politics", "crypto", "niche"}
    probs = [e.probability for e in estimates]
    assert probs == pytest.approx([0.6, 0.55, 0.45])


async def test_run_mesh_drops_failing_analyst(llm_config, politics_market):
    """One analyst raising should not crash the mesh."""
    from pythia_analyst_mesh import AnalystRegistry, run_mesh

    reg = AnalystRegistry()
    mesh = reg.build_mesh(["politics", "crypto", "niche"], llm_config)

    call_count = {"n": 0}

    async def fake_call(self, messages, config):
        call_count["n"] += 1
        if self.analyst_id == "crypto":
            raise RuntimeError("simulated LLM failure")
        return _mock_llm_response(prob=0.5, conf=0.5)

    with patch.object(BaseAnalyst, "_call_llm", new=fake_call):
        estimates = await run_mesh(politics_market, mesh, timeout_sec=5.0)

    assert len(estimates) == 2
    assert {e.analyst_id for e in estimates} == {"politics", "niche"}
