"""Tests for ``pythia_analyst_mesh.base.BaseAnalyst`` — especially the
robust ``_parse_llm_response`` helper.

We instantiate a concrete subclass for testing (BaseAnalyst is abstract).
No real LLM calls are made — we exercise only the parsing logic.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from pythia_analyst_mesh import BaseAnalyst, LLMConfig, MarketContext
from pythia_analyst_mesh.base import LLMCallError

# --------------------------------------------------------------------- #
# Concrete test-only subclass
# --------------------------------------------------------------------- #

class StubAnalyst(BaseAnalyst):
    analyst_id = "stub"
    specialty = "test"
    SYSTEM_PROMPT = "stub prompt"

    def _build_prompt(self, market: MarketContext) -> list[dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": market.question},
        ]

@pytest.fixture
def analyst() -> StubAnalyst:
    cfg = LLMConfig(provider="ollama", model="llama3")
    return StubAnalyst(cfg)

@pytest.fixture
def market() -> MarketContext:
    return MarketContext(
        market_id="mkt-1",
        question="Will X happen?",
        category="test",
        metadata={"news_context": "some news"},
        current_yes_price=0.4,
        current_no_price=0.6,
        volume_usd=1_000.0,
        closes_at="2026-12-31T23:59:59Z",
    )

# --------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------- #

def test_constructor_requires_analyst_id():
    class BadAnalyst(BaseAnalyst):
        specialty = "x"
        SYSTEM_PROMPT = "x"

        def _build_prompt(self, market):
            return []

    with pytest.raises(ValueError, match="analyst_id"):
        BadAnalyst(LLMConfig(provider="ollama", model="x"))

def test_constructor_requires_specialty():
    class BadAnalyst(BaseAnalyst):
        analyst_id = "x"
        SYSTEM_PROMPT = "x"

        def _build_prompt(self, market):
            return []

    with pytest.raises(ValueError, match="specialty"):
        BadAnalyst(LLMConfig(provider="ollama", model="x"))

# --------------------------------------------------------------------- #
# _parse_llm_response — valid JSON
# --------------------------------------------------------------------- #

def test_parse_valid_json(analyst, market):
    raw = json.dumps({
        "probability": 0.72,
        "confidence": 0.8,
        "rationale": "Because reasons.",
        "evidence": ["https://example.com/a", "https://example.com/b"],
    })
    est = analyst._parse_llm_response(raw, market)
    assert est.market_id == "mkt-1"
    assert est.analyst_id == "stub"
    assert est.probability == pytest.approx(0.72)
    assert est.confidence == pytest.approx(0.8)
    assert est.rationale == "Because reasons."
    assert est.evidence == ["https://example.com/a", "https://example.com/b"]
    # ISO timestamp
    datetime.fromisoformat(est.timestamp)

def test_parse_valid_json_with_code_fences(analyst, market):
    raw = (
        "Here is my estimate:\n"
        "```json\n"
        + json.dumps({
            "probability": 0.55,
            "confidence": 0.6,
            "rationale": "Fenced.",
            "evidence": [],
        })
        + "\n```\n"
        "Hope that helps."
    )
    est = analyst._parse_llm_response(raw, market)
    assert est.probability == pytest.approx(0.55)
    assert est.confidence == pytest.approx(0.6)
    assert est.rationale == "Fenced."

def test_parse_valid_json_with_bare_fences(analyst, market):
    raw = (
        "```\n"
        + json.dumps({
            "probability": 0.33,
            "confidence": 0.5,
            "rationale": "Bare fence.",
            "evidence": [],
        })
        + "\n```"
    )
    est = analyst._parse_llm_response(raw, market)
    assert est.probability == pytest.approx(0.33)

# --------------------------------------------------------------------- #
# _parse_llm_response — partial / malformed JSON
# --------------------------------------------------------------------- #

def test_parse_json_with_prose_around_object(analyst, market):
    raw = (
        "Let me think step by step...\n"
        "Final answer: "
        + json.dumps({
            "probability": 0.6,
            "confidence": 0.55,
            "rationale": "Surrounded by prose.",
            "evidence": ["https://example.com"],
        })
        + "\nThat's my reasoning."
    )
    est = analyst._parse_llm_response(raw, market)
    assert est.probability == pytest.approx(0.6)
    assert est.rationale == "Surrounded by prose."

def test_parse_partial_json_missing_confidence(analyst, market):
    raw = json.dumps({
        "probability": 0.42,
        "rationale": "Partial.",
        "evidence": [],
    })
    est = analyst._parse_llm_response(raw, market)
    assert est.probability == pytest.approx(0.42)
    # Missing confidence → default 0.3 (low)
    assert est.confidence == pytest.approx(0.3)
    assert est.rationale == "Partial."

def test_parse_partial_json_missing_rationale(analyst, market):
    raw = json.dumps({"probability": 0.5, "confidence": 0.5})
    est = analyst._parse_llm_response(raw, market)
    assert est.probability == pytest.approx(0.5)
    # Missing rationale falls back to truncated raw text.
    assert isinstance(est.rationale, str) and len(est.rationale) > 0

def test_parse_partial_json_missing_probability(analyst, market):
    raw = json.dumps({"confidence": 0.4, "rationale": "No prob."})
    est = analyst._parse_llm_response(raw, market)
    # Missing probability → default 0.5
    assert est.probability == pytest.approx(0.5)
    assert est.confidence == pytest.approx(0.4)

def test_parse_partial_json_missing_evidence(analyst, market):
    raw = json.dumps({
        "probability": 0.8,
        "confidence": 0.7,
        "rationale": "No evidence.",
    })
    est = analyst._parse_llm_response(raw, market)
    assert est.evidence == []

def test_parse_probability_out_of_range_clamped(analyst, market):
    raw = json.dumps({
        "probability": 5.0,  # > 1, should clamp to 1.0
        "confidence": -0.2,  # < 0, should clamp to 0.0
        "rationale": "Clamped.",
    })
    est = analyst._parse_llm_response(raw, market)
    assert est.probability == pytest.approx(1.0)
    assert est.confidence == pytest.approx(0.0)

def test_parse_evidence_as_csv_string(analyst, market):
    raw = json.dumps({
        "probability": 0.6,
        "confidence": 0.5,
        "rationale": "Csv.",
        "evidence": "https://a.com, https://b.com",
    })
    est = analyst._parse_llm_response(raw, market)
    assert est.evidence == ["https://a.com", "https://b.com"]

# --------------------------------------------------------------------- #
# _parse_llm_response — pure-prose fallbacks
# --------------------------------------------------------------------- #

def test_parse_pure_prose_with_leading_probability(analyst, market):
    raw = (
        "After careful analysis I estimate 0.65 probability that YES occurs. "
        "Recent polls suggest a small advantage."
    )
    est = analyst._parse_llm_response(raw, market)
    assert est.probability == pytest.approx(0.65)
    # Low confidence from fallback path.
    assert est.confidence == pytest.approx(0.3)

def test_parse_pure_prose_with_percentage(analyst, market):
    raw = "I think there's a 42% chance of YES."
    est = analyst._parse_llm_response(raw, market)
    assert est.probability == pytest.approx(0.42)
    assert est.confidence == pytest.approx(0.3)

def test_parse_pure_prose_no_number(analyst, market):
    raw = "I cannot determine this with any confidence."
    est = analyst._parse_llm_response(raw, market)
    # Last-resort fallback: 0.5 prior, 0.0 confidence.
    assert est.probability == pytest.approx(0.5)
    assert est.confidence == pytest.approx(0.0)
    assert "uninformative prior" in est.rationale.lower() or "could not be parsed" in est.rationale.lower()

def test_parse_empty_string(analyst, market):
    est = analyst._parse_llm_response("", market)
    assert est.probability == pytest.approx(0.5)
    assert est.confidence == pytest.approx(0.0)

def test_parse_always_returns_estimate(analyst, market):
    """The contract: _parse_llm_response must NEVER raise."""
    weird_inputs = [
        "",
        "   ",
        "{}",
        "[]",
        "null",
        "12345",
        "```",
        "no numbers here at all",
        "{broken json",
        '{"probability": "not a number"}',
        "\n\n\n",
        "Yes",  # bare 0/1 ambiguous
    ]
    for raw in weird_inputs:
        est = analyst._parse_llm_response(raw, market)
        assert est is not None
        assert 0.0 <= est.probability <= 1.0
        assert 0.0 <= est.confidence <= 1.0
        assert est.market_id == "mkt-1"
        assert est.analyst_id == "stub"

# --------------------------------------------------------------------- #
# _call_llm provider dispatch (no real network — verify error path)
# --------------------------------------------------------------------- #

async def test_call_llm_unknown_provider_raises():
    """If we somehow bypass the Literal type, dispatch raises LLMCallError."""
    cfg = LLMConfig(provider="ollama", model="x")
    analyst = StubAnalyst(cfg)
    # Bypass Pydantic Literal validation by using model_construct.
    cfg2 = LLMConfig.model_construct(provider="unknown", model="x")
    with pytest.raises(LLMCallError, match="Unknown LLM provider"):
        await analyst._call_llm([{"role": "user", "content": "hi"}], cfg2)
