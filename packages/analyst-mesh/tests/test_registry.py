"""Tests for ``pythia_analyst_mesh.registry.AnalystRegistry``."""
from __future__ import annotations

import pytest

from pythia_analyst_mesh import (
    AnalystRegistry,
    BaseAnalyst,
    CryptoAnalyst,
    LLMConfig,
    NicheAnalyst,
    PoliticsAnalyst,
    SportsAnalyst,
)
from pythia_analyst_mesh.types import MarketContext

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
def sample_market() -> MarketContext:
    return MarketContext(
        market_id="mkt-test-1",
        question="Will X happen?",
        category="politics",
        metadata={"news_context": "Stub news."},
        current_yes_price=0.4,
        current_no_price=0.6,
        volume_usd=10_000.0,
        closes_at="2026-12-31T23:59:59Z",
    )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_builtin_analysts_auto_registered():
    reg = AnalystRegistry()
    known = reg.list_known()
    assert "politics" in known
    assert "crypto" in known
    assert "sports" in known
    assert "niche" in known
    assert len(known) >= 4


def test_get_returns_class_not_instance(llm_config):
    reg = AnalystRegistry()
    cls = reg.get("politics")
    assert cls is PoliticsAnalyst
    # Calling it should produce an instance.
    inst = cls(llm_config)
    assert isinstance(inst, BaseAnalyst)
    assert inst.analyst_id == "politics"


def test_get_unknown_raises():
    reg = AnalystRegistry()
    with pytest.raises(KeyError, match="unknown analyst"):
        reg.get("nonexistent")


def test_register_new_analyst():
    class MacroAnalyst(BaseAnalyst):
        analyst_id = "macro"
        specialty = "macroeconomics"
        SYSTEM_PROMPT = "stub"

        def _build_prompt(self, market):
            return [{"role": "system", "content": self.SYSTEM_PROMPT}]

    reg = AnalystRegistry()
    assert "macro" not in reg.list_known()
    reg.register("macro", MacroAnalyst)
    assert "macro" in reg.list_known()
    assert reg.get("macro") is MacroAnalyst


def test_register_rejects_non_baseanalyst():
    """Passing a non-BaseAnalyst class should raise TypeError."""
    reg = AnalystRegistry()
    with pytest.raises(TypeError):
        reg.register("bogus", object)  # type: ignore[arg-type]


def test_register_rejects_class_without_attributes():
    """A BaseAnalyst subclass that doesn't set analyst_id should fail."""
    reg = AnalystRegistry()

    class HalfBaked(BaseAnalyst):
        # forgot analyst_id and specialty
        SYSTEM_PROMPT = "stub"

        def _build_prompt(self, market):
            return []

    with pytest.raises(ValueError):
        reg.register("halfbaked", HalfBaked)


def test_register_rejects_empty_name():
    reg = AnalystRegistry()
    with pytest.raises(ValueError):
        reg.register("", PoliticsAnalyst)


def test_build_mesh_returns_instances(llm_config):
    reg = AnalystRegistry()
    mesh = reg.build_mesh(["politics", "crypto"], llm_config)
    assert len(mesh) == 2
    assert all(isinstance(a, BaseAnalyst) for a in mesh)
    assert mesh[0].analyst_id == "politics"
    assert mesh[1].analyst_id == "crypto"
    # Shared config
    assert all(a._llm_config is llm_config for a in mesh)


def test_build_mesh_deduplicates(llm_config):
    reg = AnalystRegistry()
    mesh = reg.build_mesh(["politics", "politics", "crypto"], llm_config)
    assert len(mesh) == 2


def test_build_mesh_unknown_raises(llm_config):
    reg = AnalystRegistry()
    with pytest.raises(KeyError, match="unknown analysts"):
        reg.build_mesh(["politics", "nonexistent"], llm_config)


def test_build_mesh_all_four(llm_config):
    reg = AnalystRegistry()
    mesh = reg.build_mesh(["politics", "crypto", "sports", "niche"], llm_config)
    assert len(mesh) == 4
    ids = {a.analyst_id for a in mesh}
    assert ids == {"politics", "crypto", "sports", "niche"}


def test_contains_and_len():
    reg = AnalystRegistry()
    assert "politics" in reg
    assert "nonexistent" not in reg
    assert len(reg) >= 4


def test_getitem_alias_for_get():
    reg = AnalystRegistry()
    assert reg["politics"] is PoliticsAnalyst
    assert reg["crypto"] is CryptoAnalyst
    assert reg["sports"] is SportsAnalyst
    assert reg["niche"] is NicheAnalyst
