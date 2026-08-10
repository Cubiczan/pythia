"""Tests for `pythia_consensus.engine.ConsensusEngine`."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from pythia_consensus import ConsensusConfig, ConsensusEngine
from pythia_consensus.audit import AuditSigner
from pythia_consensus.types import Estimate


def _est(probabilities: list[float], *, market_id: str = "mkt-1") -> list[Estimate]:
    ts = datetime.now(timezone.utc).isoformat()
    return [
        Estimate(
            market_id=market_id,
            probability=p,
            confidence=0.7,
            rationale="test",
            evidence=[],
            analyst_id=f"a{i}",
            timestamp=ts,
        )
        for i, p in enumerate(probabilities)
    ]


class TestConsensusEngineDecide:
    def test_decide_matches_fuse_with_equal_weights(self) -> None:
        """Engine with no weights should match plain `fuse` with equal weights."""
        from pythia_consensus import fuse

        ests = _est([0.4, 0.6, 0.55])
        cfg = ConsensusConfig(method="logit-mean")
        engine = ConsensusEngine(cfg)

        d_engine = engine.decide(ests)
        d_fuse = fuse(ests, cfg)
        assert d_engine.consensus_prob == pytest.approx(d_fuse.consensus_prob, abs=1e-12)
        assert d_engine.gate == d_fuse.gate
        assert d_engine.agreement_score == pytest.approx(d_fuse.agreement_score, abs=1e-12)

    def test_decide_uses_engine_weights(self) -> None:
        """After update_weights, the engine should fuse using the new weights."""
        ests = _est([0.2, 0.8])
        cfg = ConsensusConfig(method="logit-mean")
        engine = ConsensusEngine(cfg)

        d_before = engine.decide(ests)
        assert d_before.consensus_prob == pytest.approx(0.5, abs=1e-9)

        # Skew weight toward a1 (prob=0.8).
        engine.update_weights({"a0": 0.4, "a1": 0.05})  # a0 has lower Brier (better)
        # But wait — lower Brier means BETTER. So a0 (Brier=0.4) is worse than
        # a1 (Brier=0.05). a1 should get the higher weight.
        d_after = engine.decide(ests)
        assert d_after.consensus_prob > 0.55  # shifted toward 0.8

    def test_config_property_is_a_copy(self) -> None:
        cfg = ConsensusConfig(method="median", agreement_threshold=0.7)
        engine = ConsensusEngine(cfg)
        # Mutating the original cfg should not affect the engine.
        cfg.method = "logit-mean"
        assert engine.config.method == "median"


class TestConsensusEngineUpdateWeights:
    def test_update_weights_lower_brier_gets_higher_weight(self) -> None:
        cfg = ConsensusConfig(method="logit-mean")
        engine = ConsensusEngine(cfg)

        engine.update_weights({"a0": 0.05, "a1": 0.25})  # a0 much better calibrated
        weights = engine.config.weights
        assert weights is not None
        assert weights["a0"] > weights["a1"]
        # Softmax(-4*0.05) / [softmax(-4*0.05) + softmax(-4*0.25)]
        # = e^-0.2 / (e^-0.2 + e^-1.0)
        expected = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.0))
        assert weights["a0"] == pytest.approx(expected, abs=1e-9)

    def test_update_weights_normalises_to_one(self) -> None:
        cfg = ConsensusConfig(method="logit-mean")
        engine = ConsensusEngine(cfg)
        engine.update_weights({"a0": 0.1, "a1": 0.2, "a2": 0.3})
        weights = engine.config.weights
        assert weights is not None
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_update_weights_preserves_previous_for_missing(self) -> None:
        """An analyst absent from a later update_weights call keeps their weight."""
        cfg = ConsensusConfig(method="logit-mean")
        engine = ConsensusEngine(cfg)
        engine.update_weights({"a0": 0.1, "a1": 0.2})
        first_a0 = engine.config.weights["a0"]

        # Update only a1 — a0 should remain unchanged.
        engine.update_weights({"a1": 0.05})
        assert engine.config.weights["a0"] == pytest.approx(first_a0, abs=1e-9)
        # a1's weight changed (got better, so should now be higher relative to before).
        # We don't assert exact value because the absolute weight depends on the
        # normalisation across both analysts.

    def test_update_weights_empty_dict_is_noop(self) -> None:
        cfg = ConsensusConfig(method="logit-mean", weights={"a0": 0.5, "a1": 0.5})
        engine = ConsensusEngine(cfg)
        engine.update_weights({})
        assert engine.config.weights == {"a0": 0.5, "a1": 0.5}

    def test_update_weights_skips_non_finite(self) -> None:
        cfg = ConsensusConfig(method="logit-mean")
        engine = ConsensusEngine(cfg)
        engine.update_weights({"a0": 0.1, "a1": float("nan"), "a2": float("inf")})
        # Only a0 should be in the weights dict.
        weights = engine.config.weights
        assert weights is not None
        assert "a0" in weights
        assert "a1" not in weights
        assert "a2" not in weights

    def test_brier_alpha_affects_weight_ratio(self) -> None:
        """A higher alpha should amplify the weight gap between good and bad analysts."""
        brier = {"good": 0.05, "bad": 0.25}

        engine_low_alpha = ConsensusEngine(ConsensusConfig(method="logit-mean"), brier_alpha=1.0)
        engine_high_alpha = ConsensusEngine(ConsensusConfig(method="logit-mean"), brier_alpha=10.0)

        engine_low_alpha.update_weights(brier)
        engine_high_alpha.update_weights(brier)

        low_w = engine_low_alpha.config.weights  # type: ignore[union-attr]
        high_w = engine_high_alpha.config.weights  # type: ignore[union-attr]

        low_ratio = low_w["good"] / low_w["bad"]
        high_ratio = high_w["good"] / high_w["bad"]
        # Higher alpha → exponentially bigger ratio.
        assert high_ratio > low_ratio


class TestConsensusEngineExplain:
    def test_explain_contains_gate_and_method(self) -> None:
        ests = _est([0.6, 0.62])
        engine = ConsensusEngine(ConsensusConfig(method="logit-mean"))
        d = engine.decide(ests)
        explanation = engine.explain(d)
        assert "gate" in explanation.lower()
        assert "logit-mean" in explanation
        assert "0.6" in explanation  # consensus_prob rounded to 4 dp shows ~0.6100
        assert "agreement_score" in explanation
        assert "GATE=" in explanation

    def test_explain_trade_gate_message(self) -> None:
        ests = _est([0.6, 0.6, 0.6])  # perfect agreement
        engine = ConsensusEngine(ConsensusConfig(method="logit-mean"))
        d = engine.decide(ests)
        assert d.gate == "trade"
        assert "GATE=trade" in engine.explain(d)

    def test_explain_wait_gate_message(self) -> None:
        ests = _est([0.6])  # too few analysts
        engine = ConsensusEngine(ConsensusConfig(method="logit-mean", min_analysts=2))
        d = engine.decide(ests)
        assert d.gate == "wait"
        assert "GATE=wait" in engine.explain(d)

    def test_explain_skip_gate_message(self) -> None:
        ests = _est([0.0, 0.0, 1.0, 1.0])  # maximal disagreement
        engine = ConsensusEngine(ConsensusConfig(method="logit-mean"))
        d = engine.decide(ests)
        assert d.gate == "skip"
        assert "GATE=skip" in engine.explain(d)

    def test_explain_lists_all_contributors(self) -> None:
        ests = _est([0.5, 0.6, 0.7])
        engine = ConsensusEngine(ConsensusConfig(method="median"))
        d = engine.decide(ests)
        explanation = engine.explain(d)
        for aid in ("a0", "a1", "a2"):
            assert aid in explanation


class TestAuditSigner:
    def test_stub_signer_when_upstream_absent(self) -> None:
        """When the upstream is not vendored, the signer falls back to a SHA-256 stub."""
        ests = _est([0.5, 0.6])
        engine = ConsensusEngine(ConsensusConfig(method="logit-mean"))
        d = engine.decide(ests)

        signer = AuditSigner()
        if not signer.upstream_available:
            sig = signer.sign(d)
            assert sig.startswith("stub:sha256:")
            # The same decision should produce the same stub signature
            # (deterministic over canonical JSON).
            sig2 = signer.sign(d)
            assert sig == sig2
        # If upstream IS available (vendored in CI), we just verify the signer
        # returns a non-empty string.
        else:
            sig = signer.sign(d)
            assert isinstance(sig, str) and len(sig) > 0

    def test_signer_handles_different_decisions(self) -> None:
        """Two different decisions should produce different signatures."""
        ests_a = _est([0.4, 0.6])
        ests_b = _est([0.3, 0.8])
        engine = ConsensusEngine(ConsensusConfig(method="logit-mean"))
        d_a = engine.decide(ests_a)
        d_b = engine.decide(ests_b)

        signer = AuditSigner()
        sig_a = signer.sign(d_a)
        sig_b = signer.sign(d_b)
        assert sig_a != sig_b
