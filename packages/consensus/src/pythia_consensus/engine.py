"""Stateful consensus engine for the Pythia mesh.

`ConsensusEngine` is a thin stateful wrapper around `fusion.fuse()`. It holds
the config and the per-analyst weight table, and exposes:

- `decide(estimates)` → run one fusion round.
- `update_weights(brier)` → re-derive weights from per-analyst Brier scores
  (typically produced by `pythia-forge` backtests).
- `explain(decision)` → human-readable audit-trail string for the demo / judge.
"""

from __future__ import annotations

import math
from typing import Sequence

from .fusion import fuse
from .types import ConsensusConfig, ConsensusDecision, Estimate

# Temperature for the softmax that converts Brier scores to weights.
# α = 4.0 means a Brier gap of 0.05 maps to a weight ratio of ~e^0.2 ≈ 1.22;
# a gap of 0.25 (huge in Brier terms) maps to a ratio of ~e^1 ≈ 2.72.
_DEFAULT_BRIER_ALPHA = 4.0

class ConsensusEngine:
    """Stateful consensus engine.

    The engine holds the active `ConsensusConfig` (including the per-analyst
    weight table) and exposes `decide()` / `update_weights()` / `explain()`.

    Parameters
    ----------
    config:
        The `ConsensusConfig` to use. If `config.weights` is `None`, equal
        weights are used until `update_weights()` is called.
    brier_alpha:
        Temperature for the Brier → weight softmax. See `update_weights`.
    """

    def __init__(
        self,
        config: ConsensusConfig,
        *,
        brier_alpha: float = _DEFAULT_BRIER_ALPHA,
    ) -> None:
        self._config = config.model_copy()
        self._brier_alpha = float(brier_alpha)

    # ------------------------------------------------------------------ API
    @property
    def config(self) -> ConsensusConfig:
        """The active config (a copy — mutate via `update_weights`)."""
        return self._config.model_copy()

    def decide(self, estimates: Sequence[Estimate]) -> ConsensusDecision:
        """Run one fusion round and return a `ConsensusDecision`."""
        return fuse(estimates, self._config)

    def update_weights(self, performance: dict[str, float]) -> None:
        """Re-derive per-analyst weights from Brier scores.

        Lower Brier = better calibration = higher weight. Conversion:

            w_i = softmax( -α · brier_i )
                = exp(-α · brier_i) / Σ_j exp(-α · brier_j)

        Brier score is the mean squared error between predicted probabilities
        and binary outcomes: ``brier = (1/N) Σ (p_i - o_i)²`` ∈ [0, 1].
        Lower is better; 0 = perfect, 0.25 = "always predict 0.5".

        Parameters
        ----------
        performance:
            Mapping ``{analyst_id: brier_score}``. Analysts not in this dict
            keep their previous weight (or equal weight if never set).
        """
        if not performance:
            return

        # Filter to finite, non-negative Brier scores.
        items = [(k, float(v)) for k, v in performance.items() if math.isfinite(float(v))]
        if not items:
            return

        # Softmax with negative sign so lower Brier → higher weight.
        # Use max-subtraction for numerical stability.
        scores = [-self._brier_alpha * b for _, b in items]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        if total <= 0.0:
            return

        new_weights = {aid: exps[i] / total for i, (aid, _) in enumerate(items)}

        # Merge into the engine's existing weight table.
        merged: dict[str, float] = dict(self._config.weights or {})
        merged.update(new_weights)
        self._config.weights = merged

    def explain(self, decision: ConsensusDecision) -> str:
        """Produce a human-readable explanation of how the decision was reached.

        This string is intended for the audit log and the demo replay UI 
        it lets a judge (or a future you) see at a glance why the mesh chose
        to trade / skip / wait on a given market.
        """
        method_blurb = {
            "logit-mean": (
                "Logit-space weighted mean: each p was mapped to logit(p) = "
                "ln(p/(1-p)) (clamped to [0.01, 0.99]), the weighted mean of "
                "the logits was taken, then sigmoid was applied to recover a "
                "probability. This preserves extreme-confidence signals "
                "better than an arithmetic mean."
            ),
            "median": (
                "Weighted median of probabilities. Robust to a single "
                "outlier analyst."
            ),
            "trimmed-mean": (
                "Trimmed weighted mean: the single highest and single lowest "
                "probabilities were dropped, then the weighted mean of the "
                "rest was taken. Falls back to plain weighted mean for n<3."
            ),
        }.get(decision.method, "Unknown fusion method.")

        gate_blurb = {
            "trade": (
                "GATE=trade: enough analysts and agreement above threshold — "
                "pass to pythia-risk for sizing."
            ),
            "skip": (
                "GATE=skip: agreement below threshold — analysts disagree too "
                "much, no edge. Do not trade."
            ),
            "wait": (
                "GATE=wait: fewer than min_analysts contributed — poll the "
                "mesh again before deciding."
            ),
        }.get(decision.gate, "Unknown gate.")

        weights_lines = "\n".join(
            f"      - {aid}: {w:.4f}"
            for aid, w in sorted(
                decision.weights_used.items(), key=lambda kv: kv[1], reverse=True
            )
        ) or "      (none — equal weights used)"

        return (
            f"ConsensusDecision for market '{decision.market_id}'\n"
            f"  consensus_prob   = {decision.consensus_prob:.4f}\n"
            f"  agreement_score  = {decision.agreement_score:.4f}\n"
            f"  gate             = {decision.gate}\n"
            f"  method           = {decision.method}\n"
            f"  contributors     = {', '.join(decision.contributor_ids)} "
            f"({len(decision.contributor_ids)} analyst(s))\n"
            f"  weights_used:\n{weights_lines}\n"
            f"  timestamp        = {decision.timestamp}\n"
            f"\nWhy this gate?\n  {gate_blurb}\n"
            f"\nHow was the probability fused?\n  {method_blurb}\n"
        )

__all__ = ["ConsensusEngine"]
