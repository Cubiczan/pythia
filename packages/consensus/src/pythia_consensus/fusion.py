"""Core fusion math for pythia-consensus.

This module is the heart of the wrapper: it turns N analyst estimates into a
single probability + agreement score + gate decision. The math is fully
self-contained — it does **not** depend on the upstream
`consensus-hardening-protocol`, which is only used by `audit.py` for signing.

Formulas
--------
Given estimates with probabilities ``p_i`` and (normalised) weights ``w_i``:

**logit-mean** (default):

    logit(p) = ln( p / (1 - p) )                # p clamped to [0.01, 0.99]
    L* = Σ w_i · logit(p_i)
    p* = sigmoid(L*) = 1 / (1 + exp(-L*))

**median**:

    p* = weighted median of {p_i} with weights {w_i}

**trimmed-mean**:

    drop the single highest and single lowest p_i (one each end), then
    weighted-mean the rest. If n < 3, falls back to plain weighted mean.

**agreement_score**:

    p̄_w = Σ w_i · p_i                          # weighted mean
    σ_w = sqrt( Σ w_i · (p_i - p̄_w)² )          # weighted population stddev
    agreement_score = clamp(1 - σ_w / 0.5, 0, 1)

Rationale: identical analysts ⇒ σ_w = 0 ⇒ score 1.0; maximally-split analysts
(half at 0, half at 1) ⇒ σ_w ≈ 0.5 ⇒ score 0.0.

**gate**:

    if n < min_analysts:                gate = "wait"
    elif agreement_score < threshold:   gate = "skip"
    else:                               gate = "trade"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from scipy.special import expit  # numerically stable sigmoid

from .types import ConsensusConfig, ConsensusDecision, ConsensusMethod, Estimate

# Probabilities are clamped into this range before being mapped to logit space,
# so log(0) / div-by-zero never occurs. 0.01 ↔ logit ≈ -4.595, 0.99 ↔ +4.595.
_P_CLAMP_LO = 0.01
_P_CLAMP_HI = 0.99

# The denominator in the agreement-score formula. σ_w = 0.5 corresponds to the
# theoretical maximum disagreement (half the mass at p=0, half at p=1).
_AGREEMENT_SIGMA_MAX = 0.5

def _normalise_weights(
    estimates: Sequence[Estimate],
    weights: dict[str, float] | None,
) -> np.ndarray:
    """Return a normalised weight vector aligned with `estimates`.

    Unknown analysts default to weight 1.0 (treated as equal). Non-finite or
    non-positive weights are clamped to a tiny epsilon so they cannot zero out
    a contributor silently.
    """
    n = len(estimates)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    raw = np.empty(n, dtype=np.float64)
    for i, est in enumerate(estimates):
        if weights and est.analyst_id in weights:
            w = float(weights[est.analyst_id])
        else:
            w = 1.0
        # Guard against NaN / inf / non-positive weights.
        if not np.isfinite(w) or w <= 0.0:
            w = 1e-9
        raw[i] = w

    total = raw.sum()
    if total <= 0.0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return raw / total

def _weighted_mean(probs: np.ndarray, weights: np.ndarray) -> float:
    """Σ w_i · p_i for already-normalised weights."""
    if probs.size == 0:
        return 0.0
    return float(np.dot(weights, probs))

def _weighted_stddev(probs: np.ndarray, weights: np.ndarray) -> float:
    """Weighted population standard deviation.

    σ_w = sqrt( Σ w_i · (p_i - p̄_w)² )
    """
    if probs.size == 0:
        return 0.0
    mean = _weighted_mean(probs, weights)
    var = float(np.dot(weights, (probs - mean) ** 2))
    return float(np.sqrt(max(var, 0.0)))

def _weighted_median(probs: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median.

    Standard definition: sort the probabilities, accumulate weights, return
    the smallest probability whose cumulative weight is ≥ 0.5.

    For equal weights and odd ``n``, this matches the plain median (middle
    element). For equal weights and even ``n``, it returns the *lower* of the
    two middle elements (a common weighted-median convention; the alternative
    "average of the two middle" is not weighted-median-standard).
    """
    if probs.size == 0:
        return 0.0
    order = np.argsort(probs, kind="stable")
    p_sorted = probs[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    # Smallest idx such that cum[idx] >= 0.5. searchsorted with side="left"
    # returns exactly this: the first index whose cum value is >= 0.5.
    idx = int(np.searchsorted(cum, 0.5, side="left"))
    if idx >= len(p_sorted):
        # Floating-point edge case: cumulative weight never quite reaches 0.5
        # due to rounding. Return the largest probability.
        return float(p_sorted[-1])
    return float(p_sorted[idx])

# ---------------------------------------------------------------------------
# Public fusion functions.
# ---------------------------------------------------------------------------
def fuse_logit_mean(probs: np.ndarray, weights: np.ndarray) -> float:
    """Logit-space weighted mean, mapped back via sigmoid.

    p_i is clamped to [0.01, 0.99] before logit to avoid div-by-zero and
    infinite logits.
    """
    if probs.size == 0:
        return 0.0
    p_clamped = np.clip(probs, _P_CLAMP_LO, _P_CLAMP_HI)
    logits = np.log(p_clamped / (1.0 - p_clamped))
    L_star = float(np.dot(weights, logits))
    return float(expit(L_star))

def fuse_median(probs: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median."""
    return _weighted_median(probs, weights)

def fuse_trimmed_mean(probs: np.ndarray, weights: np.ndarray) -> float:
    """Drop highest and lowest, weighted-mean the rest.

    For n < 3, no trimming is possible, so we fall back to plain weighted mean.
    """
    if probs.size == 0:
        return 0.0
    if probs.size < 3:
        return _weighted_mean(probs, weights)
    order = np.argsort(probs)
    # Drop first (lowest) and last (highest) by index.
    trimmed_idx = order[1:-1]
    p_t = probs[trimmed_idx]
    w_t = weights[trimmed_idx]
    total = w_t.sum()
    if total <= 0.0:
        return _weighted_mean(probs, weights)
    return float(np.dot(w_t / total, p_t))

_FUSION_DISPATCH: dict[ConsensusMethod, callable] = {  # type: ignore[type-arg]
    "logit-mean": fuse_logit_mean,
    "median": fuse_median,
    "trimmed-mean": fuse_trimmed_mean,
}

def agreement_score(
    estimates: Sequence[Estimate],
    weights: dict[str, float] | None = None,
) -> float:
    """How aligned the analysts are, on a 0..1 scale.

    ``1.0`` = all analysts gave identical probabilities.
    ``0.0`` = analysts are maximally split (some at 0, some at 1).

    Computed as ``clamp(1 - σ_w / 0.5, 0, 1)`` where ``σ_w`` is the weighted
    population stddev of the probabilities.
    """
    if not estimates:
        return 0.0
    probs = np.asarray([e.probability for e in estimates], dtype=np.float64)
    w = _normalise_weights(estimates, weights)
    sigma = _weighted_stddev(probs, w)
    score = 1.0 - (sigma / _AGREEMENT_SIGMA_MAX)
    return float(max(0.0, min(1.0, score)))

def _decide_gate(
    n: int,
    score: float,
    config: ConsensusConfig,
) -> str:
    if n < config.min_analysts:
        return "wait"
    if score < config.agreement_threshold:
        return "skip"
    return "trade"

def fuse(estimates: Sequence[Estimate], config: ConsensusConfig) -> ConsensusDecision:
    """Fuse N analyst estimates into a single `ConsensusDecision`.

    Parameters
    ----------
    estimates:
        List of `Estimate` objects. Should all share the same `market_id`;
        if they don't, the first estimate's `market_id` is used.
    config:
        Fusion method, agreement threshold, min analysts, optional weights.

    Returns
    -------
    ConsensusDecision
        With `consensus_prob`, `agreement_score`, `gate`, `contributor_ids`,
        `method`, `weights_used`, and `timestamp` populated.
    """
    if not estimates:
        raise ValueError("fuse() requires at least one Estimate; got 0.")

    market_id = estimates[0].market_id
    contributor_ids = [e.analyst_id for e in estimates]

    weights = _normalise_weights(estimates, config.weights)
    weights_used = {
        est.analyst_id: float(w) for est, w in zip(estimates, weights, strict=True)
    }

    probs = np.asarray([e.probability for e in estimates], dtype=np.float64)

    fuse_fn = _FUSION_DISPATCH[config.method]
    consensus_prob = float(fuse_fn(probs, weights))

    # Clamp into [0, 1] just in case of floating point excursions.
    consensus_prob = float(max(0.0, min(1.0, consensus_prob)))

    score = agreement_score(estimates, config.weights)
    gate = _decide_gate(len(estimates), score, config)

    return ConsensusDecision(
        market_id=market_id,
        consensus_prob=consensus_prob,
        agreement_score=score,
        gate=gate,
        contributor_ids=contributor_ids,
        method=config.method,
        weights_used=weights_used,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

__all__ = [
    "agreement_score",
    "fuse",
    "fuse_logit_mean",
    "fuse_median",
    "fuse_trimmed_mean",
]
