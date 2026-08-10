"""Kelly criterion sizing for multi-outcome LMSR markets.

Delphi markets are multi-outcome LMSR (Logarithmic Market Scoring Rule). A
market has N outcomes (e.g. ``["YES", "NO"]`` or
``["Bitcoin", "Ethereum", "Solana", "Other"]``) and a spot price per outcome
that acts as the market's implied probability for that outcome.

From the buyer's perspective, buying shares of outcome ``i`` is a binary bet:
  - Pay ``m_i`` per share (the current spot price for outcome ``i``).
  - If outcome ``i`` wins, each share pays ``1.0`` (in competition tokens).
  - If outcome ``i`` loses, each share pays ``0.0``.

So the Kelly fraction for a single outcome is the same as the binary case:

    f* = (p_i - m_i) / (1 - m_i)

where ``p_i`` is our consensus probability that outcome ``i`` wins and ``m_i``
is the market's current price for outcome ``i``.

Multi-outcome generalization
----------------------------

The generalization is NOT in the per-outcome formula — it's in:

1. The consensus layer producing a probability distribution over N outcomes
   (not just ``P(YES)``).
2. The risk engine iterating over all outcomes, computing Kelly for each,
   and selecting the one with the largest positive edge.
3. The ``Market`` type carrying ``outcomes: list[str]`` and
   ``spot_prices: list[float]`` instead of ``yes_price: float``.

LMSR prices sum to ~1.0 (the cost function enforces this), so the outcomes
are NOT independent — but the per-outcome Kelly only depends on ``p_i`` and
``m_i``, so we can evaluate each independently and pick the best.

Fractional Kelly
----------------

Full Kelly maximizes long-run log-wealth growth *if* ``p`` is exactly
correct. In practice ``p`` is an LLM-fused consensus with calibration error,
so we apply a fraction multiplier (default 0.25 = quarter-Kelly):

    f_final = fraction * f*

Quarter-Kelly sacrifices ~25% of expected log-growth for ~4x lower variance
and robustness to ~25% miscalibration in ``p``. It is the default in
``configs/live-mvp.toml``.

Edge cases
----------

    p_i == m_i  → f* = 0  (no edge on outcome i)
    p_i <  m_i  → f* < 0  (negative edge, skip this outcome)
    p_i → 1, m_i → 0 → f* → 1 (capped by max_stake_usd and exposure)
    m_i → 0 or m_i → 1 → denominator blows up → guarded with max(1e-6, ...)

Backward compatibility
----------------------

The original binary-market functions (``_compute_kelly_fraction``,
``size_trade_kelly``, ``size_trade_fixed``) are preserved for backward
compatibility. Binary markets are the N=2 special case where
``outcomes = ["YES", "NO"]`` and ``spot_prices = [yes_price, 1 - yes_price]``.
"""

from __future__ import annotations

from typing import Sequence


def _compute_kelly_fraction(
    p_consensus: float,
    market_price: float,
    fraction: float = 0.25,
) -> float:
    """Compute the fractional-Kelly bankroll fraction for a single outcome.

    This is the per-outcome Kelly formula. For a multi-outcome market, call
    this once per outcome and pick the one with the largest positive result.

    Parameters
    ----------
    p_consensus : float
        Consensus probability that this outcome wins, in [0, 1].
    market_price : float
        Current spot price for this outcome (also a probability, in [0, 1]).
    fraction : float, optional
        Fractional Kelly multiplier (0.25 = quarter-Kelly), by default 0.25.

    Returns
    -------
    float
        Fraction of bankroll to stake, clamped to ``[0, 1]`` (no leverage, no
        shorting). Returns 0.0 when there is no edge or a negative edge.

    Math
    ----
        f* = (p - m) / (1 - m)       (full Kelly, compact form)
        f_final = fraction * clamp(f*, 0, 1)
    """
    if not 0.0 <= p_consensus <= 1.0:
        raise ValueError(f"p_consensus must be in [0, 1], got {p_consensus!r}")
    if not 0.0 <= market_price <= 1.0:
        raise ValueError(f"market_price must be in [0, 1], got {market_price!r}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")

    denom = max(1.0 - market_price, 1e-6)
    full_kelly = (p_consensus - market_price) / denom
    clamped = max(0.0, min(1.0, full_kelly))
    return fraction * clamped


# Public alias — kept for backward compatibility.
kelly_fraction = _compute_kelly_fraction


def size_trade_kelly(
    p_consensus: float,
    market_price: float,
    bankroll_usd: float,
    kelly_fraction: float = 0.25,
    max_stake_usd: float = 50.0,
) -> float:
    """Compute the dollar stake for a single-outcome bet using fractional Kelly.

    This is the binary-market API. For multi-outcome markets, use
    ``size_trade_kelly_multi`` instead.

    Parameters
    ----------
    p_consensus : float
        Consensus probability that this outcome wins, in [0, 1].
    market_price : float
        Current spot price for this outcome, in [0, 1].
    bankroll_usd : float
        Current total bankroll in USD (cash + open positions).
    kelly_fraction : float, optional
        Fractional Kelly multiplier (0.25 = quarter-Kelly), by default 0.25.
    max_stake_usd : float, optional
        Hard cap on the dollar stake, by default 50.0.

    Returns
    -------
    float
        Dollar stake, in ``[0, max_stake_usd]``. Rounded to 2 decimal places.
    """
    if bankroll_usd < 0.0:
        raise ValueError(f"bankroll_usd must be >= 0, got {bankroll_usd!r}")
    if max_stake_usd <= 0.0:
        raise ValueError(f"max_stake_usd must be > 0, got {max_stake_usd!r}")

    f = _compute_kelly_fraction(
        p_consensus=p_consensus,
        market_price=market_price,
        fraction=kelly_fraction,
    )
    stake = f * bankroll_usd
    stake = min(stake, max_stake_usd)
    return round(stake, 2)


def size_trade_kelly_multi(
    consensus_probs: Sequence[float],
    spot_prices: Sequence[float],
    bankroll_usd: float,
    kelly_fraction: float = 0.25,
    max_stake_usd: float = 50.0,
    no_edge_tolerance: float = 0.02,
) -> tuple[int, float, float]:
    """Find the best outcome to bet on and compute the Kelly stake.

    Iterates over all outcomes, computes the fractional-Kelly stake for each,
    and returns the one with the largest positive edge.

    Parameters
    ----------
    consensus_probs : Sequence[float]
        Consensus probability for each outcome, in [0, 1]. Length N.
    spot_prices : Sequence[float]
        Current spot price for each outcome, in [0, 1]. Length N (must match
        ``consensus_probs``).
    bankroll_usd : float
        Current total bankroll in USD.
    kelly_fraction : float, optional
        Fractional Kelly multiplier (0.25 = quarter-Kelly), by default 0.25.
    max_stake_usd : float, optional
        Hard cap on the dollar stake, by default 50.0.
    no_edge_tolerance : float, optional
        If the best edge (|p_i - m_i|) is below this, return a zero stake
        (no edge), by default 0.02.

    Returns
    -------
    tuple[int, float, float]
        ``(best_outcome_idx, stake_usd, edge)`` where:
        - ``best_outcome_idx`` is the 0-based index of the chosen outcome
          (or -1 if no outcome has a positive edge).
        - ``stake_usd`` is the dollar stake, in [0, max_stake_usd].
        - ``edge`` is ``p_i - m_i`` for the chosen outcome (positive = we
          think the outcome is underpriced).

    Raises
    ------
    ValueError
        If ``consensus_probs`` and ``spot_prices`` have different lengths,
        or if either contains values outside [0, 1].
    """
    if len(consensus_probs) != len(spot_prices):
        raise ValueError(
            f"consensus_probs (len={len(consensus_probs)}) and "
            f"spot_prices (len={len(spot_prices)}) must have the same length"
        )
    if len(consensus_probs) == 0:
        raise ValueError("consensus_probs must not be empty")
    if bankroll_usd < 0.0:
        raise ValueError(f"bankroll_usd must be >= 0, got {bankroll_usd!r}")
    if max_stake_usd <= 0.0:
        raise ValueError(f"max_stake_usd must be > 0, got {max_stake_usd!r}")

    best_idx = -1
    best_edge = 0.0
    best_stake = 0.0

    for i, (p_i, m_i) in enumerate(zip(consensus_probs, spot_prices)):
        if not 0.0 <= p_i <= 1.0:
            raise ValueError(
                f"consensus_probs[{i}] must be in [0, 1], got {p_i!r}"
            )
        if not 0.0 <= m_i <= 1.0:
            raise ValueError(
                f"spot_prices[{i}] must be in [0, 1], got {m_i!r}"
            )

        edge = p_i - m_i
        if edge <= no_edge_tolerance:
            continue  # no positive edge beyond tolerance

        stake = size_trade_kelly(
            p_consensus=p_i,
            market_price=m_i,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            max_stake_usd=max_stake_usd,
        )
        if stake > best_stake:
            best_idx = i
            best_edge = edge
            best_stake = stake

    return best_idx, best_stake, best_edge


def size_trade_fixed(
    fixed_stake_usd: float,
    max_stake_usd: float,
) -> float:
    """Compute a fixed (non-Kelly) dollar stake, capped by ``max_stake_usd``.

    Used when ``RiskConfig.sizing == "fixed"`` — typically for paper-trading or
    sanity-check runs where you want a constant stake per trade regardless of
    edge.
    """
    if fixed_stake_usd < 0.0:
        raise ValueError(f"fixed_stake_usd must be >= 0, got {fixed_stake_usd!r}")
    if max_stake_usd <= 0.0:
        raise ValueError(f"max_stake_usd must be > 0, got {max_stake_usd!r}")
    return round(min(fixed_stake_usd, max_stake_usd), 2)
