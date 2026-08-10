"""Kelly criterion sizing for binary-outcome markets.

This module implements the Kelly criterion for Delphi / Polymarket-style binary
markets where:
    - A YES share pays out $1.00 if the event happens, $0.00 otherwise.
    - You can buy YES at the current ``market_price`` (a probability quote
      between 0 and 1).
    - No shorting, no leverage — Kelly fractions are clamped to [0, 1].

The math
--------

Let:
    p = consensus probability that YES wins  (from pythia_consensus)
    q = 1 - p                                 (probability NO wins)
    m = market_price                          (current YES share price)
    b = net odds received per dollar staked   = (1 - m) / m

Why ``b = (1 - m) / m``:
    Spend $1 on YES shares at price m → you receive 1/m shares.
    If YES wins, each share pays $1 → you receive $1/m.
    Net profit = $1/m - $1 = $(1 - m) / m.
    If NO wins, you lose the $1.

Full Kelly fraction of bankroll to stake:

    f* = (b * p - q) / b

Substituting ``b = (1 - m) / m`` and ``q = 1 - p``:

    f* = (b * p - (1 - p)) / b
       = p - (1 - p) / b
       = p - (1 - p) * m / (1 - m)

A more compact equivalent form (multiply top & bottom by m):

    f* = (p - m) / (1 - m)

i.e. **full-Kelly stake as a fraction of bankroll = edge / gross-payoff-odds**.

Fractional Kelly
----------------

Full Kelly maximises long-run log-wealth growth *if* p is exactly correct. In
practice p is an LLM-fused consensus with calibration error, so we apply a
fraction multiplier (default 0.25 = quarter-Kelly):

    f_final = fraction * f*

Quarter-Kelly sacrifices ~25% of expected log-growth for ~4x lower variance
and robustness to ~25% miscalibration in p. It is the default in
``configs/live-mvp.toml``.

Edge cases
----------

    p == m  → f* = 0  (no edge, stake is 0)
    p <  m  → f* < 0  (negative edge on YES, clamp to 0; engine flips side to NO)
    p → 1, m → 0 → f* → 1 (capped by max_stake_usd and max_total_exposure_usd)
    m → 0 or m → 1 → denominator blows up → guarded with max(1e-6, ...)

Naming note
-----------

The public function is exposed as ``kelly_fraction`` (per the spec). It is
defined internally as ``_compute_kelly_fraction`` because
``size_trade_kelly`` has a *parameter* named ``kelly_fraction`` (also per spec)
which would shadow the function inside that scope. The private alias keeps
both call sites working without renaming either the public function or the
parameter.
"""

from __future__ import annotations

def _compute_kelly_fraction(
    p_consensus: float,
    market_price: float,
    fraction: float = 0.25,
) -> float:
    """Compute the fractional-Kelly bankroll fraction for a binary YES stake.

    Parameters
    ----------
    p_consensus : float
        Consensus probability that YES wins, in [0, 1].
    market_price : float
        Current YES share price (also a probability, in [0, 1]).
    fraction : float, optional
        Fractional Kelly multiplier (0.25 = quarter-Kelly), by default 0.25.

    Returns
    -------
    float
        Fraction of bankroll to stake, clamped to ``[0, 1]`` (no leverage, no
        shorting). Returns 0.0 when there is no edge or a negative edge on YES.

    Math
    ----
        b = (1 - m) / m              (net odds per dollar)
        f* = (b * p - (1 - p)) / b   (full Kelly)
           = (p - m) / (1 - m)       (compact form)
        f_final = fraction * clamp(f*, 0, 1)

    The compact form ``f* = (p - m) / (1 - m)`` is numerically more stable than
    the ``b = (1-m)/m`` form when ``m → 0`` (the ``b`` form explodes). We use
    the compact form and guard the denominator with a small epsilon.
    """
    if not 0.0 <= p_consensus <= 1.0:
        raise ValueError(f"p_consensus must be in [0, 1], got {p_consensus!r}")
    if not 0.0 <= market_price <= 1.0:
        raise ValueError(f"market_price must be in [0, 1], got {market_price!r}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")

    # Compact form: f* = (p - m) / (1 - m)
    # Guard against m == 1 (denominator 0) by clamping to a tiny epsilon.
    denom = max(1.0 - market_price, 1e-6)
    full_kelly = (p_consensus - market_price) / denom

    # No leverage, no shorting via negative Kelly.
    clamped = max(0.0, min(1.0, full_kelly))

    return fraction * clamped

# Public alias — see module docstring "Naming note" for why this is aliased.
kelly_fraction = _compute_kelly_fraction

def size_trade_kelly(
    p_consensus: float,
    market_price: float,
    bankroll_usd: float,
    kelly_fraction: float = 0.25,
    max_stake_usd: float = 50.0,
) -> float:
    """Compute the dollar stake for a binary YES bet using fractional Kelly.

    Parameters
    ----------
    p_consensus : float
        Consensus probability that YES wins, in [0, 1].
    market_price : float
        Current YES share price, in [0, 1].
    bankroll_usd : float
        Current total bankroll in USD (cash + open positions).
    kelly_fraction : float, optional
        Fractional Kelly multiplier (0.25 = quarter-Kelly), by default 0.25.
        (Note: this parameter shadows the module-level ``kelly_fraction``
        function — see module docstring.)
    max_stake_usd : float, optional
        Hard cap on the dollar stake, by default 50.0.

    Returns
    -------
    float
        Dollar stake, in ``[0, max_stake_usd]``. Rounded to 2 decimal places
        (cents) for cleanliness.

    Math
    ----
        f = _compute_kelly_fraction(p_consensus, market_price, kelly_fraction)
        stake = f * bankroll_usd
        stake = min(stake, max_stake_usd)
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

def size_trade_fixed(
    fixed_stake_usd: float,
    max_stake_usd: float,
) -> float:
    """Compute a fixed (non-Kelly) dollar stake, capped by ``max_stake_usd``.

    Used when ``RiskConfig.sizing == "fixed"`` — typically for paper-trading or
    sanity-check runs where you want a constant stake per trade regardless of
    edge.

    Parameters
    ----------
    fixed_stake_usd : float
        The intended fixed stake. Must be >= 0.
    max_stake_usd : float
        Hard cap. Must be > 0.

    Returns
    -------
    float
        ``min(fixed_stake_usd, max_stake_usd)``, rounded to 2 decimals.
    """
    if fixed_stake_usd < 0.0:
        raise ValueError(f"fixed_stake_usd must be >= 0, got {fixed_stake_usd!r}")
    if max_stake_usd <= 0.0:
        raise ValueError(f"max_stake_usd must be > 0, got {max_stake_usd!r}")
    return round(min(fixed_stake_usd, max_stake_usd), 2)
