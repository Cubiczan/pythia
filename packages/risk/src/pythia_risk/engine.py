"""RiskEngine: the 6-gate evaluation pipeline for Delphi trades.

The engine is called by ``pythia-executor`` after ``pythia-consensus`` has
produced a ``ConsensusDecision``. It runs six gates, in order, and returns a
``TradePlan``:

    1. Market-type allowed?  → REJECT  flag: ``market_type_not_allowed``
    2. Drawdown breaker?     → REJECT  flag: ``drawdown_breaker``
    3. Cool-down active?     → REJECT  flag: ``cool_down_active``
    4. Exposure cap hit?     → size reduced (or REJECT if reduced to 0)
                              flag: ``exposure_cap``
    5. No edge?              → REJECT  flag: ``no_edge``
    6. Per-market cap?       → size reduced (silent)

If all gates pass (or gate 4 reduces-but-doesn't-reject), the engine returns
``APPROVE`` with the computed size, side, and outcome_idx.

Multi-outcome markets
---------------------

Delphi markets are multi-outcome LMSR. The engine receives a ``Market`` with
``outcomes: list[str]`` and ``spot_prices: list[float]``, and a
``ConsensusDecision`` with ``consensus_prob`` (the probability of the
market's primary outcome — for binary markets this is P(YES)).

For binary markets (``outcomes = ["YES", "NO"]``), the engine uses
``consensus_prob`` as P(YES) and ``1 - consensus_prob`` as P(NO), then picks
the side with the larger positive edge.

For N-outcome markets, the engine distributes ``consensus_prob`` across
outcomes using the spot price ratio as a prior (since the consensus layer
currently produces a single probability, not a full distribution). This is
a pragmatic approximation — a future consensus upgrade will produce a full
``probabilities: list[float]`` distribution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pythia_risk.sizing import (
    _compute_kelly_fraction,
    size_trade_fixed,
    size_trade_kelly,
    size_trade_kelly_multi,
)
from pythia_risk.types import (
    BankrollState,
    ConsensusDecision,
    Market,
    RiskConfig,
    TradePlan,
    TradeReceipt,
)

# Default tolerance for the no-edge gate. Overridden by config.no_edge_tolerance.
DEFAULT_NO_EDGE_TOLERANCE: float = 0.02


class RiskEngine:
    """Delphi risk-gating engine wrapping icohangar-ops/meshcfo.

    Parameters
    ----------
    config : RiskConfig
        Risk configuration (sizing method, caps, market-type rules, etc.).

    Attributes
    ----------
    config : RiskConfig
        The config the engine was constructed with.
    state : BankrollState
        Mutable internal state, updated by ``update_state`` /
        ``record_loss`` / ``record_win``. Note that ``evaluate`` does NOT
        read from ``self.state`` — it takes a bankroll parameter explicitly.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config: RiskConfig = config
        self.state: BankrollState = BankrollState(
            cash_usd=config.max_total_exposure_usd,
            open_positions_usd=0.0,
            peak_bankroll_usd=config.max_total_exposure_usd,
            current_bankroll_usd=config.max_total_exposure_usd,
            drawdown_pct=0.0,
            last_loss_at=None,
        )

    # ------------------------------------------------------------------ evaluate

    def evaluate(
        self,
        decision: ConsensusDecision,
        market: Market,
        current_bankroll: BankrollState,
    ) -> TradePlan:
        """Run the 6-gate pipeline and return a TradePlan.

        Parameters
        ----------
        decision : ConsensusDecision
            Fused analyst output (from pythia_consensus).
        market : Market
            Market metadata (outcomes, spot_prices, category, ...).
        current_bankroll : BankrollState
            Live bankroll snapshot.

        Returns
        -------
        TradePlan
            APPROVE / REJECT with rationale + risk_flags + outcome_idx.
        """
        flags: list[str] = []
        now = datetime.now(timezone.utc)
        consensus_prob = float(decision.consensus_prob)
        market_id = market.market_id

        # ----- Step 1: market-type allowed? -------------------------------
        rules = self.config.market_type_rules.get(market.category)
        if rules is None or not rules.allowed:
            flags.append("market_type_not_allowed")
            return TradePlan(
                market_id=market_id,
                side=market.outcomes[0] if market.outcomes else "YES",
                outcome_idx=0,
                size_usd=0.0,
                limit_price=None,
                rationale=(
                    f"Market category '{market.category}' is not in the allowed "
                    f"market_type_rules (or explicitly disallowed)."
                ),
                risk_flags=flags,
                decision="REJECT",
                timestamp=now.isoformat(),
            )

        # ----- Step 2: drawdown breaker? -----------------------------------
        if current_bankroll.drawdown_pct >= self.config.max_drawdown_pct:
            flags.append("drawdown_breaker")
            return TradePlan(
                market_id=market_id,
                side=market.outcomes[0] if market.outcomes else "YES",
                outcome_idx=0,
                size_usd=0.0,
                limit_price=None,
                rationale=(
                    f"Drawdown {current_bankroll.drawdown_pct:.2f}% >= "
                    f"max_drawdown_pct {self.config.max_drawdown_pct:.2f}%. "
                    f"Circuit breaker tripped; trading paused."
                ),
                risk_flags=flags,
                decision="REJECT",
                timestamp=now.isoformat(),
            )

        # ----- Step 3: cool-down active? -----------------------------------
        if current_bankroll.last_loss_at is not None:
            elapsed_min = (now - current_bankroll.last_loss_at).total_seconds() / 60.0
            if elapsed_min < self.config.cool_down_min_after_loss:
                flags.append("cool_down_active")
                return TradePlan(
                    market_id=market_id,
                    side=market.outcomes[0] if market.outcomes else "YES",
                    outcome_idx=0,
                    size_usd=0.0,
                    limit_price=None,
                    rationale=(
                        f"Cool-down active: {elapsed_min:.1f} min since last loss, "
                        f"< {self.config.cool_down_min_after_loss:.1f} min threshold."
                    ),
                    risk_flags=flags,
                    decision="REJECT",
                    timestamp=now.isoformat(),
                )

        # ----- Step 5: compute per-outcome edges and pick the best ---------
        # Build per-outcome consensus probabilities. For binary markets,
        # consensus_prob is P(YES) and P(NO) = 1 - p. For N-outcome markets,
        # we distribute consensus_prob across outcomes using spot prices as
        # a prior (a pragmatic approximation until consensus produces a full
        # distribution).
        outcomes = market.outcomes or ["YES", "NO"]
        spot_prices = market.spot_prices or [0.5, 0.5]
        if len(spot_prices) < len(outcomes):
            # Pad with uniform distribution if we're short.
            n = len(outcomes)
            spot_prices = [1.0 / n] * n

        if len(outcomes) == 2:
            # Binary: consensus_prob is P(outcomes[0]).
            consensus_probs = [consensus_prob, 1.0 - consensus_prob]
        else:
            # N-outcome: distribute consensus_prob proportional to spot prices.
            # This is a placeholder until consensus produces a full distribution.
            # The idea: if the consensus is "bullish" (prob > 0.5 on the primary
            # outcome), scale up the outcomes with higher spot prices; if
            # bearish, scale them down. We normalize at the end.
            total_spot = sum(spot_prices) or 1.0
            weights = [s / total_spot for s in spot_prices]
            # Scale: if consensus_prob > 0.5, boost; if < 0.5, dampen.
            # Simple approach: consensus_probs[i] = weights[i] * consensus_prob * n
            # (this keeps the relative ordering and sums to consensus_prob * n).
            # Then renormalize to sum to 1.
            n = len(outcomes)
            raw = [w * consensus_prob * n for w in weights]
            total_raw = sum(raw) or 1.0
            consensus_probs = [r / total_raw for r in raw]

        # Find the best outcome to bet on.
        no_edge_tol = getattr(self.config, "no_edge_tolerance", DEFAULT_NO_EDGE_TOLERANCE)
        best_idx, best_stake, best_edge = size_trade_kelly_multi(
            consensus_probs=consensus_probs,
            spot_prices=spot_prices,
            bankroll_usd=current_bankroll.current_bankroll_usd,
            kelly_fraction=self.config.kelly_fraction,
            max_stake_usd=min(
                self.config.max_stake_per_market_usd,
                rules.max_stake_usd,
            ),
            no_edge_tolerance=no_edge_tol,
        )

        # ----- Step 5b: no edge? -------------------------------------------
        if best_idx < 0 or best_stake <= 0.0:
            flags.append("no_edge")
            # Build a descriptive message for the first outcome with the closest edge.
            edges = [consensus_probs[i] - spot_prices[i] for i in range(len(outcomes))]
            max_edge = max(edges) if edges else 0.0
            return TradePlan(
                market_id=market_id,
                side=outcomes[0],
                outcome_idx=0,
                size_usd=0.0,
                limit_price=None,
                rationale=(
                    f"No edge: best outcome edge={max_edge:+.4f} < "
                    f"tolerance {no_edge_tol:.4f}. "
                    f"Outcomes={outcomes}, spot_prices={[round(s, 4) for s in spot_prices]}, "
                    f"consensus_probs={[round(p, 4) for p in consensus_probs]}."
                ),
                risk_flags=flags,
                decision="REJECT",
                timestamp=now.isoformat(),
            )

        side = outcomes[best_idx]
        m_side = spot_prices[best_idx]
        p_side = consensus_probs[best_idx]
        stake = best_stake

        # ----- Step 6: sizing method override (fixed) ----------------------
        if self.config.sizing == "fixed":
            fixed_stake = min(
                self.config.max_stake_per_market_usd,
                rules.max_stake_usd,
            )
            stake = size_trade_fixed(
                fixed_stake_usd=fixed_stake,
                max_stake_usd=fixed_stake,
            )
            sizing_note = f"fixed stake on {side} (idx={best_idx}): ${stake:.2f}"
        else:
            sizing_note = (
                f"fractional-Kelly (fraction={self.config.kelly_fraction:.2f}) "
                f"stake on {side} (idx={best_idx}): p={p_side:.4f}, m={m_side:.4f}, "
                f"bankroll=${current_bankroll.current_bankroll_usd:.2f}"
            )

        # ----- Step 4: exposure cap ----------------------------------------
        available_headroom = max(
            0.0,
            self.config.max_total_exposure_usd - current_bankroll.open_positions_usd,
        )
        if stake > available_headroom:
            stake = round(available_headroom, 2)
            flags.append("exposure_cap")
            if stake <= 0.0:
                return TradePlan(
                    market_id=market_id,
                    side=side,
                    outcome_idx=best_idx,
                    size_usd=0.0,
                    limit_price=None,
                    rationale=(
                        f"Exposure cap exhausted: open_positions="
                        f"${current_bankroll.open_positions_usd:.2f} >= "
                        f"max_total_exposure_usd=${self.config.max_total_exposure_usd:.2f}."
                    ),
                    risk_flags=flags,
                    decision="REJECT",
                    timestamp=now.isoformat(),
                )

        # ----- Step 7: per-market cap (silent reduce) ---------------------
        cap = min(self.config.max_stake_per_market_usd, rules.max_stake_usd)
        if stake > cap:
            stake = round(cap, 2)

        # ----- Build the rationale & return -------------------------------
        full_kelly = _compute_kelly_fraction(p_side, m_side, fraction=1.0)
        rationale = (
            f"{sizing_note}. "
            f"Full-Kelly fraction would be {full_kelly:.4f} of bankroll; "
            f"applied fraction={self.config.kelly_fraction:.2f} → "
            f"final stake ${stake:.2f} on {side} (outcome_idx={best_idx}). "
            f"Edge={best_edge:+.4f} (consensus {p_side:.4f} vs price {m_side:.4f})."
        )

        return TradePlan(
            market_id=market_id,
            side=side,
            outcome_idx=best_idx,
            size_usd=stake,
            limit_price=m_side,
            rationale=rationale,
            risk_flags=flags,
            decision="APPROVE",
            timestamp=now.isoformat(),
        )

    # ------------------------------------------------------------------ state ops

    def update_state(self, trade: TradeReceipt) -> None:
        """Update internal BankrollState after a trade fills."""
        size = float(trade.size_usd)
        self.state.cash_usd = max(0.0, self.state.cash_usd - size)
        self.state.open_positions_usd += size
        self.state.current_bankroll_usd = (
            self.state.cash_usd + self.state.open_positions_usd
        )
        if self.state.current_bankroll_usd > self.state.peak_bankroll_usd:
            self.state.peak_bankroll_usd = self.state.current_bankroll_usd
        self._recompute_drawdown()

    def record_loss(self, market_id: str, loss_usd: float) -> None:
        """Record a realised loss on a settled market."""
        if loss_usd < 0.0:
            raise ValueError(f"loss_usd must be >= 0, got {loss_usd!r}")
        self.state.open_positions_usd = max(
            0.0, self.state.open_positions_usd - loss_usd
        )
        self.state.current_bankroll_usd = max(
            0.0, self.state.current_bankroll_usd - loss_usd
        )
        self.state.last_loss_at = datetime.now(timezone.utc)
        self._recompute_drawdown()

    def record_win(self, market_id: str, win_usd: float) -> None:
        """Record a realised win on a settled market."""
        if win_usd < 0.0:
            raise ValueError(f"win_usd must be >= 0, got {win_usd!r}")
        self.state.current_bankroll_usd += win_usd
        if self.state.current_bankroll_usd > self.state.peak_bankroll_usd:
            self.state.peak_bankroll_usd = self.state.current_bankroll_usd
        self._recompute_drawdown()

    # ------------------------------------------------------------------ internals

    def _recompute_drawdown(self) -> None:
        """Recompute ``state.drawdown_pct`` off the current peak."""
        peak = self.state.peak_bankroll_usd
        if peak <= 0.0:
            self.state.drawdown_pct = 0.0
            return
        dd = (peak - self.state.current_bankroll_usd) / peak * 100.0
        self.state.drawdown_pct = max(0.0, dd)
