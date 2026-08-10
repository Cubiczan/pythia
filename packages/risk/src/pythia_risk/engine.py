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
``APPROVE`` with the computed size and side.

The engine holds an internal ``BankrollState`` that it keeps in sync via
``update_state`` / ``record_loss`` / ``record_win`` as the executor fills
trades and the market settles them. But ``evaluate`` itself takes the
bankroll as a parameter — it does not read from ``self.state``. This makes
``evaluate`` pure and trivially testable / replayable.

VERIFY comments mark places where we touch an upstream meshcfo API that may
not exist yet (the repo is currently a thin stub). When meshcfo is vendored,
re-check those call sites against the pinned commit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pythia_risk.sizing import _compute_kelly_fraction, size_trade_fixed, size_trade_kelly
from pythia_risk.types import (
    BankrollState,
    ConsensusDecision,
    Market,
    RiskConfig,
    TradePlan,
    TradeReceipt,
)

# Tolerance for the no-edge gate: if |consensus_prob - market.yes_price| < this,
# we treat it as no edge. 2 percentage points (0.02) per the design.
NO_EDGE_TOLERANCE: float = 0.02

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
        # Initial state mirrors the config's total exposure cap as starting
        # cash. Real callers should overwrite this with the live ledger state
        # before the first evaluate() call.
        # VERIFY: meshcfo may expose a `Ledger.snapshot() -> BankrollState`
        # method we should call here instead of synthesising a state.
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
            Market metadata (id, yes_price, category, ...).
        current_bankroll : BankrollState
            Live bankroll snapshot.

        Returns
        -------
        TradePlan
            APPROVE / REJECT with rationale + risk_flags.
        """
        flags: list[str] = []
        now = datetime.now(timezone.utc)
        consensus_prob = float(decision.consensus_prob)
        market_price = float(market.yes_price)
        market_id = market.market_id

        # ----- Step 1: market-type allowed? -------------------------------
        rules = self.config.market_type_rules.get(market.category)
        if rules is None or not rules.allowed:
            flags.append("market_type_not_allowed")
            return TradePlan(
                market_id=market_id,
                side="YES",
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
                side="YES",
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
                    side="YES",
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

        # ----- Step 6 (early): side determination -------------------------
        # We need the side before sizing (Kelly differs for YES vs NO).
        # Step 5 (no-edge) is also checked here since it gates side selection.
        edge = consensus_prob - market_price
        if abs(edge) < NO_EDGE_TOLERANCE:
            flags.append("no_edge")
            return TradePlan(
                market_id=market_id,
                side="YES",
                size_usd=0.0,
                limit_price=None,
                rationale=(
                    f"No edge: |consensus_prob {consensus_prob:.4f} - "
                    f"yes_price {market_price:.4f}| = {abs(edge):.4f} < "
                    f"tolerance {NO_EDGE_TOLERANCE:.4f}."
                ),
                risk_flags=flags,
                decision="REJECT",
                timestamp=now.isoformat(),
            )

        if edge > 0:
            side: str = "YES"
        else:
            side = "NO"

        # ----- Step 5: sizing ----------------------------------------------
        # Kelly math is symmetric: from the NO buyer's perspective, the
        # market price for NO is (1 - m), and the consensus probability of
        # NO winning is (1 - p). So we feed the flipped values into the same
        # kelly_fraction function.
        if side == "YES":
            p_side = consensus_prob
            m_side = market_price
        else:
            p_side = 1.0 - consensus_prob
            m_side = 1.0 - market_price

        if self.config.sizing == "kelly-fractional":
            stake = size_trade_kelly(
                p_consensus=p_side,
                market_price=m_side,
                bankroll_usd=current_bankroll.current_bankroll_usd,
                kelly_fraction=self.config.kelly_fraction,
                max_stake_usd=min(
                    self.config.max_stake_per_market_usd,
                    rules.max_stake_usd,
                ),
            )
            sizing_note = (
                f"fractional-Kelly (fraction={self.config.kelly_fraction:.2f}) "
                f"stake on {side}: p_side={p_side:.4f}, m_side={m_side:.4f}, "
                f"bankroll=${current_bankroll.current_bankroll_usd:.2f}"
            )
        elif self.config.sizing == "fixed":
            # Fixed stake = the per-market cap (deterministic for paper trading).
            fixed_stake = min(
                self.config.max_stake_per_market_usd,
                rules.max_stake_usd,
            )
            stake = size_trade_fixed(
                fixed_stake_usd=fixed_stake,
                max_stake_usd=fixed_stake,
            )
            sizing_note = f"fixed stake on {side}: ${stake:.2f}"
        else:  # pragma: no cover - SizingMethod Literal makes this unreachable
            raise RuntimeError(f"Unknown sizing method: {self.config.sizing!r}")

        # ----- Step 4: exposure cap ----------------------------------------
        # If adding `stake` to current open positions exceeds the cap, reduce
        # the stake to the available headroom. If headroom is <= 0, REJECT.
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
                    side=side,  # type: ignore[arg-type]
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
        # Already enforced inside size_trade_kelly via the max_stake_usd
        # argument above. We re-assert here for the fixed-stake path and to
        # be defensive against any future code change.
        cap = min(self.config.max_stake_per_market_usd, rules.max_stake_usd)
        if stake > cap:
            stake = round(cap, 2)

        # ----- Build the rationale & return -------------------------------
        full_kelly = _compute_kelly_fraction(p_side, m_side, fraction=1.0)
        rationale = (
            f"{sizing_note}. "
            f"Full-Kelly fraction would be {full_kelly:.4f} of bankroll; "
            f"applied fraction={self.config.kelly_fraction:.2f} → "
            f"final stake ${stake:.2f} on {side}. "
            f"Edge={edge:+.4f} (consensus {consensus_prob:.4f} vs price {market_price:.4f})."
        )

        return TradePlan(
            market_id=market_id,
            side=side,  # type: ignore[arg-type]
            size_usd=stake,
            limit_price=market_price,  # suggest a limit at the current price
            rationale=rationale,
            risk_flags=flags,
            decision="APPROVE",
            timestamp=now.isoformat(),
        )

    # ------------------------------------------------------------------ state ops

    def update_state(self, trade: TradeReceipt) -> None:
        """Update internal BankrollState after a trade fills.

        Moves cash → open_positions and (re)computes drawdown off the peak.
        Called by pythia-executor after each successful fill.

        VERIFY: meshcfo's `Ledger.commit(trade)` may already do this; if so,
        delegate to it and skip the local mutation.
        """
        size = float(trade.size_usd)
        # Cash decreases by stake; open positions increase by stake.
        self.state.cash_usd = max(0.0, self.state.cash_usd - size)
        self.state.open_positions_usd += size
        # current_bankroll = cash + open_positions (mark-to-market at cost).
        self.state.current_bankroll_usd = (
            self.state.cash_usd + self.state.open_positions_usd
        )
        # Peak tracking — a new high-water mark resets the drawdown denominator.
        if self.state.current_bankroll_usd > self.state.peak_bankroll_usd:
            self.state.peak_bankroll_usd = self.state.current_bankroll_usd
        self._recompute_drawdown()

    def record_loss(self, market_id: str, loss_usd: float) -> None:
        """Record a realised loss on a settled market.

        Decrements ``current_bankroll`` by ``loss_usd``, sets ``last_loss_at``
        to now (which arms the cool-down gate), and recomputes drawdown. The
        peak is NOT moved — only wins can move the peak.

        Parameters
        ----------
        market_id : str
            Settled market id (for audit logging; not used in computation).
        loss_usd : float
            Positive number = dollars lost. Must be >= 0.
        """
        if loss_usd < 0.0:
            raise ValueError(f"loss_usd must be >= 0, got {loss_usd!r}")
        # Open positions decrease (the position is gone), cash unchanged
        # (the loss is realised, not a cash outflow here). Current bankroll
        # drops by the loss amount.
        self.state.open_positions_usd = max(
            0.0, self.state.open_positions_usd - loss_usd
        )
        self.state.current_bankroll_usd = max(
            0.0, self.state.current_bankroll_usd - loss_usd
        )
        self.state.last_loss_at = datetime.now(timezone.utc)
        self._recompute_drawdown()
        # VERIFY: meshcfo may want a hook here, e.g. `Ledger.on_loss(market_id, loss_usd)`.

    def record_win(self, market_id: str, win_usd: float) -> None:
        """Record a realised win on a settled market.

        Adds ``win_usd`` to ``current_bankroll`` and updates the peak if this
        is a new high-water mark. Does not arm the cool-down.

        Parameters
        ----------
        market_id : str
            Settled market id (for audit logging; not used in computation).
        win_usd : float
            Positive number = dollars won (net profit). Must be >= 0.
        """
        if win_usd < 0.0:
            raise ValueError(f"win_usd must be >= 0, got {win_usd!r}")
        # The original stake comes back as cash; win_usd is the net profit
        # on top. For simplicity we add win_usd to current bankroll and let
        # update_state handle the stake movement separately (the executor
        # calls update_state on fill, then record_win/record_loss on settle).
        self.state.current_bankroll_usd += win_usd
        if self.state.current_bankroll_usd > self.state.peak_bankroll_usd:
            self.state.peak_bankroll_usd = self.state.current_bankroll_usd
        self._recompute_drawdown()
        # VERIFY: meshcfo may want a hook here, e.g. `Ledger.on_win(market_id, win_usd)`.

    # ------------------------------------------------------------------ internals

    def _recompute_drawdown(self) -> None:
        """Recompute ``state.drawdown_pct`` off the current peak.

            drawdown_pct = (peak - current) / peak * 100

        Guarded against peak == 0 (returns 0).
        """
        peak = self.state.peak_bankroll_usd
        if peak <= 0.0:
            self.state.drawdown_pct = 0.0
            return
        dd = (peak - self.state.current_bankroll_usd) / peak * 100.0
        self.state.drawdown_pct = max(0.0, dd)
