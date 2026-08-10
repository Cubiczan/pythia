"""Politics analyst — elections, legislation, geopolitical events."""
from __future__ import annotations

from ..base import BaseAnalyst
from ..types import ChatMessage, MarketContext

class PoliticsAnalyst(BaseAnalyst):
    """Specialist for political / geopolitical markets.

    Examples: election outcomes, legislative passage, treaty ratification,
    cabinet confirmations, executive-order effects.
    """

    analyst_id = "politics"
    specialty = "politics"
    SYSTEM_PROMPT = (
        "You are a political analyst specializing in elections, legislation, "
        "and geopolitical events. Estimate the probability that the market's "
        "YES outcome occurs. Be calibrated: if uncertain, say so via a "
        "confidence < 0.6. Cite primary sources (polling aggregators, "
        "government gazettes, official statements) wherever possible, and "
        "prefer recent polling medians over single surveys."
    )

    def _build_prompt(self, market: MarketContext) -> list[ChatMessage]:
        user = (
            self._format_market_block(market)
            + "\n\nAdditional instructions for political markets:\n"
            "- Weight polling aggregators (e.g. RealClearPolls averages, "
            "FiveThirtyEight-style blends) above single polls.\n"
            "- Account for the difference between polled vote share and "
            "Electoral-College / district-level outcomes.\n"
            "- For legislation, weight committee passage + whip counts; "
            "discount floor statements.\n"
            "- For geopolitics, distinguish official government statements "
            "from analyst commentary.\n"
            + self._format_output_contract()
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
