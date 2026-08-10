"""Sports analyst — match outcomes, player stats, injuries, schedules."""
from __future__ import annotations

from ..base import BaseAnalyst
from ..types import ChatMessage, MarketContext

class SportsAnalyst(BaseAnalyst):
    """Specialist for sports / esports markets.

    Examples: match winners, point spreads, player-stat thresholds
    (yards, goals, kills), playoff qualification, injury-driven props.
    """

    analyst_id = "sports"
    specialty = "sports"
    SYSTEM_PROMPT = (
        "You are a sports analyst specializing in match outcomes, player "
        "stats, injuries, and schedule factors. Estimate the probability "
        "that the market's YES outcome occurs. Be calibrated: if uncertain, "
        "say so via a confidence < 0.6. Prefer model-based ratings (Elo, "
        "Sagarin, FiveThirtyEight-style) and recent-form windows over "
        "narrative; flag injuries as a primary uncertainty source."
    )

    def _build_prompt(self, market: MarketContext) -> list[ChatMessage]:
        user = (
            self._format_market_block(market)
            + "\n\nAdditional instructions for sports markets:\n"
            "- For match outcomes, use team-rating systems (Elo / Sagarin / "
            "DVOA) and home-field adjustments; convert ratings to win "
            "probability rather than copying the YES price.\n"
            "- For player-stat props, weight recent 5–10 game rolling "
            "averages + opponent defensive ratings; flag any injury / "
            "rest-day designations.\n"
            "- For season-long markets (playoffs, awards), weight current "
            "standings + remaining strength of schedule.\n"
            "- Cite ESPN / official league injury reports / Basketball-"
            "Reference / FBref URLs as evidence.\n"
            + self._format_output_contract()
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
