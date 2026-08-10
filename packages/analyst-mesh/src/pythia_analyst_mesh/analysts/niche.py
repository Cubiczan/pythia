"""Niche / subjective analyst — the most uniquely valuable on Delphi."""
from __future__ import annotations

from ..base import BaseAnalyst
from ..types import ChatMessage, MarketContext


class NicheAnalyst(BaseAnalyst):
    """Specialist for subjective / cultural / niche markets.

    Examples: best-of awards (album, film, paper), viral events, community
    outcomes ("which fork ships first"), cultural moments ("will X meme
    trend again"), niche community predictions.

    This is the analyst where the Pythia mesh earns its edge over a single
    Polymarket-style model: subjective questions have no authoritative
    datasource, so diversity-of-perspective + calibrated uncertainty
    beats raw news-speed.
    """

    analyst_id = "niche"
    specialty = "niche"
    SYSTEM_PROMPT = (
        "You are a cultural critic and trend analyst. You specialize in "
        "subjective questions: best-of awards, cultural moments, viral "
        "events, niche community outcomes. Estimate the probability that "
        "the market's YES outcome occurs. Be calibrated: subjective "
        "questions carry irreducible uncertainty, so default to "
        "confidence < 0.6 unless you have concrete structural evidence. "
        "Surface the consensus-vs-dissent framing explicitly in your "
        "rationale, and prefer recent cultural signals over historical "
        "precedent when regimes shift fast."
    )

    def _build_prompt(self, market: MarketContext) -> list[ChatMessage]:
        user = (
            self._format_market_block(market)
            + "\n\nAdditional instructions for niche / subjective markets:\n"
            '- Identify the most likely "anchor" the market is implicitly '
            "pricing (insider consensus, awards-body precedent, viral "
            "trajectory). State it in your rationale.\n"
            "- For best-of awards, weight award-body historical bias + "
            "narrative momentum; do not anchor on raw popularity metrics.\n"
            "- For viral-event markets, weight current velocity (search "
            "trends, social mention rate) but discount the duration tail.\n"
            "- For niche community outcomes, prefer primary community "
            "sources (forums, governance votes, dev mailing lists) over "
            "mainstream coverage.\n"
            "- When the question is genuinely un-anchored, set confidence "
            "< 0.4 and let consensus-fusion do the averaging.\n"
            + self._format_output_contract()
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
