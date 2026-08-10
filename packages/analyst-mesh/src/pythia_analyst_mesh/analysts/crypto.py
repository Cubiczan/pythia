"""Crypto analyst — token prices, on-chain metrics, DeFi events, upgrades."""
from __future__ import annotations

from ..base import BaseAnalyst
from ..types import ChatMessage, MarketContext


class CryptoAnalyst(BaseAnalyst):
    """Specialist for crypto / DeFi markets.

    Examples: token price thresholds, TVL milestones, protocol upgrade
    activations, airdrop eligibility, hack / exploit resolution.
    """

    analyst_id = "crypto"
    specialty = "crypto"
    SYSTEM_PROMPT = (
        "You are a crypto markets analyst specializing in token prices, "
        "on-chain metrics, DeFi events, and protocol upgrades. Estimate "
        "the probability that the market's YES outcome occurs. Be "
        "calibrated: if uncertain, say so via a confidence < 0.6. Cite "
        "on-chain data (Etherscan, Dune, DefiLlama) and governance forums "
        "wherever possible. Distinguish realized on-chain state from "
        "announcements that may yet be delayed."
    )

    def _build_prompt(self, market: MarketContext) -> list[ChatMessage]:
        user = (
            self._format_market_block(market)
            + "\n\nAdditional instructions for crypto markets:\n"
            "- For price-threshold questions, derive the implied move from "
            "current spot + realized volatility; do not anchor on the YES price.\n"
            "- For protocol-upgrade questions, weight on-chain signals "
            "(timelock scheduled, multi-sig queued, testnet activation) over "
            "blog posts.\n"
            "- For TVL / DeFi milestones, use DefiLlama-style 7-day rolling "
            "averages, not single-day snapshots.\n"
            "- Cite Etherscan / Arbiscan / Dune / DefiLlama / governance "
            "forum URLs as evidence.\n"
            + self._format_output_contract()
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
