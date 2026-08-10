"""pythia-analyst-mesh

Specialist LLM analyst agents for the Pythia Delphi trading mesh.

Public API:

    BaseAnalyst          — ABC for all specialist analysts
    Estimate             — output of an analyst (prob + confidence + rationale + evidence)
    MarketContext        — input to an analyst (market question + metadata + price)
    LLMConfig            — swappable provider config (openai | anthropic | gensyn | ollama)
    AnalystRegistry      — register / look up / build meshes of analysts
    PoliticsAnalyst      — elections, legislation, geopolitics
    CryptoAnalyst        — token prices, on-chain metrics, DeFi events
    SportsAnalyst        — match outcomes, player stats, injuries
    NicheAnalyst         — subjective / cultural / viral-event markets

Usage::

    from pythia_analyst_mesh import (
        AnalystRegistry, LLMConfig, MarketContext, run_mesh,
    )
    import asyncio

    llm = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-...")
    mesh = AnalystRegistry().build_mesh(["politics", "crypto"], llm)
    estimates = asyncio.run(run_mesh(market, mesh, timeout_sec=30.0))
"""
from __future__ import annotations

from .analysts import (
    CryptoAnalyst,
    NicheAnalyst,
    PoliticsAnalyst,
    SportsAnalyst,
)
from .base import BaseAnalyst
from .registry import AnalystRegistry
from .runner import run_mesh
from .types import ChatMessage, Estimate, LLMConfig, MarketContext

__all__ = [
    "AnalystRegistry",
    "BaseAnalyst",
    "ChatMessage",
    "CryptoAnalyst",
    "Estimate",
    "LLMConfig",
    "MarketContext",
    "NicheAnalyst",
    "PoliticsAnalyst",
    "SportsAnalyst",
    "run_mesh",
]

__version__ = "0.1.0"
