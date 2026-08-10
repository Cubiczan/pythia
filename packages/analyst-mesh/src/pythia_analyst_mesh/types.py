"""Pydantic models shared across the analyst mesh.

These are the data contracts between pythia-delphi-adapter (market fetch),
pythia-analyst-mesh (this repo), and pythia-consensus (downstream fusion).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

class Estimate(BaseModel):
    """Output of a single analyst.

    Attributes
    ----------
    market_id:
        Identifier of the Delphi market this estimate applies to.
    probability:
        P(YES) in [0.0, 1.0]. The analyst's belief that the market resolves YES.
    confidence:
        Analyst's own calibration score in [0.0, 1.0]. NOT a probability 
        it is the analyst's self-reported trust in its own estimate.
        Analysts are prompted to default to < 0.6 when uncertain.
    rationale:
        1-3 sentence justification. Required (non-empty).
    evidence:
        List of evidence URLs the analyst cited. May be empty.
    analyst_id:
        Slug identifying which specialist produced this estimate
        (e.g. "politics", "crypto"). Set automatically by ``BaseAnalyst``.
    timestamp:
        ISO 8601 UTC string. Set automatically if not provided.
    """

    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(..., min_length=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1)
    evidence: list[str] = Field(default_factory=list)
    analyst_id: str = Field(..., min_length=1)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    @field_validator("rationale")
    @classmethod
    def _strip_rationale(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("rationale must not be empty / whitespace-only")
        return v

    @field_validator("timestamp")
    @classmethod
    def _ensure_iso(cls, v: str) -> str:
        # Be lenient: if it already parses as ISO, keep it; else stamp now.
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            return datetime.now(UTC).isoformat()

class MarketContext(BaseModel):
    """Input to an analyst.

    Constructed by ``pythia-delphi-adapter`` (raw Delphi read) and enriched
    by ``pythia-strata`` (news, on-chain, social). This is the only thing
    an analyst sees about the market.
    """

    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_yes_price: float | None = Field(default=None, ge=0.0, le=1.0)
    current_no_price: float | None = Field(default=None, ge=0.0, le=1.0)
    volume_usd: float | None = Field(default=None, ge=0.0)
    closes_at: str | None = Field(default=None)

    @field_validator("closes_at")
    @classmethod
    def _validate_iso_or_none(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"closes_at must be ISO 8601 or None, got: {v!r}") from exc
        return v

class LLMConfig(BaseModel):
    """Configuration for a single LLM provider.

    The mesh holds ONE shared ``LLMConfig`` (set in ``live-mvp.toml``) and
    passes it to every analyst via ``AnalystRegistry.build_mesh``. To A/B
    different providers per analyst, instantiate analysts manually with
    different configs.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "anthropic", "gensyn", "ollama"]
    model: str = Field(..., min_length=1)
    api_key: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=1, le=32_000)

    # Provider-specific overrides (optional).
    base_url: str | None = None  # for ollama / gensyn / openai-compatible proxies
    timeout_sec: float = Field(default=30.0, ge=1.0)

# Type alias used in BaseAnalyst._build_prompt — a single chat message.
ChatMessage = dict[str, str]
"""A chat message: ``{"role": "system"|"user"|"assistant", "content": "..."}``."""
