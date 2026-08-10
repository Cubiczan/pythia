"""Base class for all Pythia analyst agents.

Subclasses implement ``_build_prompt`` (and may override ``estimate``).
The base class provides:

- ``_call_llm`` — provider-agnostic dispatch (openai | anthropic | gensyn | ollama)
                  wrapped with ``tenacity`` retries.
- ``_parse_llm_response`` — robust JSON extraction with graceful fallbacks.
- shared state: ``analyst_id``, ``specialty``, ``_llm_config``.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .types import ChatMessage, Estimate, LLMConfig, MarketContext

logger = logging.getLogger(__name__)

class LLMCallError(RuntimeError):
    """Raised when every retry attempt on an LLM call has failed."""

class BaseAnalyst(ABC):
    """Abstract base for specialist analyst agents.

    Subclasses MUST set:
        ``analyst_id``  — short slug, e.g. ``"politics"``
        ``specialty``   — human-readable category
        ``SYSTEM_PROMPT`` — 2-4 sentence string with calibration guidance

    Subclasses MUST implement:
        ``_build_prompt(market) -> list[ChatMessage]``

    Subclasses MAY override:
        ``estimate(market) -> Estimate``  (default impl below is usually fine)
    """

    # Class-level attributes; subclasses override.
    analyst_id: str = ""
    specialty: str = ""
    SYSTEM_PROMPT: str = ""

    def __init__(self, llm_config: LLMConfig) -> None:
        if not self.analyst_id:
            raise ValueError(
                f"{type(self).__name__} must set class attribute `analyst_id`."
            )
        if not self.specialty:
            raise ValueError(
                f"{type(self).__name__} must set class attribute `specialty`."
            )
        self._llm_config = llm_config

    # ------------------------------------------------------------------ #
    # Abstract API
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _build_prompt(self, market: MarketContext) -> list[ChatMessage]:
        """Build the chat-message list to send to the LLM.

        Returns
        -------
        list[ChatMessage]
            Usually ``[{"role": "system", ...}, {"role": "user", ...}]``.
            The system message typically uses ``self.SYSTEM_PROMPT``.
        """
        ...

    # ------------------------------------------------------------------ #
    # Shared prompt-building helpers (used by all 4 built-in analysts)
    # ------------------------------------------------------------------ #

    def _format_market_block(self, market: MarketContext) -> str:
        """Render the market as a structured user-prompt block.

        Explicitly notes that the current YES price is the market's implied
        probability, so the LLM can decide whether to agree or disagree.
        """
        lines: list[str] = [
            f"Market question: {market.question}",
            f"Category: {market.category}",
        ]
        if market.current_yes_price is not None:
            lines.append(
                f"Market's current implied P(YES) (YES price): "
                f"{market.current_yes_price:.3f}  "
                f"[Note: this is the market's current probability estimate. "
                f"Decide whether you agree or disagree.]"
            )
        if market.current_no_price is not None:
            lines.append(f"Market's current NO price: {market.current_no_price:.3f}")
        if market.volume_usd is not None:
            lines.append(f"Volume (USD): {market.volume_usd:,.2f}")
        if market.closes_at:
            lines.append(f"Resolves / closes at: {market.closes_at}")
        # News / on-chain / social context passed via metadata.
        news = market.metadata.get("news_context") or market.metadata.get("news")
        if news:
            lines.append(f"Recent context: {news}")
        # Allow other metadata keys to surface as "Extra context:".
        extra = {
            k: v
            for k, v in market.metadata.items()
            if k not in {"news_context", "news"} and isinstance(v, (str, int, float, bool))
        }
        if extra:
            lines.append("Extra context: " + ", ".join(f"{k}={v}" for k, v in extra.items()))
        return "\n".join(lines)

    def _format_output_contract(self) -> str:
        """The strict output contract appended to every user prompt."""
        return (
            "\n\nRespond with ONLY a JSON object of this shape (no prose, "
            "no markdown fences):\n"
            "{\n"
            '  "probability": <float in [0.0, 1.0] = P(YES)>,\n'
            '  "confidence": <float in [0.0, 1.0] = your calibration>,\n'
            '  "rationale": "<1-3 sentences>",\n'
            '  "evidence": ["<url1>", "<url2>"]\n'
            "}\n"
            "If you are uncertain, set confidence < 0.6. "
            "If you have no evidence URLs, return an empty list."
        )

    async def estimate(self, market: MarketContext) -> Estimate:
        """Default estimate pipeline: build prompt → call LLM → parse.

        Override if you need custom behaviour (e.g. tool use, multi-turn).
        """
        messages = self._build_prompt(market)
        raw = await self._call_llm(messages, self._llm_config)
        return self._parse_llm_response(raw, market)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException, LLMCallError)
        ),
        reraise=True,
    )
    async def _call_llm(self, messages: list[ChatMessage], config: LLMConfig) -> str:
        """Provider-agnostic LLM dispatch.

        Dispatches on ``config.provider``:

        - ``"openai"``     → openai>=1.0 async client
        - ``"anthropic"``  → anthropic>=0.20 async client
        - ``"gensyn"``     → raw httpx POST  (# VERIFY: Gensyn REST shape pending docs)
        - ``"ollama"``     → httpx POST to local /api/chat

        Returns the assistant's text content (str). Raises ``LLMCallError``
        if the provider returns a non-2xx after all retries.
        """
        provider = config.provider
        if provider == "openai":
            return await self._call_openai(messages, config)
        if provider == "anthropic":
            return await self._call_anthropic(messages, config)
        if provider == "gensyn":
            return await self._call_gensyn(messages, config)
        if provider == "ollama":
            return await self._call_ollama(messages, config)
        raise LLMCallError(f"Unknown LLM provider: {provider!r}")

    async def _call_openai(self, messages: list[ChatMessage], config: LLMConfig) -> str:
        # Lazy import so the mesh loads even if openai SDK isn't installed.
        from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

        # VERIFY: openai>=1.0 SDK signature — api_key may also be set via env.
        client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.timeout_sec,
        )
        try:
            resp = await client.chat.completions.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except (APIError, APIConnectionError, RateLimitError) as exc:
            raise LLMCallError(f"openai call failed: {exc}") from exc
        # resp.choices[0].message.content is Optional[str] in the SDK.
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            raise LLMCallError("openai returned empty content")
        return content

    async def _call_anthropic(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> str:
        from anthropic import APIConnectionError, APIError, AsyncAnthropic

        # VERIFY: anthropic>=0.20 async client. messages.create signature.
        client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.timeout_sec,
        )
        # Anthropic requires a separate top-level `system` param.
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] != "system"]
        system_text = "\n\n".join(m["content"] for m in system_msgs)
        try:
            resp = await client.messages.create(
                model=config.model,
                system=system_text,
                messages=user_msgs,  # type: ignore[arg-type]
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except (APIError, APIConnectionError) as exc:
            raise LLMCallError(f"anthropic call failed: {exc}") from exc
        # Anthropic returns a list of content blocks; concatenate text ones.
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        if not parts:
            raise LLMCallError("anthropic returned empty content")
        return "".join(parts)

    async def _call_gensyn(self, messages: list[ChatMessage], config: LLMConfig) -> str:
        """Raw httpx call to a Gensyn-hosted inference endpoint.

        # VERIFY: Gensyn's public REST/HTTP shape is not yet documented.
        # The shape below is a reasonable OpenAI-compatible guess; adjust
        # when Gensyn publishes their chat-completions spec.
        """
        base = config.base_url or "https://infer.gensyn.ai/v1"
        url = f"{base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        async with httpx.AsyncClient(timeout=config.timeout_sec) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise LLMCallError(f"gensyn transport error: {exc}") from exc
            if resp.status_code >= 400:
                raise LLMCallError(
                    f"gensyn returned {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
        # OpenAI-compatible shape:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMCallError(f"gensyn response missing content: {data!r}") from exc

    async def _call_ollama(self, messages: list[ChatMessage], config: LLMConfig) -> str:
        """Local Ollama via /api/chat (no API key)."""
        base = config.base_url or "http://localhost:11434"
        url = f"{base.rstrip('/')}/api/chat"
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=config.timeout_sec) as client:
            try:
                resp = await client.post(url, json=payload)
            except httpx.HTTPError as exc:
                raise LLMCallError(f"ollama transport error: {exc}") from exc
            if resp.status_code >= 400:
                raise LLMCallError(
                    f"ollama returned {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMCallError(f"ollama response missing content: {data!r}") from exc

    # ------------------------------------------------------------------ #
    # Response parsing — robust, never raises
    # ------------------------------------------------------------------ #

    # Match the first {...} block (greedy enough for typical LLM output).
    _JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
    # Match a probability in one of these forms:
    #   - decimal in [0,1]: 0.42, .42, 1.0, 1, 0
    #   - integer percentage: 42%, 100%, 5%
    _PROB_RE = re.compile(
        r"(?<![A-Za-z0-9.])"
        r"(?P<dec>0?\.\d+|\.\d+|1\.0+|1|0)"          # decimal form
        r"(?![\d.])"
        r"|"
        r"(?P<pct>\d{1,3})\s*%"                        # percentage form
    )

    def _parse_llm_response(self, raw: str, market: MarketContext) -> Estimate:
        """Parse an LLM response into an ``Estimate`` — never raises.

        Strategy (in order):
        1. Strip Markdown code fences, try ``json.loads`` on the whole string.
        2. Find the first ``{...}`` block, try ``json.loads`` on it.
        3. Fall back to scanning for a leading probability number; emit a
           low-confidence Estimate from that.
        4. Last resort: confidence 0.0, probability 0.5 (uninformative prior),
           rationale = the raw text truncated.
        """
        text = (raw or "").strip()
        parsed: dict[str, Any] | None = None

        # 1. Strip code fences and try whole-string parse.
        cleaned = self._strip_code_fences(text)
        parsed = self._try_json(cleaned)

        # 2. Extract first {...} block.
        if parsed is None:
            m = self._JSON_BLOCK_RE.search(cleaned)
            if m:
                parsed = self._try_json(m.group(0))

        # 3. Probability-number fallback.
        if parsed is None:
            prob = self._scan_probability(cleaned)
            if prob is not None:
                return Estimate(
                    market_id=market.market_id,
                    probability=prob,
                    confidence=0.3,  # low — we couldn't parse the full envelope
                    rationale=cleaned[:300] or "LLM returned unparsable output.",
                    evidence=[],
                    analyst_id=self.analyst_id,
                    timestamp=datetime.now(UTC).isoformat(),
                )

        # 4. Last-resort fallback.
        if parsed is None:
            logger.warning(
                "analyst=%s market=%s could not parse LLM response; "
                "returning uninformative prior",
                self.analyst_id,
                market.market_id,
            )
            return Estimate(
                market_id=market.market_id,
                probability=0.5,
                confidence=0.0,
                rationale=(
                    "LLM response could not be parsed; "
                    "returning uninformative prior P(YES)=0.5."
                ),
                evidence=[],
                analyst_id=self.analyst_id,
                timestamp=datetime.now(UTC).isoformat(),
            )

        # Build an Estimate from the parsed dict, tolerating missing / extra keys.
        return self._build_estimate_from_dict(parsed, market, cleaned)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```json ... ``` / ``` ... ``` fences if present."""
        if "```" not in text:
            return text
        # Drop the opening fence (with optional language tag).
        opened = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text, count=1)
        # Drop the closing fence.
        return re.sub(r"\n?```\s*$", "", opened).strip()

    @staticmethod
    def _try_json(s: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            return None
        if isinstance(obj, dict):
            return obj
        return None

    @staticmethod
    def _scan_probability(text: str) -> float | None:
        """Find the first probability-looking number in the text.

        Recognises:
        - decimals in [0, 1]: ``0.42``, ``.42``, ``1.0``, ``0``, ``1``
        - integer percentages: ``42%``, ``100%``, ``5%``
        """
        for m in BaseAnalyst._PROB_RE.finditer(text):
            dec = m.group("dec")
            pct = m.group("pct")
            try:
                if dec is not None:
                    val = float(dec)
                    # Reject things like "42" without % (would be misread as prob).
                    # The regex only matches bare 0 or 1 as integers, so this is safe.
                    if 0.0 <= val <= 1.0:
                        return val
                if pct is not None:
                    val = float(pct) / 100.0
                    if 0.0 <= val <= 1.0:
                        return val
            except (ValueError, TypeError):
                continue
        return None

    def _build_estimate_from_dict(
        self,
        d: dict[str, Any],
        market: MarketContext,
        raw_text: str,
    ) -> Estimate:
        """Coerce a parsed dict into an ``Estimate``, clamping bad values."""
        # Probability — try several common keys, clamp to [0,1].
        prob = self._extract_float(
            d,
            keys=("probability", "prob", "p_yes", "p", "P(YES)", "p_yes_est"),
            default=0.5,
            lo=0.0,
            hi=1.0,
        )
        # Confidence — try several common keys, clamp to [0,1].
        conf = self._extract_float(
            d,
            keys=("confidence", "conf", "calibration", "self_confidence"),
            default=0.3,
            lo=0.0,
            hi=1.0,
        )
        # Rationale — must be non-empty.
        rationale = (
            str(d.get("rationale") or d.get("reasoning") or d.get("explanation") or "")
            .strip()
            or raw_text[:300]
            or "No rationale provided."
        )
        # Evidence — accept list of strings or comma-separated string.
        evidence = self._extract_evidence(d.get("evidence") or d.get("urls") or d.get("sources"))
        # Analyst id — prefer dict value, fall back to self.
        analyst_id = str(d.get("analyst_id") or self.analyst_id)
        # Timestamp — prefer dict value, fall back to now.
        timestamp = str(d.get("timestamp") or datetime.now(UTC).isoformat())
        return Estimate(
            market_id=market.market_id,
            probability=prob,
            confidence=conf,
            rationale=rationale,
            evidence=evidence,
            analyst_id=analyst_id,
            timestamp=timestamp,
        )

    @staticmethod
    def _extract_float(
        d: dict[str, Any], keys: tuple[str, ...], default: float, lo: float, hi: float
    ) -> float:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    v = float(d[k])  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                return max(lo, min(hi, v))
        return default

    @staticmethod
    def _extract_evidence(raw: Any) -> list[str]:
        if isinstance(raw, list):
            return [str(x) for x in raw if x]
        if isinstance(raw, str):
            return [s.strip() for s in raw.split(",") if s.strip()]
        return []
