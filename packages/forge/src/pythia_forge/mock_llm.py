"""MockLLM — deterministic LLM replacement for backtests.

The backtest harness needs to run the full mesh → consensus → risk pipeline
thousands of times during a tune sweep. Calling real LLMs for each market
would be:

- **slow** (seconds per analyst per market),
- **expensive** (real token cost), and
- **non-deterministic** (same question → different probability each run,
  which makes A/B comparisons across configs meaningless).

``MockLLM`` solves all three: it maps a market question to a probability
estimate via simple keyword heuristics, deterministically, in microseconds.
It is injected into the mesh by monkey-patching each analyst's ``_call_llm``
method (see ``Backtester._install_mock_llm``) so the *entire* downstream
pipeline (``_parse_llm_response`` → ``Estimate`` → ``fuse`` → ``evaluate``)
runs unmodified.

> ⚠️  **NOT FOR PRODUCTION.**
> MockLLM produces realistic-looking disagreement between analysts so the
> consensus fusion and agreement-score gates are exercised, but its
> "probability estimates" have zero predictive power. Always validate final
> strategy configs with ``--use-real-llm`` against a held-out market set
> before deploying.

Heuristic map
-------------
Keyword(s) in question        | Base P(YES)
------------------------------|-----------
"Trump"                       | 0.55
"Bitcoin", "BTC", "ETH"       | 0.50
"election"                    | 0.50
"Fed", "rate"                 | 0.48
"Super Bowl", "NBA", "NFL"    | 0.52
"AI", "GPT", "LLM"            | 0.60
(default)                     | 0.50

Each specialist analyst then gets a small per-specialty nudge applied to the
base probability (see ``SPECIALTY_NUDGES``), so the four analysts produce
*genuinely different* estimates on the same question — which is what the
consensus fusion logic needs to see to produce meaningful agreement scores.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Keyword → base-probability heuristics.
# ---------------------------------------------------------------------------

# Ordered list of (compiled regex, base probability). First match wins.
# We use regex (case-insensitive) so "Bitcoin" matches "bitcoin" and "BITCOIN".
_KEYWORD_RULES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\btrump\b", re.IGNORECASE), 0.55),
    (re.compile(r"\bbitcoin\b|\bBTC\b|\bETH\b|\bethereum\b", re.IGNORECASE), 0.50),
    (re.compile(r"\belection\b", re.IGNORECASE), 0.50),
    (re.compile(r"\bfed\b|\brate\s+cut\b|\binterest\s+rate\b", re.IGNORECASE), 0.48),
    (re.compile(r"\bsuper\s+bowl\b|\bNBA\b|\bNFL\b|\bworld\s+cup\b", re.IGNORECASE), 0.52),
    (re.compile(r"\bAI\b|\bGPT\b|\bLLM\b|\bartificial\s+intelligence\b", re.IGNORECASE), 0.60),
]

# Per-specialty nudge added to the base probability (clamped to [0.05, 0.95]).
# These are tiny on purpose — just enough to produce realistic disagreement
# without making any single analyst systematically right or wrong.
SPECIALTY_NUDGES: dict[str, float] = {
    "politics": 0.03,   # leans slightly bullish on political markets
    "crypto": 0.02,     # leans slightly bullish on crypto markets
    "sports": -0.02,    # leans slightly bearish (upsets happen)
    "niche": 0.01,      # near-neutral on subjective markets
}

_DEFAULT_PROBABILITY: float = 0.50
_CONFIDENCE: float = 0.60  # fixed moderate confidence — exercises the agreement gate


class MockLLM:
    """Deterministic, keyword-based LLM replacement for backtests.

    Usage::

        mock = MockLLM()
        prob = mock.estimate_probability(
            question="Will Bitcoin close above $100k?",
            analyst_id="crypto",
        )
        # → 0.52 (0.50 base + 0.02 crypto nudge)

    Or via the LLM-call-shaped interface (used by ``Backtester._install_mock_llm``)::

        response_text = mock.respond(messages=[{"role": "user", "content": "..."}],
                                     analyst_id="crypto")
        # → '{"probability": 0.52, "confidence": 0.6, "rationale": "...", "evidence": []}'

    The returned string mimics what a real LLM would return (JSON envelope),
    so the existing ``BaseAnalyst._parse_llm_response`` pipeline produces a
    valid ``Estimate`` without modification.
    """

    def __init__(
        self,
        *,
        keyword_rules: list[tuple[re.Pattern[str], float]] | None = None,
        specialty_nudges: dict[str, float] | None = None,
        default_probability: float = _DEFAULT_PROBABILITY,
        confidence: float = _CONFIDENCE,
    ) -> None:
        self._rules = keyword_rules if keyword_rules is not None else _KEYWORD_RULES
        self._nudges = specialty_nudges if specialty_nudges is not None else SPECIALTY_NUDGES
        self._default_prob = default_probability
        self._confidence = confidence

    # ------------------------------------------------------------------ #
    # Core heuristic
    # ------------------------------------------------------------------ #

    def estimate_probability(self, question: str, analyst_id: str = "") -> float:
        """Map a question + analyst to a deterministic P(YES) estimate.

        Steps:
        1. Find the first matching keyword rule → base probability.
           If none match, use the default (0.50).
        2. Apply the per-specialty nudge for ``analyst_id`` (if known).
        3. Clamp to [0.05, 0.95] so the logit-space fusion never sees a 0 or 1
           (which would produce infinite logits).

        Deterministic: the same (question, analyst_id) always returns the
        same probability. This is critical for reproducible backtests.
        """
        base = self._default_prob
        for pattern, prob in self._rules:
            if pattern.search(question):
                base = prob
                break

        nudge = self._nudges.get(analyst_id, 0.0)
        prob = base + nudge
        # Clamp away from 0 and 1 so logit-mean fusion is well-defined.
        return max(0.05, min(0.95, prob))

    # ------------------------------------------------------------------ #
    # LLM-call-shaped interface
    # ------------------------------------------------------------------ #

    def respond(
        self,
        messages: list[dict[str, str]],
        analyst_id: str = "",
    ) -> str:
        """Return a JSON string mimicking a real LLM's chat-completions response.

        The mesh's ``BaseAnalyst._call_llm`` returns the assistant's text
        content (a ``str``); ``_parse_llm_response`` then extracts the JSON
        envelope. This method produces exactly that shape so the full
        downstream pipeline runs unmodified.

        Parameters
        ----------
        messages:
            The chat-message list (``[{"role": "system", ...}, {"role": "user", ...}]``).
            Only the user message content is inspected — we extract the market
            question from it via a simple regex.
        analyst_id:
            The specialist analyst's slug (e.g. ``"politics"``). Used to apply
            the per-specialty nudge.

        Returns
        -------
        str
            A JSON string: ``{"probability": <float>, "confidence": <float>,
            "rationale": <str>, "evidence": []}``.
        """
        question = self._extract_question(messages)
        prob = self.estimate_probability(question, analyst_id)
        payload: dict[str, Any] = {
            "probability": round(prob, 4),
            "confidence": self._confidence,
            "rationale": (
                f"[MOCK LLM] Deterministic keyword-based estimate: P(YES)={prob:.2f}. "
                f"This is NOT a real LLM response — replace with --use-real-llm "
                f"for production validation."
            ),
            "evidence": [],
        }
        return json.dumps(payload)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_question(messages: list[dict[str, str]]) -> str:
        """Pull the market question out of the chat-message list.

        The mesh's ``BaseAnalyst._format_market_block`` renders the question
        as the first line of the user message: ``"Market question: <q>"``.
        We grab everything after that prefix. If we can't find it, fall back
        to the whole user message content (the keyword regexes will still
        match against it).
        """
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Look for "Market question: <...>" — capture up to newline.
                m = re.search(r"Market question:\s*(.+?)(?:\n|$)", content)
                if m:
                    return m.group(1).strip()
                return content
        return ""


__all__ = ["MockLLM", "SPECIALTY_NUDGES"]
