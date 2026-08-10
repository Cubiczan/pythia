"""PythiaExecutor: the 12-step mesh → consensus → risk → ATT pipeline.

This is the orchestrator that ties the four sibling Pythia sub-repos
together. For each Delphi market it:

1.  fetches the market via ``DelphiClient.get_market``,
2.  builds a ``MarketContext`` for the analyst mesh,
3.  runs the mesh concurrently,
4.  skips if fewer than ``consensus.min_analysts`` estimates came back,
5.  fuses estimates via ``ConsensusEngine.decide``,
6.  skips if the gate is anything other than ``"trade"``,
7.  sizes the trade via ``RiskEngine.evaluate``,
8.  skips if risk returns ``REJECT``,
9.  in paper mode: synthesises a ``TradeReceipt`` with ``status="PAPER"``,
10. in live mode: signs the order and submits via ``DelphiClient.place_order``,
11. appends the full ``PipelineResult`` to the audit log as signed JSONL,
12. updates ``risk_engine.state`` via ``update_state(receipt)``.

Anywhere the exact ATT signing scheme is uncertain, the code is annotated
with ``# VERIFY:``. The current implementation prefers Ed25519 (via the
``cryptography`` library) and falls back to HMAC-SHA256 when the library
isn't installed or the env var doesn't decode as a 32-byte Ed25519 key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pythia_analyst_mesh import MarketContext, run_mesh
from pythia_consensus import ConsensusEngine
from pythia_delphi_adapter import Market as AdapterMarket
from pythia_delphi_adapter import OrderSide, TradeReceipt
from pythia_risk import RiskEngine, TradePlan
from pythia_risk import TradeReceipt as RiskTradeReceipt

from .types import ExecutorConfig, PipelineResult

if TYPE_CHECKING:
    from pythia_analyst_mesh import BaseAnalyst
    from pythia_delphi_adapter import DelphiClient

logger = logging.getLogger(__name__)


class PythiaExecutor:
    """The trade-orchestration pipeline.

    Parameters
    ----------
    delphi_client:
        Async client for the Gensyn Delphi ATT. Must implement
        ``get_market(market_id) -> Market`` and ``place_order(...) -> TradeReceipt``.
    mesh:
        List of instantiated ``BaseAnalyst`` objects (politics / crypto / ...).
    consensus_engine:
        ``ConsensusEngine`` from pythia-consensus. ``.config.min_analysts`` is
        read to decide the early-skip on insufficient estimates.
    risk_engine:
        ``RiskEngine`` from pythia-risk. ``.state`` is read for the current
        bankroll and mutated via ``update_state`` after each fill.
    config:
        ``ExecutorConfig`` (mode, signing key env, retry settings).
    audit_log_path:
        ``Path`` to the JSONL audit log. Created if it doesn't exist;
        appended to on every ``run_for_market`` call.
    """

    def __init__(
        self,
        delphi_client: DelphiClient,
        mesh: list[BaseAnalyst],
        consensus_engine: ConsensusEngine,
        risk_engine: RiskEngine,
        config: ExecutorConfig,
        audit_log_path: Path,
    ) -> None:
        self.delphi_client = delphi_client
        self.mesh = list(mesh)
        self.consensus_engine = consensus_engine
        self.risk_engine = risk_engine
        self.config = config
        self.audit_log_path = Path(audit_log_path)

        # Lazily resolved signing key — see _load_signing_key.
        self._signing_key: bytes | None = None
        self._signing_key_loaded = False

        # Ensure the audit log's parent directory exists.
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ API

    async def run_for_market(self, market_id: str) -> PipelineResult:
        """Run the full 12-step pipeline for one market.

        Returns a ``PipelineResult`` regardless of whether the pipeline
        reached submission or skipped early. The ``skipped_reason`` field
        is ``None`` iff we submitted (paper or live).
        """
        timestamp = datetime.now(UTC).isoformat()

        # ----- Step 1: fetch market ----------------------------------------
        market: AdapterMarket = await self.delphi_client.get_market(market_id)

        # ----- Step 2: build MarketContext for the mesh --------------------
        market_ctx = self._build_market_context(market)

        # ----- Step 3: run the analyst mesh --------------------------------
        estimates = await run_mesh(market_ctx, self.mesh, timeout_sec=30.0)

        # ----- Step 4: insufficient analysts? ------------------------------
        min_analysts = self.consensus_engine.config.min_analysts
        if len(estimates) < min_analysts:
            result = PipelineResult(
                market_id=market_id,
                estimates=estimates,
                decision=None,
                plan=None,
                receipt=None,
                skipped_reason="insufficient_analysts",
                timestamp=timestamp,
            )
            self._write_audit(result)
            logger.info(
                "market=%s skipped: insufficient_analysts (got %d, need %d)",
                market_id, len(estimates), min_analysts,
            )
            return result

        # ----- Step 5: fuse via consensus ----------------------------------
        decision = self.consensus_engine.decide(estimates)

        # ----- Step 6: gate check ------------------------------------------
        if decision.gate != "trade":
            result = PipelineResult(
                market_id=market_id,
                estimates=estimates,
                decision=decision,
                plan=None,
                receipt=None,
                skipped_reason=f"gate_{decision.gate}",
                timestamp=timestamp,
            )
            self._write_audit(result)
            logger.info(
                "market=%s skipped: gate_%s (agreement=%.3f, threshold=%.3f)",
                market_id, decision.gate, decision.agreement_score,
                self.consensus_engine.config.agreement_threshold,
            )
            return result

        # ----- Step 7: risk sizing -----------------------------------------
        risk_market = self._adapt_market_for_risk(market)
        plan = self.risk_engine.evaluate(decision, risk_market, self.risk_engine.state)

        # ----- Step 8: risk rejected? --------------------------------------
        if plan.decision != "APPROVE":
            flags_str = ",".join(plan.risk_flags) if plan.risk_flags else "no_flags"
            result = PipelineResult(
                market_id=market_id,
                estimates=estimates,
                decision=decision,
                plan=plan,
                receipt=None,
                skipped_reason=f"risk_rejected:{flags_str}",
                timestamp=timestamp,
            )
            self._write_audit(result)
            logger.info(
                "market=%s skipped: risk_rejected (%s)", market_id, flags_str
            )
            return result

        # ----- Step 9 / 10: submit (paper or live) -------------------------
        if self.config.mode == "paper":
            receipt = self._build_paper_receipt(plan, market)
            logger.info(
                "market=%s PAPER trade: side=%s size=$%.2f",
                market_id, plan.side, plan.size_usd,
            )
        else:
            # Live mode: sign + submit.
            signature = self._sign_order(plan)
            receipt = await self._submit_live_order(plan, signature)
            logger.info(
                "market=%s LIVE trade: side=%s size=$%.2f order_id=%s",
                market_id, plan.side, plan.size_usd, receipt.att_order_id,
            )

        # ----- Step 11: audit log ------------------------------------------
        result = PipelineResult(
            market_id=market_id,
            estimates=estimates,
            decision=decision,
            plan=plan,
            receipt=receipt,
            skipped_reason=None,
            timestamp=timestamp,
        )
        self._write_audit(result)

        # ----- Step 12: update risk state ----------------------------------
        risk_receipt = self._adapt_receipt_for_risk(receipt)
        self.risk_engine.update_state(risk_receipt)

        return result

    # ------------------------------------------------------------------ sign

    def _sign_order(self, plan: TradePlan) -> str:
        """Sign a TradePlan for ATT submission.

        # VERIFY: The exact ATT signing scheme (header name, payload format,
        # key encoding) is not yet documented upstream. The current
        # implementation:
        #
        #   - If the ``cryptography`` package is installed AND the env var
        #     named by ``config.signing_key_env`` decodes as a 32-byte
        #     Ed25519 private key (base64-url), sign the canonical JSON of
        #     the order body with Ed25519 and return base64-url signature.
        #   - Otherwise, fall back to HMAC-SHA256 over the same canonical
        #     JSON, keyed by the env var value (or a deterministic stub key
        #     if the env var is empty).
        #
        # Once ATT publishes the canonical scheme, swap the body of this
        # method; the public signature (``str`` in, ``str`` out) stays.
        """
        key_bytes = self._load_signing_key()
        message = self._canonical_order_bytes(plan)

        if key_bytes is not None and len(key_bytes) == 32:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PrivateKey,
                )

                priv = Ed25519PrivateKey.from_private_bytes(key_bytes)
                sig = priv.sign(message)
                return base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
            except Exception as exc:
                logger.warning(
                    "Ed25519 signing failed (%s); falling back to HMAC-SHA256", exc
                )

        # HMAC-SHA256 fallback.
        secret = key_bytes or b"pythia-executor-stub-key"
        digest = hmac.new(secret, message, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def _signing_key_fingerprint(self) -> str:
        """Return a short fingerprint of the signing key (never the key)."""
        key_bytes = self._load_signing_key()
        if key_bytes is None:
            return "stub-key"
        digest = hashlib.sha256(key_bytes).hexdigest()
        return digest[:16]

    # ------------------------------------------------------------------ audit

    def _write_audit(self, result: PipelineResult) -> None:
        """Append a signed JSONL line to ``audit_log_path``.

        The line is the JSON of the ``PipelineResult`` plus a ``signature``
        field (HMAC-SHA256 or Ed25519 over the canonical JSON) and a
        ``signature_algo`` field. The audit log is append-only; if the file
        doesn't exist it's created (parent dir is ensured in ``__init__``).
        """
        payload = self._result_to_jsonable(result)
        # Compute signature over canonical (sorted-keys) JSON.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        signature = self._sign_bytes(canonical.encode("utf-8"))
        payload["signature"] = signature
        payload["signature_algo"] = "ed25519" if self._has_ed25519_key() else "hmac-sha256"
        payload["signed_by"] = self._signing_key_fingerprint()
        line = json.dumps(payload, default=str)
        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------ helpers

    def _build_market_context(self, market: AdapterMarket) -> MarketContext:
        """Convert adapter ``Market`` → mesh ``MarketContext``."""
        category_str = (
            market.category.value if hasattr(market.category, "value")
            else str(market.category)
        )
        closes_at_str = (
            market.closes_at.isoformat() if market.closes_at is not None else None
        )
        return MarketContext(
            market_id=market.market_id,
            question=market.question,
            category=category_str,
            metadata={
                "arbiter_model": market.arbiter_model,
                "liquidity_usd": market.liquidity_usd,
                "status": market.status.value if hasattr(market.status, "value") else str(market.status),
            },
            current_yes_price=market.yes_price,
            current_no_price=market.no_price,
            volume_usd=market.volume_usd,
            closes_at=closes_at_str,
        )

    def _adapt_market_for_risk(self, market: AdapterMarket) -> Any:
        """Convert adapter ``Market`` → risk ``pythia_risk.Market``.

        The risk engine consumes its own local ``Market`` type (a thinner
        mirror of the adapter's). We adapt here so the executor is the only
        place that needs to know about the type mismatch.
        """
        from pythia_risk import Market as RiskMarket

        category_str = (
            market.category.value if hasattr(market.category, "value")
            else str(market.category)
        ).lower()
        return RiskMarket(
            market_id=market.market_id,
            yes_price=market.yes_price,
            category=category_str,
            question=market.question,
            close_date=market.closes_at,
        )

    def _build_paper_receipt(
        self, plan: TradePlan, market: AdapterMarket
    ) -> TradeReceipt:
        """Synthesise a paper-mode ``TradeReceipt`` (no submission).

        ``status`` is set to the string ``"PAPER"`` (not a valid
        ``OrderStatus`` enum value) via ``model_construct`` to bypass
        validation. The audit-log serialiser handles this gracefully.
        """
        return TradeReceipt.model_construct(
            market_id=plan.market_id,
            side=OrderSide(plan.side),
            size_usd=plan.size_usd,
            fill_price=plan.limit_price if plan.limit_price is not None else market.yes_price,
            att_order_id=f"paper-{uuid.uuid4()}",
            status="PAPER",  # type: ignore[arg-type]  # bypasses OrderStatus enum
            signed_by="paper-mode",
            timestamp=datetime.now(UTC),
        )

    async def _submit_live_order(
        self, plan: TradePlan, signature: str
    ) -> TradeReceipt:
        """Sign + submit a live order via ``DelphiClient.place_order``.

        # VERIFY: how the signature is transmitted to ATT. The current
        # adapter ``place_order`` does not accept a signature parameter —
        # we attach it to the receipt as an extra field for the audit log.
        # Once the adapter exposes a ``signature=`` kwarg (or a header
        # hook), pass it through there.
        """
        correlation_id = str(uuid.uuid4())
        receipt = await self.delphi_client.place_order(
            market_id=plan.market_id,
            side=OrderSide(plan.side),
            size_usd=plan.size_usd,
            limit_price=plan.limit_price,
            correlation_id=correlation_id,
        )
        # Attach signature + fingerprint as extra fields (extra="allow").
        # Pydantic v2 lets us set extra attributes on a constructed model.
        try:
            receipt.receipt_signature = signature  # type: ignore[attr-defined]
            receipt.signed_by = self._signing_key_fingerprint()
        except Exception:
            pass
        return receipt

    def _adapt_receipt_for_risk(self, receipt: TradeReceipt) -> RiskTradeReceipt:
        """Convert adapter ``TradeReceipt`` → risk ``TradeReceipt``.

        The risk engine's ``update_state`` expects its own local receipt
        type (which uses ``str`` for side/timestamp rather than enum/datetime).
        """
        side_str = (
            receipt.side.value if hasattr(receipt.side, "value")
            else str(receipt.side)
        )
        ts_str = (
            receipt.timestamp.isoformat()
            if hasattr(receipt.timestamp, "isoformat")
            else str(receipt.timestamp)
        )
        fill_price = (
            float(receipt.fill_price) if receipt.fill_price is not None else 0.0
        )
        return RiskTradeReceipt(
            market_id=receipt.market_id,
            side=side_str,
            size_usd=float(receipt.size_usd),
            fill_price=fill_price,
            att_order_id=receipt.att_order_id,
            signed_by=receipt.signed_by or "",
            timestamp=ts_str,
            audit_log_path=str(self.audit_log_path),
        )

    def _load_signing_key(self) -> bytes | None:
        """Lazily load the signing key from the configured env var.

        Returns the raw key bytes (for Ed25519, 32 bytes) or ``None`` if
        the env var is unset / empty. Caches the result.
        """
        if self._signing_key_loaded:
            return self._signing_key
        env_value = os.environ.get(self.config.signing_key_env, "").strip()
        if not env_value:
            self._signing_key = None
        else:
            # Try base64-url decode first (Ed25519 keys are typically b64).
            try:
                decoded = base64.urlsafe_b64decode(env_value + "=" * (-len(env_value) % 4))
                self._signing_key = decoded
            except Exception:
                # Fall back to raw UTF-8 bytes (HMAC-SHA256 accepts any key).
                self._signing_key = env_value.encode("utf-8")
        self._signing_key_loaded = True
        return self._signing_key

    def _has_ed25519_key(self) -> bool:
        """True iff the signing key is exactly 32 bytes (Ed25519-sized)."""
        key = self._load_signing_key()
        return key is not None and len(key) == 32

    def _canonical_order_bytes(self, plan: TradePlan) -> bytes:
        """Canonical JSON bytes of the order body, for signing.

        Sorts keys + uses compact separators so the signature is
        deterministic across runs / machines.
        """
        body = {
            "market_id": plan.market_id,
            "side": plan.side,
            "size_usd": round(float(plan.size_usd), 8),
            "limit_price": plan.limit_price,
            "timestamp": plan.timestamp,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return canonical.encode("utf-8")

    def _sign_bytes(self, data: bytes) -> str:
        """Sign raw bytes with Ed25519 if available, else HMAC-SHA256."""
        key_bytes = self._load_signing_key()
        if key_bytes is not None and len(key_bytes) == 32:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PrivateKey,
                )

                priv = Ed25519PrivateKey.from_private_bytes(key_bytes)
                sig = priv.sign(data)
                return base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
            except Exception:
                pass
        # HMAC-SHA256 fallback.
        secret = key_bytes or b"pythia-executor-stub-key"
        digest = hmac.new(secret, data, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def _result_to_jsonable(self, result: PipelineResult) -> dict[str, Any]:
        """Convert a ``PipelineResult`` to a JSON-safe dict.

        Uses ``model_dump(mode="json")`` for the pydantic sub-models so
        datetimes become ISO strings, enums become their values, etc.
        Handles the paper-mode receipt whose ``status`` is the string
        ``"PAPER"`` (not a valid ``OrderStatus``) gracefully.
        """
        def _dump(model: BaseModel | None) -> dict[str, Any] | None:
            if model is None:
                return None
            try:
                # Suppress the pydantic warning that fires when an enum
                # field carries a non-enum value (e.g. paper-mode receipts
                # with status="PAPER"). The serialization still produces
                # the correct string output.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    return model.model_dump(mode="json")
            except Exception:
                # Fallback for models with non-enum values in enum fields
                # (e.g. paper-mode receipts with status="PAPER").
                raw = model.model_dump()
                return json.loads(json.dumps(raw, default=str))

        return {
            "market_id": result.market_id,
            "estimates": [e.model_dump(mode="json") for e in result.estimates],
            "decision": _dump(result.decision),
            "plan": _dump(result.plan),
            "receipt": _dump(result.receipt),
            "skipped_reason": result.skipped_reason,
            "timestamp": result.timestamp,
        }


__all__ = ["PythiaExecutor"]
