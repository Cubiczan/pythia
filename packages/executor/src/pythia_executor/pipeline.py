"""PythiaExecutor: the mesh → consensus → risk → SDK pipeline.

This is the orchestrator that ties the four sibling Pythia packages
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
10. in live mode: ensures ERC-20 approval and submits via
    ``DelphiClient.buy_shares``,
11. appends the full ``PipelineResult`` to the audit log as signed JSONL,
12. updates ``risk_engine.state`` via ``update_state(receipt)``.

The executor uses the @gensyn-ai/gensyn-delphi-sdk via the adapter's Node
bridge. Trades are submitted as on-chain transactions (buy_shares returns
a transaction_hash, not an HTTP order id).
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
from pythia_delphi_adapter import TradeReceipt
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
        Async client for the Gensyn Delphi SDK. Must implement
        ``get_market(id) -> Market`` and ``buy_shares(...) -> TradeReceipt``.
    mesh:
        List of instantiated ``BaseAnalyst`` objects (politics / crypto / ...).
    consensus_engine:
        ``ConsensusEngine`` from pythia-consensus.
    risk_engine:
        ``RiskEngine`` from pythia-risk.
    config:
        ``ExecutorConfig`` (mode, signing key env, retry settings).
    audit_log_path:
        ``Path`` to the JSONL audit log. Created if it doesn't exist.
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

        self._signing_key: bytes | None = None
        self._signing_key_loaded = False

        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ API

    async def run_for_market(self, market_id: str) -> PipelineResult:
        """Run the full pipeline for one market.

        Returns a ``PipelineResult`` regardless of whether the pipeline
        reached submission or skipped early.
        """
        timestamp = datetime.now(UTC).isoformat()

        # ----- Step 1: fetch market ----------------------------------------
        market: AdapterMarket = await self.delphi_client.get_market(
            market_id, prices_and_implied_probabilities=True
        )

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
                "market=%s PAPER trade: outcome=%s (idx=%d) size=$%.2f",
                market_id, plan.side, plan.outcome_idx, plan.size_usd,
            )
        else:
            # Live mode: ensure approval + submit via SDK.
            receipt = await self._submit_live_order(plan, market)
            logger.info(
                "market=%s LIVE trade: outcome=%s (idx=%d) size=$%.2f tx=%s",
                market_id, plan.side, plan.outcome_idx, plan.size_usd,
                receipt.transaction_hash,
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
        risk_receipt = self._adapt_receipt_for_risk(receipt, plan)
        self.risk_engine.update_state(risk_receipt)

        return result

    # ------------------------------------------------------------------ sign

    def _sign_order(self, plan: TradePlan) -> str:
        """Sign a TradePlan for audit-trail attestation.

        This is NOT the on-chain transaction signature (the SDK handles that
        internally via the configured signer). This signature is over the
        canonical JSON of the plan, written to the audit log so reviewers
        can verify the plan wasn't tampered with between risk sizing and
        execution.
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
        """Append a signed JSONL line to ``audit_log_path``."""
        payload = self._result_to_jsonable(result)
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
        category_str = str(market.category)
        settles_at_str = (
            market.settles_at.isoformat() if market.settles_at is not None else None
        )
        # Use spot_prices from the adapter Market (populated when
        # prices_and_implied_probabilities=True is passed to get_market).
        spot_prices = market.spot_prices or []
        outcomes = market.outcomes or ["YES", "NO"]

        # Legacy convenience fields for binary markets.
        yes_price = spot_prices[0] if len(spot_prices) >= 1 else None
        no_price = spot_prices[1] if len(spot_prices) >= 2 else None

        return MarketContext(
            market_id=market.market_address,
            question=market.question,
            category=category_str,
            metadata={
                "app_market_id": market.app_market_id,
                "market_url": market.market_url,
                "status": market.status.value if hasattr(market.status, "value") else str(market.status),
                "outcomes": outcomes,
            },
            outcomes=outcomes,
            spot_prices=spot_prices,
            current_yes_price=yes_price,
            current_no_price=no_price,
            volume_usd=None,  # SDK Market doesn't expose volume_usd
            closes_at=settles_at_str,
        )

    def _adapt_market_for_risk(self, market: AdapterMarket) -> Any:
        """Convert adapter ``Market`` → risk ``pythia_risk.Market``."""
        from pythia_risk import Market as RiskMarket

        return RiskMarket(
            market_id=market.market_address,
            category=str(market.category).lower(),
            question=market.question,
            outcomes=market.outcomes or ["YES", "NO"],
            spot_prices=market.spot_prices or [],
            close_date=market.settles_at,
        )

    def _build_paper_receipt(
        self, plan: TradePlan, market: AdapterMarket
    ) -> TradeReceipt:
        """Synthesise a paper-mode ``TradeReceipt`` (no on-chain submission).

        The receipt uses a fake transaction hash so the audit log shape is
        identical to a live receipt.
        """
        return TradeReceipt(
            market_address=plan.market_id,
            outcome_idx=plan.outcome_idx,
            side="buy",
            shares=str(int(plan.size_usd * 10**18)),  # mock 18-decimal shares
            transaction_hash=f"0xpaper-{uuid.uuid4().hex}",
            max_tokens_in=str(int(plan.size_usd * 10**18)),
            timestamp=datetime.now(UTC),
        )

    async def _submit_live_order(self, plan: TradePlan, market: AdapterMarket) -> TradeReceipt:
        """Submit a live buy order via the SDK.

        Calls ``ensure_token_approval`` first (idempotent), then ``buy_shares``.
        The SDK handles signing via the configured signer (CDP or private_key).
        """
        # Convert dollar stake to 18-decimal token amount (competition token
        # uses 18 decimals). This is a simplification — the real conversion
        # depends on the current spot price, which we get from quote_buy.
        shares_out_wei = str(int(plan.size_usd * 10**18))
        max_tokens_wei = str(int(plan.size_usd * 10**18 * 1.05))  # 5% slippage

        # Ensure the gateway can spend our tokens.
        await self.delphi_client.ensure_token_approval(
            market_address=plan.market_id,
            minimum_amount=max_tokens_wei,
        )

        # Submit the buy.
        receipt = await self.delphi_client.buy_shares(
            market_address=plan.market_id,
            outcome_idx=plan.outcome_idx,
            shares_out=shares_out_wei,
            max_tokens_in=max_tokens_wei,
        )
        return receipt

    def _adapt_receipt_for_risk(
        self, receipt: TradeReceipt, plan: TradePlan
    ) -> RiskTradeReceipt:
        """Convert adapter ``TradeReceipt`` → risk ``TradeReceipt``."""
        ts_str = (
            receipt.timestamp.isoformat()
            if hasattr(receipt.timestamp, "isoformat")
            else str(receipt.timestamp)
        )
        fill_price = float(plan.limit_price) if plan.limit_price is not None else 0.0
        return RiskTradeReceipt(
            market_id=receipt.market_address,
            side=plan.side,
            outcome_idx=plan.outcome_idx,
            size_usd=float(plan.size_usd),
            fill_price=fill_price,
            att_order_id=receipt.transaction_hash,
            signed_by=self._signing_key_fingerprint(),
            timestamp=ts_str,
            audit_log_path=str(self.audit_log_path),
        )

    def _load_signing_key(self) -> bytes | None:
        """Lazily load the signing key from the configured env var."""
        if self._signing_key_loaded:
            return self._signing_key
        env_value = os.environ.get(self.config.signing_key_env, "").strip()
        if not env_value:
            self._signing_key = None
        else:
            try:
                decoded = base64.urlsafe_b64decode(env_value + "=" * (-len(env_value) % 4))
                self._signing_key = decoded
            except Exception:
                self._signing_key = env_value.encode("utf-8")
        self._signing_key_loaded = True
        return self._signing_key

    def _has_ed25519_key(self) -> bool:
        key = self._load_signing_key()
        return key is not None and len(key) == 32

    def _canonical_order_bytes(self, plan: TradePlan) -> bytes:
        """Canonical JSON bytes of the order body, for signing."""
        body = {
            "market_id": plan.market_id,
            "outcome_idx": plan.outcome_idx,
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
        secret = key_bytes or b"pythia-executor-stub-key"
        digest = hmac.new(secret, data, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def _result_to_jsonable(self, result: PipelineResult) -> dict[str, Any]:
        """Convert a ``PipelineResult`` to a JSON-safe dict."""
        def _dump(model: BaseModel | None) -> dict[str, Any] | None:
            if model is None:
                return None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    return model.model_dump(mode="json")
            except Exception:
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
