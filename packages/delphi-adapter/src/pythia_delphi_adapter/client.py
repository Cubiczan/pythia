"""Async client for the @gensyn-ai/gensyn-delphi-sdk via a Node subprocess bridge.

This is the only module the rest of the Pythia mesh imports from
``pythia_delphi_adapter``. It wraps every SDK method the mesh needs behind
typed Python coroutines, so callers never see raw JSON-RPC.

Architecture:
    DelphiClient (Python)
        └─ Bridge (Python subprocess manager)
             └─ node bridge.mjs (Node.js)
                  └─ @gensyn-ai/gensyn-delphi-sdk (TypeScript)
                       └─ DelphiClient (TypeScript)
                            ├─ REST API (listMarkets, getMarket, listPositions)
                            └─ viem WalletClient (buyShares, sellShares, redeem, liquidate)

The Python ``DelphiClient`` is a thin façade. Each method:
  1. Validates params via the pydantic models in ``models.py``.
  2. Calls ``Bridge.call(method, params)`` which sends a JSON-RPC request.
  3. Parses the result back into a typed pydantic model.

Configuration:
    The SDK reads its config from environment variables. The Python client
    does NOT duplicate this — set the env vars before constructing the
    client (or before calling ``start()`` on the underlying bridge):

        DELPHI_NETWORK=competition-testnet
        DELPHI_API_ACCESS_KEY=...     # from https://delphi-api-access.gensyn.ai/
        DELPHI_SIGNER_TYPE=private_key
        WALLET_PRIVATE_KEY=0x...      # only for private_key signing

    For CDP Server Wallet signing (the SDK default), set CDP_API_KEY_ID,
    CDP_API_KEY_SECRET, CDP_WALLET_SECRET, CDP_WALLET_ADDRESS instead.

Paper mode:
    The SDK has no built-in "paper" flag — every ``buyShares`` / ``sellShares``
    call sends a real on-chain transaction. Pythia's paper mode is implemented
    at the executor level (``pythia_executor``): the executor simply doesn't
    call ``buy_shares`` / ``sell_shares`` in paper mode, but still calls the
    read-only methods (``list_markets``, ``get_market``, ``quote_buy``) so
    the rest of the pipeline runs identically.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from .bridge import Bridge, DEFAULT_CALL_TIMEOUT_SEC
from .errors import BridgeError, DelphiAPIError
from .models import (
    BalanceResponse,
    BuySharesParams,
    EnsureTokenApprovalParams,
    EnsureTokenApprovalResponse,
    HealthResponse,
    LiquidateParams,
    LiquidateResponse,
    ListMarketsParams,
    ListPositionsParams,
    Market,
    Position,
    QuoteBuyParams,
    QuoteBuyResponse,
    QuoteSellParams,
    QuoteSellResponse,
    RedeemMarketParams,
    RedeemMarketResponse,
    SellSharesParams,
    TradeReceipt,
)


__all__ = ["DelphiClient", "DEFAULT_CALL_TIMEOUT_SEC"]


class DelphiClient:
    """Async client for the Gensyn Delphi SDK.

    Usage:

        async with DelphiClient() as client:
            health = await client.health()
            markets = await client.list_markets(status="open", limit=10)
            quote = await client.quote_buy(
                market_address="0x...",
                outcome_idx=0,
                shares_out="1000000000000000000",  # 1 share
            )

    The ``async with`` context manager starts the Node bridge on enter and
    stops it on exit. For long-running services, construct once and call
    ``await client.start()`` / ``await client.stop()`` manually.
    """

    def __init__(
        self,
        *,
        bridge: Bridge | None = None,
        network: str | None = None,
        env: dict[str, str] | None = None,
        auto_start: bool = True,
    ) -> None:
        self._env = dict(env or {})
        if network:
            self._env.setdefault("DELPHI_NETWORK", network)
        self._bridge = bridge or Bridge(env=self._env)
        self._owns_bridge = bridge is None
        self._auto_start = auto_start

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Node bridge subprocess."""
        await self._bridge.start()

    async def stop(self) -> None:
        """Stop the Node bridge subprocess (if we own it)."""
        if self._owns_bridge:
            await self._bridge.stop()

    async def __aenter__(self) -> "DelphiClient":
        if self._auto_start:
            # Start the bridge regardless of ownership — start() is idempotent.
            await self._bridge.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_bridge:
            await self._bridge.stop()

    # ------------------------------------------------------------------
    # Health & info
    # ------------------------------------------------------------------

    async def health(self, *, timeout: float = 10.0) -> HealthResponse:
        """Check REST API health. Does not require authentication."""
        result = await self._call("health", {}, timeout=timeout)
        return HealthResponse.model_validate(result)

    # ------------------------------------------------------------------
    # Market reads
    # ------------------------------------------------------------------

    async def list_markets(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
        order_by: str = "liquidity",
        status: str | None = None,
        category: str | None = None,
        verifiable: bool | None = None,
        competition_id: str | None = None,
        prices_and_implied_probabilities: bool = False,
        timeout: float = 30.0,
    ) -> list[Market]:
        """List markets with pagination and optional filters.

        Set ``prices_and_implied_probabilities=True`` to fetch on-chain spot
        prices and implied probabilities for each market's outcomes via a
        multicall. This is slower (one extra RPC round-trip per batch) but
        required for the risk engine to size positions against current prices.
        """
        params = ListMarketsParams(
            skip=skip,
            limit=limit,
            order_by=order_by,
            status=status,
            category=category,
            verifiable=verifiable,
            competition_id=competition_id,
            prices_and_implied_probabilities=prices_and_implied_probabilities,
        )
        result = await self._call("listMarkets", params.model_dump(by_alias=True), timeout=timeout)
        markets_data = (result or {}).get("markets") or []
        return [Market.model_validate(m) for m in markets_data]

    async def get_market(
        self,
        market_id: str,
        *,
        competition_id: str | None = None,
        prices_and_implied_probabilities: bool = False,
        timeout: float = 15.0,
    ) -> Market:
        """Retrieve a single market by its app UUID or contract address."""
        params = {
            "id": market_id,
            "competitionId": competition_id,
            "pricesAndImpliedProbabilities": prices_and_implied_probabilities,
        }
        result = await self._call("getMarket", params, timeout=timeout)
        return Market.model_validate(result)

    async def list_positions(
        self,
        wallet: str,
        *,
        skip: int = 0,
        limit: int = 50,
        redeemed_or_liquidated: bool | None = None,
        timeout: float = 15.0,
    ) -> list[Position]:
        """Retrieve positions for a given wallet address."""
        params = ListPositionsParams(
            wallet=wallet,
            skip=skip,
            limit=limit,
            redeemed_or_liquidated=redeemed_or_liquidated,
        )
        result = await self._call("listPositions", params.model_dump(by_alias=True), timeout=timeout)
        positions_data = (result or {}).get("positions") or []
        return [Position.model_validate(p) for p in positions_data]

    async def get_market_status(self, market_address: str, *, timeout: float = 10.0) -> str:
        """Read a market's lifecycle status directly from its gateway.

        Unlike ``get_market``, this hits the chain rather than the indexed
        REST API, so it reflects state that hasn't been indexed yet.
        """
        result = await self._call(
            "getMarketStatus", {"marketAddress": market_address}, timeout=timeout
        )
        return str(result)

    # ------------------------------------------------------------------
    # Quotes (no on-chain tx)
    # ------------------------------------------------------------------

    async def quote_buy(
        self,
        *,
        market_address: str,
        outcome_idx: int,
        shares_out: str,
        timeout: float = 15.0,
    ) -> QuoteBuyResponse:
        """Quote the collateral required to buy ``shares_out`` of an outcome."""
        params = QuoteBuyParams(
            marketAddress=market_address,
            outcomeIdx=outcome_idx,
            sharesOut=shares_out,
        )
        result = await self._call("quoteBuy", params.model_dump(by_alias=True), timeout=timeout)
        return QuoteBuyResponse.model_validate(result)

    async def quote_sell(
        self,
        *,
        market_address: str,
        outcome_idx: int,
        shares_in: str,
        timeout: float = 15.0,
    ) -> QuoteSellResponse:
        """Quote the collateral you'd receive for selling ``shares_in``."""
        params = QuoteSellParams(
            marketAddress=market_address,
            outcomeIdx=outcome_idx,
            sharesIn=shares_in,
        )
        result = await self._call("quoteSell", params.model_dump(by_alias=True), timeout=timeout)
        return QuoteSellResponse.model_validate(result)

    # ------------------------------------------------------------------
    # Trading (on-chain writes)
    # ------------------------------------------------------------------

    async def buy_shares(
        self,
        *,
        market_address: str,
        outcome_idx: int,
        shares_out: str,
        max_tokens_in: str,
        timeout: float = 120.0,
    ) -> TradeReceipt:
        """Buy shares of a specific outcome.

        Submits an on-chain transaction via the LMSR Gateway. The
        ``max_tokens_in`` parameter is slippage protection: the transaction
        reverts if the collateral required exceeds this amount.

        The SDK returns just ``{transaction_hash}``; we expand it into a
        full ``TradeReceipt`` with the input parameters and timestamp so
        the audit log has a complete record.
        """
        params = BuySharesParams(
            marketAddress=market_address,
            outcomeIdx=outcome_idx,
            sharesOut=shares_out,
            maxTokensIn=max_tokens_in,
        )
        result = await self._call("buyShares", params.model_dump(by_alias=True), timeout=timeout)
        return TradeReceipt(
            market_address=market_address,
            outcome_idx=outcome_idx,
            side="buy",
            shares=shares_out,
            transaction_hash=result.get("transactionHash", ""),
            max_tokens_in=max_tokens_in,
            timestamp=datetime.now(timezone.utc),
        )

    async def sell_shares(
        self,
        *,
        market_address: str,
        outcome_idx: int,
        shares_in: str,
        min_tokens_out: str,
        timeout: float = 120.0,
    ) -> TradeReceipt:
        """Sell shares of a specific outcome."""
        params = SellSharesParams(
            marketAddress=market_address,
            outcomeIdx=outcome_idx,
            sharesIn=shares_in,
            minTokensOut=min_tokens_out,
        )
        result = await self._call("sellShares", params.model_dump(by_alias=True), timeout=timeout)
        return TradeReceipt(
            market_address=market_address,
            outcome_idx=outcome_idx,
            side="sell",
            shares=shares_in,
            transaction_hash=result.get("transactionHash", ""),
            min_tokens_out=min_tokens_out,
            timestamp=datetime.now(timezone.utc),
        )

    async def redeem_market(self, market_address: str, *, timeout: float = 120.0) -> RedeemMarketResponse:
        """Redeem winning shares in a settled market for collateral."""
        params = RedeemMarketParams(marketAddress=market_address)
        result = await self._call("redeemMarket", params.model_dump(by_alias=True), timeout=timeout)
        return RedeemMarketResponse.model_validate(result)

    async def liquidate(
        self,
        *,
        market_address: str,
        outcome_indices: list[int],
        timeout: float = 120.0,
    ) -> LiquidateResponse:
        """Liquidate positions in an expired/failed market."""
        params = LiquidateParams(
            marketAddress=market_address,
            outcomeIndices=outcome_indices,
        )
        result = await self._call("liquidate", params.model_dump(by_alias=True), timeout=timeout)
        return LiquidateResponse.model_validate(result)

    # ------------------------------------------------------------------
    # Token / approval
    # ------------------------------------------------------------------

    async def ensure_token_approval(
        self,
        *,
        market_address: str,
        minimum_amount: str,
        approve_amount: str | None = None,
        timeout: float = 120.0,
    ) -> EnsureTokenApprovalResponse:
        """Ensure the gateway has enough ERC-20 allowance to spend on our behalf.

        Sends an approval transaction only if the current allowance is below
        ``minimum_amount``. Idempotent — safe to call before every trade.
        """
        params = EnsureTokenApprovalParams(
            marketAddress=market_address,
            minimumAmount=minimum_amount,
            approveAmount=approve_amount,
        )
        result = await self._call(
            "ensureTokenApproval", params.model_dump(by_alias=True), timeout=timeout
        )
        return EnsureTokenApprovalResponse.model_validate(result)

    # ------------------------------------------------------------------
    # Balance reads
    # ------------------------------------------------------------------

    async def get_eth_balance(self, *, timeout: float = 10.0) -> str:
        """Native ETH balance of the signer wallet (18-decimal string)."""
        result = await self._call("getEthBalance", {}, timeout=timeout)
        return _bigint_to_str(result)

    async def get_erc20_balance(
        self,
        *,
        token_address: str | None = None,
        timeout: float = 10.0,
    ) -> BalanceResponse:
        """ERC-20 token balance + decimals for the signer wallet."""
        params = {"tokenAddress": token_address}
        result = await self._call("getErc20BalanceWithDecimals", params, timeout=timeout)
        # The SDK returns {balance: bigint, decimals: number} — bigint is
        # serialized as {__type: 'bigint', value: '...'} by the bridge.
        balance_raw = result.get("balance")
        return BalanceResponse(
            balance=_bigint_to_str(balance_raw),
            decimals=int(result.get("decimals", 18)),
        )

    # ------------------------------------------------------------------
    # Gateway routing
    # ------------------------------------------------------------------

    async def resolve_gateway(self, market_address: str, *, timeout: float = 10.0) -> str:
        """Resolve which gateway serves a given market address."""
        result = await self._call(
            "resolveGateway", {"marketAddress": market_address}, timeout=timeout
        )
        return str(result)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float,
    ) -> Any:
        """Send a JSON-RPC call to the Node bridge."""
        if self._auto_start and self._owns_bridge:
            # Ensure the bridge is running. start() is idempotent.
            try:
                await self._bridge.start()
            except BridgeError:
                raise
        return await self._bridge.call(method, params, timeout=timeout)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bigint_to_str(value: Any) -> str:
    """Convert a bridge-serialized bigint back to a decimal string.

    The bridge serializes ``bigint`` as ``{"__type": "bigint", "value": "<dec>"}``
    for precision. Python's ``int`` is unbounded, but downstream Pythia
    consumers (risk engine, audit log) prefer the canonical string form.
    """
    if value is None:
        return "0"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, dict) and value.get("__type") == "bigint":
        return str(value.get("value", "0"))
    return str(value)
