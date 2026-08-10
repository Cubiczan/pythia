"""Tests for the Bridge subprocess manager and DelphiClient.

These tests use a fake bridge process (a small Python script that mimics
bridge.mjs's JSON-RPC protocol) so they don't require the actual Node.js
SDK to be installed. This keeps the test suite hermetic.

For integration tests that exercise the real SDK, see
``test_integration.py`` (skipped unless ``PYTHIA_INTEGRATION=1`` is set).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pythia_delphi_adapter.bridge import Bridge, BRIDGE_READY_TIMEOUT_SEC
from pythia_delphi_adapter.client import DelphiClient, _bigint_to_str
from pythia_delphi_adapter.errors import BridgeError, BridgeNotReadyError, DelphiAPIError
from pythia_delphi_adapter.models import MarketStatus


# ---------------------------------------------------------------------------
# Fake bridge process — a Python script that speaks the JSON-RPC protocol
# ---------------------------------------------------------------------------

FAKE_BRIDGE_PY = textwrap.dedent("""
    import sys, json, time

    # Signal ready
    sys.stderr.write("[bridge] ready\\n")
    sys.stderr.flush()

    # Pre-canned responses keyed by method name
    RESPONSES = {
        "health": {"status": "ok"},
        "listMarkets": {"markets": []},
        "getMarket": {"id": "0xabc", "appMarketId": "uuid-1", "marketUrl": "https://example",
                      "status": "open", "category": "crypto", "deployer": "0xfeed",
                      "implementation": "0ximp", "metadataUri": "ipfs://x",
                      "metadataUriContentHash": "0xh", "dataSources": None,
                      "createdAt": "2026-08-01T00:00:00Z", "fetchedAt": None,
                      "fetchResponseStatus": None, "resolvesAt": None, "settledAt": None,
                      "settlesAt": None, "winningOutcomeIdx": None, "tradingFee": None,
                      "proof": None, "error": None, "verifiable": True,
                      "metadata": {"question": "Test?", "outcomes": ["YES", "NO"]}},
        "getMarketStatus": "open",
        "quoteBuy": {"tokensIn": "650000000000000000"},
        "buyShares": {"transactionHash": "0xdeadbeef"},
        "getEthBalance": {"__type": "bigint", "value": "1000000000000000000"},
        "getErc20BalanceWithDecimals": {"balance": {"__type": "bigint", "value": "5000000000000000000"}, "decimals": 18},
    }

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        method = req.get("method")
        if method in RESPONSES:
            resp = {"jsonrpc": "2.0", "id": rid, "result": RESPONSES[method]}
        elif method == "errorMethod":
            resp = {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": "SDK error",
                              "data": {"shortMessage": "reverted"}}}
        else:
            resp = {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
""")


@pytest.fixture
def fake_bridge_script(tmp_path: Path) -> Path:
    """Write the fake bridge to a .py file and return its path."""
    script = tmp_path / "fake_bridge.py"
    script.write_text(FAKE_BRIDGE_PY)
    return script


@pytest.fixture
async def bridge_with_fake(fake_bridge_script: Path) -> Bridge:
    """A Bridge instance that runs the fake Python bridge instead of node."""
    # We can't use bridge.mjs — instead, run python with the fake script.
    # Trick: override the node binary and bridge script to use python.
    b = Bridge(
        node_bin=sys.executable,
        bridge_script=fake_bridge_script,
        ready_timeout=5.0,
        default_call_timeout=5.0,
    )
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


# ---------------------------------------------------------------------------
# _bigint_to_str helper
# ---------------------------------------------------------------------------

class TestBigintToStr:
    def test_none(self):
        assert _bigint_to_str(None) == "0"

    def test_string(self):
        assert _bigint_to_str("123") == "123"

    def test_int(self):
        assert _bigint_to_str(456) == "456"

    def test_bigint_marker(self):
        assert _bigint_to_str({"__type": "bigint", "value": "999"}) == "999"

    def test_float(self):
        assert _bigint_to_str(12.0) == "12"


# ---------------------------------------------------------------------------
# Bridge lifecycle
# ---------------------------------------------------------------------------

class TestBridgeLifecycle:
    async def test_start_and_stop(self, bridge_with_fake: Bridge):
        b = bridge_with_fake
        assert b._proc is not None
        assert b._proc.returncode is None  # still running
        # Bridge is already started by the fixture

    async def test_start_is_idempotent(self, bridge_with_fake: Bridge):
        b = bridge_with_fake
        # Calling start again should be a no-op
        await b.start()
        assert b._proc is not None
        assert b._proc.returncode is None

    async def test_stop_kills_process(self, fake_bridge_script: Path):
        b = Bridge(
            node_bin=sys.executable,
            bridge_script=fake_bridge_script,
            ready_timeout=5.0,
        )
        await b.start()
        proc = b._proc
        assert proc is not None
        await b.stop()
        # Process should have exited
        assert proc.returncode is not None

    async def test_start_fails_on_missing_script(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.py"
        b = Bridge(
            node_bin=sys.executable,
            bridge_script=missing,
            ready_timeout=1.0,
        )
        with pytest.raises(BridgeError, match="Bridge script not found"):
            await b.start()

    async def test_start_times_out_on_no_ready_signal(self, tmp_path: Path):
        # A script that never prints "[bridge] ready"
        bad_script = tmp_path / "bad_bridge.py"
        bad_script.write_text("import sys; sys.stderr.write('nothing\\n'); sys.stderr.flush(); sys.stdin.read()")
        b = Bridge(
            node_bin=sys.executable,
            bridge_script=bad_script,
            ready_timeout=1.5,
        )
        with pytest.raises(BridgeError, match="did not signal ready"):
            await b.start()


# ---------------------------------------------------------------------------
# Bridge.call
# ---------------------------------------------------------------------------

class TestBridgeCall:
    async def test_health(self, bridge_with_fake: Bridge):
        result = await bridge_with_fake.call("health", {})
        assert result == {"status": "ok"}

    async def test_list_markets(self, bridge_with_fake: Bridge):
        result = await bridge_with_fake.call("listMarkets", {"limit": 10})
        assert result == {"markets": []}

    async def test_get_market(self, bridge_with_fake: Bridge):
        result = await bridge_with_fake.call("getMarket", {"id": "0xabc"})
        assert result["id"] == "0xabc"
        assert result["status"] == "open"

    async def test_get_market_status(self, bridge_with_fake: Bridge):
        result = await bridge_with_fake.call("getMarketStatus", {"marketAddress": "0xabc"})
        assert result == "open"

    async def test_quote_buy(self, bridge_with_fake: Bridge):
        result = await bridge_with_fake.call("quoteBuy", {
            "marketAddress": "0xabc", "outcomeIdx": 0, "sharesOut": "1000000000000000000",
        })
        assert result == {"tokensIn": "650000000000000000"}

    async def test_buy_shares(self, bridge_with_fake: Bridge):
        result = await bridge_with_fake.call("buyShares", {
            "marketAddress": "0xabc", "outcomeIdx": 0,
            "sharesOut": "1000000000000000000", "maxTokensIn": "700000000000000000",
        })
        assert result == {"transactionHash": "0xdeadbeef"}

    async def test_error_propagation(self, bridge_with_fake: Bridge):
        with pytest.raises(DelphiAPIError) as exc_info:
            await bridge_with_fake.call("errorMethod", {})
        assert "SDK error" in str(exc_info.value)
        assert exc_info.value.code == -32000
        assert exc_info.value.data["shortMessage"] == "reverted"

    async def test_method_not_found(self, bridge_with_fake: Bridge):
        with pytest.raises(DelphiAPIError, match="Method not found: nonexistent"):
            await bridge_with_fake.call("nonexistent", {})

    async def test_call_before_start_raises(self, fake_bridge_script: Path):
        b = Bridge(
            node_bin=sys.executable,
            bridge_script=fake_bridge_script,
            ready_timeout=5.0,
        )
        with pytest.raises(BridgeNotReadyError):
            await b.call("health", {})

    async def test_bigint_serialization(self, bridge_with_fake: Bridge):
        # The fake bridge returns bigint markers for getEthBalance
        result = await bridge_with_fake.call("getEthBalance", {})
        assert result == {"__type": "bigint", "value": "1000000000000000000"}


# ---------------------------------------------------------------------------
# DelphiClient (uses fake bridge)
# ---------------------------------------------------------------------------

class TestDelphiClient:
    async def test_health(self, bridge_with_fake: Bridge):
        client = DelphiClient(bridge=bridge_with_fake, auto_start=False)
        health = await client.health()
        assert health.status == "ok"

    async def test_list_markets(self, bridge_with_fake: Bridge):
        client = DelphiClient(bridge=bridge_with_fake, auto_start=False)
        markets = await client.list_markets(limit=10)
        assert markets == []

    async def test_get_market(self, bridge_with_fake: Bridge):
        client = DelphiClient(bridge=bridge_with_fake, auto_start=False)
        market = await client.get_market("0xabc")
        assert market.market_address == "0xabc"
        assert market.status == MarketStatus.OPEN
        assert market.outcomes == ["YES", "NO"]

    async def test_quote_buy(self, bridge_with_fake: Bridge):
        client = DelphiClient(bridge=bridge_with_fake, auto_start=False)
        quote = await client.quote_buy(
            market_address="0xabc",
            outcome_idx=0,
            shares_out="1000000000000000000",
        )
        assert quote.tokens_in == "650000000000000000"

    async def test_buy_shares_returns_receipt(self, bridge_with_fake: Bridge):
        client = DelphiClient(bridge=bridge_with_fake, auto_start=False)
        receipt = await client.buy_shares(
            market_address="0xabc",
            outcome_idx=0,
            shares_out="1000000000000000000",
            max_tokens_in="700000000000000000",
        )
        assert receipt.transaction_hash == "0xdeadbeef"
        assert receipt.side == "buy"
        assert receipt.market_address == "0xabc"
        assert receipt.outcome_idx == 0

    async def test_get_eth_balance(self, bridge_with_fake: Bridge):
        client = DelphiClient(bridge=bridge_with_fake, auto_start=False)
        balance = await client.get_eth_balance()
        assert balance == "1000000000000000000"

    async def test_get_erc20_balance(self, bridge_with_fake: Bridge):
        client = DelphiClient(bridge=bridge_with_fake, auto_start=False)
        bal = await client.get_erc20_balance()
        assert bal.balance == "5000000000000000000"
        assert bal.decimals == 18

    async def test_error_propagation(self, bridge_with_fake: Bridge):
        client = DelphiClient(bridge=bridge_with_fake, auto_start=False)
        # The fake bridge returns an error for "errorMethod"
        with pytest.raises(DelphiAPIError):
            await client._call("errorMethod", {}, timeout=5.0)

    async def test_context_manager(self, fake_bridge_script: Path):
        # When the client owns the bridge, async with should start/stop it
        b = Bridge(
            node_bin=sys.executable,
            bridge_script=fake_bridge_script,
            ready_timeout=5.0,
        )
        async with DelphiClient(bridge=b) as client:
            health = await client.health()
            assert health.status == "ok"
