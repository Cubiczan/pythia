"""Tests for the pydantic models in pythia_delphi_adapter.models.

These tests verify that the models correctly parse the JSON shapes the
@gensyn-ai/gensyn-delphi-sdk produces, including edge cases like missing
optional fields, string-encoded integers, and alias mappings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from pythia_delphi_adapter.models import (
    BalanceResponse,
    BuySharesParams,
    HealthResponse,
    LiquidateParams,
    ListMarketsParams,
    ListPositionsParams,
    Market,
    MarketMetadata,
    MarketStatus,
    Network,
    Position,
    QuoteBuyParams,
    QuoteBuyResponse,
    RedeemMarketResponse,
    SignerType,
    TradeReceipt,
)


# ---------------------------------------------------------------------------
# Sample SDK JSON fixtures (shapes copied from the SDK's TypeScript types)
# ---------------------------------------------------------------------------

SAMPLE_MARKET_JSON = {
    "id": "0x1234567890abcdef1234567890abcdef12345678",
    "appMarketId": "01J7Q5Z3M2N4R6P8",
    "marketUrl": "https://testnet.delphi.fyi/m/01J7Q5Z3M2N4R6P8",
    "status": "open",
    "category": "crypto",
    "deployer": "0xabcdef000000000000000000000000000000abcd",
    "implementation": "0xaaaa0000000000000000000000000000000000aa",
    "metadataUri": "ipfs://QmXYZ",
    "metadataUriContentHash": "0xdeadbeef",
    "dataSources": None,
    "createdAt": "2026-08-01T12:00:00Z",
    "fetchedAt": "2026-08-10T00:00:00Z",
    "fetchResponseStatus": "ok",
    "resolvesAt": "2026-12-31T23:59:59Z",
    "settledAt": None,
    "settlesAt": "2026-12-31T23:59:59Z",
    "winningOutcomeIdx": None,
    "tradingFee": "0",
    "proof": None,
    "error": None,
    "verifiable": True,
    "metadata": {
        "question": "Will Bitcoin close above $100,000 on 2026-12-31?",
        "outcomes": ["YES", "NO"],
        "model_identifier": "claude-opus-4-1",
        "prompt_context": "Resolve based on Coinbase BTC-USD close price.",
    },
    "spotPrices": [0.65, 0.35],
    "spotImpliedProbabilities": [0.65, 0.35],
}

SAMPLE_MULTI_OUTCOME_MARKET_JSON = {
    **SAMPLE_MARKET_JSON,
    "id": "0xmultioutcome1234567890abcdef1234567890abcd",
    "appMarketId": "01J7MULTI0001",
    "metadata": {
        "question": "Which L1 will have the highest TVL on 2026-12-31?",
        "outcomes": ["Bitcoin", "Ethereum", "Solana", "Other"],
        "model_identifier": "gpt-5",
    },
    "spotPrices": [0.45, 0.30, 0.20, 0.05],
    "spotImpliedProbabilities": [0.45, 0.30, 0.20, 0.05],
}

SAMPLE_POSITION_JSON = {
    "id": "pos_001",
    "marketProxy": "0x1234567890abcdef1234567890abcdef12345678",
    "wallet": "0xfeed0000000000000000000000000000000000fd",
    "outcomeIdx": "0",
    "shares": "1000000000000000000",
    "redeemedOrLiquidated": False,
    "tokensRedeemed": "0",
    "marketStatus": "open",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_market_status_values(self):
        assert MarketStatus.OPEN.value == "open"
        assert MarketStatus.SETTLED.value == "settled"
        assert MarketStatus.EXPIRED.value == "expired"
        assert MarketStatus.FAILED.value == "failed"
        assert MarketStatus.AWAITING_SETTLEMENT.value == "awaiting_settlement"

    def test_network_values(self):
        assert Network.TESTNET.value == "testnet"
        assert Network.MAINNET.value == "mainnet"
        assert Network.COMPETITION_TESTNET.value == "competition-testnet"

    def test_signer_type_values(self):
        assert SignerType.CDP_SERVER_WALLET.value == "cdp_server_wallet"
        assert SignerType.PRIVATE_KEY.value == "private_key"


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

class TestMarket:
    def test_parses_binary_market(self):
        m = Market.model_validate(SAMPLE_MARKET_JSON)
        assert m.market_address == "0x1234567890abcdef1234567890abcdef12345678"
        assert m.app_market_id == "01J7Q5Z3M2N4R6P8"
        assert m.status == MarketStatus.OPEN
        assert m.category == "crypto"
        assert m.outcomes == ["YES", "NO"]
        assert m.question == "Will Bitcoin close above $100,000 on 2026-12-31?"
        assert m.spot_prices == [0.65, 0.35]
        assert m.spot_implied_probabilities == [0.65, 0.35]
        assert m.winning_outcome_idx is None
        assert m.created_at == datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert m.settles_at == datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        assert m.settled_at is None

    def test_parses_multi_outcome_market(self):
        m = Market.model_validate(SAMPLE_MULTI_OUTCOME_MARKET_JSON)
        assert len(m.outcomes) == 4
        assert m.outcomes == ["Bitcoin", "Ethereum", "Solana", "Other"]
        assert len(m.spot_prices) == 4
        assert m.spot_prices[0] == 0.45

    def test_parses_settled_market_with_winning_outcome(self):
        settled = {
            **SAMPLE_MARKET_JSON,
            "status": "settled",
            "winningOutcomeIdx": "0",
            "settledAt": "2027-01-01T00:00:00Z",
        }
        m = Market.model_validate(settled)
        assert m.status == MarketStatus.SETTLED
        assert m.winning_outcome_idx == 0
        assert m.settled_at == datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_parses_failed_market(self):
        failed = {
            **SAMPLE_MARKET_JSON,
            "status": "failed",
            "winningOutcomeIdx": None,
        }
        m = Market.model_validate(failed)
        assert m.status == MarketStatus.FAILED
        assert m.winning_outcome_idx is None

    def test_parses_market_without_metadata(self):
        no_meta = {**SAMPLE_MARKET_JSON, "metadata": None}
        m = Market.model_validate(no_meta)
        assert m.metadata is None
        assert m.outcomes == []
        assert m.question == ""

    def test_parses_market_without_prices(self):
        no_prices = {k: v for k, v in SAMPLE_MARKET_JSON.items()
                     if k not in ("spotPrices", "spotImpliedProbabilities")}
        m = Market.model_validate(no_prices)
        assert m.spot_prices is None
        assert m.spot_implied_probabilities is None

    def test_coerces_winning_outcome_idx_string(self):
        m = Market.model_validate({**SAMPLE_MARKET_JSON, "winningOutcomeIdx": "2"})
        assert m.winning_outcome_idx == 2

    def test_handles_empty_winning_outcome_idx(self):
        m = Market.model_validate({**SAMPLE_MARKET_JSON, "winningOutcomeIdx": ""})
        assert m.winning_outcome_idx is None

    def test_naive_datetime_coerced_to_utc(self):
        m = Market.model_validate({**SAMPLE_MARKET_JSON, "createdAt": "2026-08-01T12:00:00"})
        assert m.created_at.tzinfo == timezone.utc


class TestMarketMetadata:
    def test_parses_full_metadata(self):
        md = MarketMetadata.model_validate(SAMPLE_MARKET_JSON["metadata"])
        assert md.question.startswith("Will Bitcoin")
        assert md.outcomes == ["YES", "NO"]
        assert md.model_identifier == "claude-opus-4-1"

    def test_parses_minimal_metadata(self):
        md = MarketMetadata.model_validate({"question": "test?"})
        assert md.question == "test?"
        assert md.outcomes == []
        assert md.model_identifier is None


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_parses_position(self):
        p = Position.model_validate(SAMPLE_POSITION_JSON)
        assert p.position_id == "pos_001"
        assert p.market_address == "0x1234567890abcdef1234567890abcdef12345678"
        assert p.outcome_idx == 0
        assert p.shares == "1000000000000000000"
        assert p.redeemed_or_liquidated is False
        assert p.market_status == MarketStatus.OPEN

    def test_coerces_outcome_idx_string(self):
        p = Position.model_validate({**SAMPLE_POSITION_JSON, "outcomeIdx": "3"})
        assert p.outcome_idx == 3

    def test_handles_redeemed_position(self):
        redeemed = {
            **SAMPLE_POSITION_JSON,
            "redeemedOrLiquidated": True,
            "tokensRedeemed": "1500000000000000000",
            "marketStatus": "settled",
        }
        p = Position.model_validate(redeemed)
        assert p.redeemed_or_liquidated is True
        assert p.tokens_redeemed == "1500000000000000000"


# ---------------------------------------------------------------------------
# Trade params / responses
# ---------------------------------------------------------------------------

class TestQuoteBuyParams:
    def test_builds_params_with_aliases(self):
        p = QuoteBuyParams(
            marketAddress="0xabc",
            outcomeIdx=0,
            sharesOut="1000000000000000000",
        )
        dumped = p.model_dump(by_alias=True)
        assert dumped["marketAddress"] == "0xabc"
        assert dumped["outcomeIdx"] == 0
        assert dumped["sharesOut"] == "1000000000000000000"


class TestQuoteBuyResponse:
    def test_parses_response(self):
        r = QuoteBuyResponse.model_validate({"tokensIn": "650000000000000000"})
        assert r.tokens_in == "650000000000000000"


class TestBuySharesParams:
    def test_builds_params(self):
        p = BuySharesParams(
            marketAddress="0xabc",
            outcomeIdx=1,
            sharesOut="1000000000000000000",
            maxTokensIn="700000000000000000",
        )
        dumped = p.model_dump(by_alias=True)
        assert dumped["maxTokensIn"] == "700000000000000000"


class TestTradeReceipt:
    def test_builds_receipt(self):
        r = TradeReceipt(
            market_address="0xabc",
            outcome_idx=0,
            side="buy",
            shares="1000000000000000000",
            transaction_hash="0xdeadbeef",
            timestamp=datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        assert r.side == "buy"
        assert r.transaction_hash == "0xdeadbeef"


class TestLiquidateParams:
    def test_builds_params(self):
        p = LiquidateParams(
            marketAddress="0xabc",
            outcomeIndices=[0, 1, 2],
        )
        dumped = p.model_dump(by_alias=True)
        assert dumped["outcomeIndices"] == [0, 1, 2]


class TestRedeemMarketResponse:
    def test_parses_response(self):
        r = RedeemMarketResponse.model_validate({
            "marketAddress": "0xabc",
            "transactionHash": "0xdeadbeef",
            "sharesIn": "1000000000000000000",
            "tokensOut": "950000000000000000",
        })
        assert r.market_address == "0xabc"
        assert r.tokens_out == "950000000000000000"


# ---------------------------------------------------------------------------
# List params
# ---------------------------------------------------------------------------

class TestListMarketsParams:
    def test_defaults(self):
        p = ListMarketsParams()
        dumped = p.model_dump(by_alias=True)
        assert dumped["skip"] == 0
        assert dumped["limit"] == 50
        assert dumped["orderBy"] == "liquidity"
        assert dumped["pricesAndImpliedProbabilities"] is False

    def test_with_prices(self):
        p = ListMarketsParams(prices_and_implied_probabilities=True)
        dumped = p.model_dump(by_alias=True)
        assert dumped["pricesAndImpliedProbabilities"] is True

    def test_with_competition_id(self):
        p = ListMarketsParams(competition_id="uuid-here")
        dumped = p.model_dump(by_alias=True)
        assert dumped["competitionId"] == "uuid-here"


class TestListPositionsParams:
    def test_builds_params(self):
        p = ListPositionsParams(wallet="0xfeed")
        dumped = p.model_dump(by_alias=True)
        assert dumped["wallet"] == "0xfeed"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestHealthResponse:
    def test_ok(self):
        h = HealthResponse.model_validate({"status": "ok"})
        assert h.status == "ok"

    def test_error(self):
        h = HealthResponse.model_validate({"status": "degraded"})
        assert h.status == "degraded"


class TestBalanceResponse:
    def test_parses_balance(self):
        b = BalanceResponse(balance="1000000000000000000", decimals=18)
        assert b.balance == "1000000000000000000"
        assert b.decimals == 18

    def test_defaults_to_18_decimals(self):
        b = BalanceResponse(balance="0")
        assert b.decimals == 18
