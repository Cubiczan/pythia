#!/usr/bin/env node
/**
 * Pythia Delphi SDK Bridge
 *
 * A long-running Node.js process that the Python `pythia_delphi_adapter` package
 * spawns as a subprocess. The Python side sends JSON-RPC 2.0 messages over stdin
 * (one per line); this bridge loads the official @gensyn-ai/gensyn-delphi-sdk
 * DelphiClient, dispatches the method call, and writes the JSON-RPC response
 * back over stdout.
 *
 * Why a bridge instead of reimplementing the SDK in Python:
 *   - The SDK is the official integration surface (DoraHacks expects agents to
 *     use it).
 *   - It handles LMSR math, gateway routing, ERC-20 approvals, CDP wallet
 *     signing, and EIP-1193 signatures — reimplementing all of that in Python
 *     would be error-prone and drift from upstream.
 *   - Spawn overhead is one-time (we keep the process alive for the life of
 *     the Python adapter), and per-call latency is dominated by the on-chain
 *     RPC round-trip anyway.
 *
 * Protocol:
 *   Request:  {"jsonrpc":"2.0","id":<int>,"method":<string>,"params":<object>}
 *   Response: {"jsonrpc":"2.0","id":<int>,"result":<any>}
 *             or
 *             {"jsonrpc":"2.0","id":<int>,"error":{"code":<-int>,"message":<string>,"data":<any?>}}
 *
 *   BigInt fields (common in viem / SDK responses: transactionHash is hex string,
 *   balances/allowances are bigint) are serialized as strings to preserve
 *   precision across the JSON boundary. The Python side casts back as needed.
 *
 * Environment:
 *   The bridge inherits the parent process env. The SDK reads its config from
 *   DELPHI_NETWORK, DELPHI_API_ACCESS_KEY, DELPHI_SIGNER_TYPE,
 *   WALLET_PRIVATE_KEY (or CDP_*), etc. — see the SDK README for the full list.
 */

import { DelphiClient } from '@gensyn-ai/gensyn-delphi-sdk';
import { createPrivateKeySigner, createCdpSigner } from '@gensyn-ai/gensyn-delphi-sdk';

const client = new DelphiClient();

// ─── BigInt-safe JSON serialization ─────────────────────────────────
// viem returns bigint for all numeric on-chain values. JSON.stringify throws
// on bigint by default, so we install a custom replacer that converts
// bigint -> string with a type marker.
function serialize(value) {
  return JSON.stringify(value, (key, val) => {
    if (typeof val === 'bigint') {
      return { __type: 'bigint', value: val.toString() };
    }
    if (val instanceof Uint8Array) {
      return { __type: 'bytes', value: Buffer.from(val).toString('hex') };
    }
    return val;
  });
}

// ─── Method dispatch ────────────────────────────────────────────────
const METHODS = {
  // ─── Health & info ───────────────────────────────────────────────
  health: async () => client.health(),

  // ─── Market reads ────────────────────────────────────────────────
  listMarkets: async (params) => client.listMarkets(params || {}),
  getMarket: async (params) => client.getMarket(params),
  listPositions: async (params) => client.listPositions(params || {}),
  getMarketStatus: async (params) => client.getMarketStatus(params),

  // ─── Quotes (no on-chain tx, just simulation) ───────────────────
  quoteBuy: async (params) => client.quoteBuy(params),
  quoteSell: async (params) => client.quoteSell(params),
  quoteRedeem: async (params) => client.quoteRedeem(params),
  quoteLiquidate: async (params) => client.quoteLiquidate(params),

  // ─── Trading (on-chain writes) ───────────────────────────────────
  buyShares: async (params) => client.buyShares(params),
  sellShares: async (params) => client.sellShares(params),
  redeemMarket: async (params) => client.redeemMarket(params),
  redeemPositions: async (params) => client.redeemPositions(params),
  liquidate: async (params) => client.liquidate(params),

  // ─── Token / approval ────────────────────────────────────────────
  ensureTokenApproval: async (params) => client.ensureTokenApproval(params),
  approveToken: async (params) => client.approveToken(params),
  getTokenAllowance: async (params) => client.getTokenAllowance(params),

  // ─── Balance reads ───────────────────────────────────────────────
  getEthBalance: async () => client.getEthBalance(),
  getErc20Balance: async (params) => client.getErc20Balance(params?.tokenAddress),
  getErc20BalanceWithDecimals: async (params) =>
    client.getErc20BalanceWithDecimals(params?.tokenAddress),

  // ─── Gateway routing ─────────────────────────────────────────────
  resolveGateway: async (params) => client.resolveGateway(params),

  // ─── Raw subgraph access (passthrough) ───────────────────────────
  subgraphQuery: async (params) => {
    const subgraph = client.getSubgraph();
    return subgraph.query(params.query, params.variables);
  },
};

// ─── JSON-RPC dispatch ──────────────────────────────────────────────
async function handleRequest(req) {
  const { id, method, params } = req;
  if (id === undefined) {
    // Notification (no response expected)
    return null;
  }
  if (!Object.prototype.hasOwnProperty.call(METHODS, method)) {
    return {
      jsonrpc: '2.0',
      id,
      error: { code: -32601, message: `Method not found: ${method}` },
    };
  }
  try {
    const result = await METHODS[method](params);
    return { jsonrpc: '2.0', id, result: JSON.parse(serialize(result)) };
  } catch (err) {
    return {
      jsonrpc: '2.0',
      id,
      error: {
        code: -32000,
        message: err.message || String(err),
        data: {
          name: err.name,
          // SDK errors often include useful details
          ...(err.details ? { details: err.details } : {}),
          ...(err.shortMessage ? { shortMessage: err.shortMessage } : {}),
          ...(err.cause ? { cause: String(err.cause) } : {}),
        },
      },
    };
  }
}

// ─── Main loop: read line-delimited JSON from stdin ─────────────────
process.stdin.setEncoding('utf8');

let buffer = '';
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  // Split on newlines; keep the partial last line in the buffer
  let idx;
  while ((idx = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (!line) continue;
    try {
      const req = JSON.parse(line);
      handleRequest(req).then((resp) => {
        if (resp !== null) {
          process.stdout.write(JSON.stringify(resp) + '\n');
        }
      });
    } catch (err) {
      // Malformed JSON — send a parse-error response if we can recover an id
      process.stderr.write(`[bridge] JSON parse error: ${err.message}\n`);
      process.stderr.write(`[bridge] offending line: ${line.slice(0, 200)}\n`);
    }
  }
});

process.stdin.on('end', () => {
  process.stderr.write('[bridge] stdin closed, exiting\n');
  process.exit(0);
});

process.on('uncaughtException', (err) => {
  process.stderr.write(`[bridge] uncaughtException: ${err.stack || err}\n`);
});

process.on('unhandledRejection', (err) => {
  process.stderr.write(`[bridge] unhandledRejection: ${err?.stack || err}\n`);
});

// Signal readiness
process.stderr.write('[bridge] ready\n');
