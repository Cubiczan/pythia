"""Node subprocess bridge for the @gensyn-ai/gensyn-delphi-sdk.

The official Gensyn Delphi SDK is a TypeScript ESM package
(``@gensyn-ai/gensyn-delphi-sdk`` on npm). Rather than reimplement its LMSR
math, gateway routing, ERC-20 approval flow, and signer logic in Python, we
spawn a long-running Node.js subprocess (``bridge.mjs``) that loads the SDK
and dispatches JSON-RPC 2.0 calls on our behalf.

Lifecycle:
    - ``Bridge.start()`` spawns the node process and waits for the
      ``[bridge] ready`` line on stderr.
    - ``Bridge.call(method, params)`` sends a JSON-RPC request over stdin
      and awaits the matching response on stdout.
    - ``Bridge.stop()`` sends EOF to stdin and waits for the process to exit.

The bridge is intentionally stateless beyond the subprocess handle — all
SDK state (client config, signer, gateway cache) lives inside the Node
process. The Python side only needs to know the method name and params.

BigInt handling:
    The SDK uses ``bigint`` for all on-chain numeric values (balances,
    allowances, share counts, token amounts). The bridge serializes
    ``bigint`` as ``{"__type":"bigint","value":"<decimal-string>"}`` so
    precision survives the JSON boundary. The Python side treats these as
    plain strings — Python's ``int`` is unbounded but downstream consumers
    (Pythia risk engine, audit log) prefer the canonical string form.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from .errors import BridgeError, BridgeNotReadyError, DelphiAPIError

# Path to the bridge.mjs script, bundled inside this package.
BRIDGE_SCRIPT = Path(__file__).parent / "bridge.mjs"

# How long to wait for the bridge to print "[bridge] ready" on stderr.
BRIDGE_READY_TIMEOUT_SEC = 15.0

# How long to wait for a single JSON-RPC response by default.
DEFAULT_CALL_TIMEOUT_SEC = 60.0


class Bridge:
    """Manages a long-running Node.js subprocess running bridge.mjs.

    A single ``Bridge`` instance should be shared across all SDK calls for
    the lifetime of the adapter — the Node process caches the signer, the
    gateway routing table, and the HTTP connection pool, so re-spawning per
    call would waste ~200ms each time.

    The bridge is safe to use from a single asyncio event loop. For
    multi-loop or threaded use, create one ``Bridge`` per loop.
    """

    def __init__(
        self,
        *,
        node_bin: str | None = None,
        bridge_script: Path | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        ready_timeout: float = BRIDGE_READY_TIMEOUT_SEC,
        default_call_timeout: float = DEFAULT_CALL_TIMEOUT_SEC,
        log_stderr: bool = False,
    ) -> None:
        self._node_bin = node_bin or os.environ.get("PYTHIA_NODE_BIN") or self._find_node()
        self._bridge_script = bridge_script or BRIDGE_SCRIPT
        self._cwd = cwd or self._resolve_cwd()
        self._env = {**os.environ, **(env or {})}
        self._ready_timeout = ready_timeout
        self._default_call_timeout = default_call_timeout
        self._log_stderr = log_stderr

        self._proc: asyncio.subprocess.Process | None = None
        self._next_id: int = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._stderr_buffer: list[str] = []
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the Node bridge and wait for the ``[bridge] ready`` signal."""
        if self._proc is not None and self._proc.returncode is None:
            return  # already running

        if not self._bridge_script.exists():
            raise BridgeError(
                f"Bridge script not found at {self._bridge_script}. "
                "The pythia-delphi-adapter package may be installed incorrectly."
            )

        self._proc = await asyncio.create_subprocess_exec(
            self._node_bin,
            str(self._bridge_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd),
            env=self._env,
        )

        self._stdout_task = asyncio.create_task(
            self._read_stdout(), name="bridge-stdout"
        )
        self._stderr_task = asyncio.create_task(
            self._read_stderr(), name="bridge-stderr"
        )

        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self._ready_timeout)
        except TimeoutError as exc:
            await self._kill()
            tail = "\n".join(self._stderr_buffer[-20:])
            raise BridgeError(
                f"Bridge did not signal ready within {self._ready_timeout}s. "
                f"Stderr tail:\n{tail}"
            ) from exc

    async def stop(self) -> None:
        """Close stdin and wait for the bridge to exit gracefully."""
        if self._proc is None:
            return
        self._closed = True
        if self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.close()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=10.0)
        except TimeoutError:
            await self._kill()

    async def _kill(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass
        try:
            await self._proc.wait()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # JSON-RPC call
    # ------------------------------------------------------------------

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Send a JSON-RPC 2.0 request and await the response.

        Raises:
            BridgeNotReadyError: if ``start()`` hasn't been called or the
                process has exited.
            DelphiAPIError: if the SDK method raised an error. The error
                message includes the SDK's shortMessage / details when
                available.
            asyncio.TimeoutError: if no response arrives within ``timeout``.
        """
        if self._proc is None or self._proc.returncode is not None:
            raise BridgeNotReadyError(
                "Bridge process is not running. Call await bridge.start() first."
            )
        if self._proc.stdin is None or self._proc.stdin.is_closing():
            raise BridgeNotReadyError("Bridge stdin is closed.")

        call_id = self._next_id
        self._next_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": method,
            "params": params or {},
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[call_id] = future

        line = _json_dumps(request) + "\n"
        assert self._proc.stdin is not None
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

        effective_timeout = timeout if timeout is not None else self._default_call_timeout
        try:
            return await asyncio.wait_for(future, timeout=effective_timeout)
        except TimeoutError:
            self._pending.pop(call_id, None)
            raise
        except DelphiAPIError:
            self._pending.pop(call_id, None)
            raise

    # ------------------------------------------------------------------
    # Stream readers
    # ------------------------------------------------------------------

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            try:
                msg = _json_loads(line.decode("utf-8").strip())
            except Exception:
                continue
            if not isinstance(msg, dict) or "jsonrpc" not in msg:
                continue
            call_id = msg.get("id")
            if call_id is None:
                continue
            future = self._pending.pop(call_id, None)
            if future is None or future.done():
                continue
            if "error" in msg and msg["error"] is not None:
                err = msg["error"]
                future.set_exception(
                    DelphiAPIError(
                        message=err.get("message", "Unknown SDK error"),
                        code=err.get("code"),
                        data=err.get("data"),
                    )
                )
            else:
                future.set_result(msg.get("result"))

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            self._stderr_buffer.append(text)
            if self._log_stderr:
                print(f"[bridge-stderr] {text}", file=sys.stderr)
            if "[bridge] ready" in text:
                self._ready.set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_node() -> str:
        """Find the node executable on PATH."""
        import shutil
        for candidate in ("node", "nodejs"):
            path = shutil.which(candidate)
            if path:
                return path
        # Fall back to plain "node" — the caller will get a clear error
        # if it's missing when the subprocess spawn fails.
        return "node"

    @staticmethod
    def _resolve_cwd() -> Path:
        """Resolve the working directory for the bridge process.

        The bridge needs to find ``node_modules/@gensyn-ai/gensyn-delphi-sdk``,
        which lives at the repo root (two levels up from this file).
        """
        # This file is at packages/delphi-adapter/src/pythia_delphi_adapter/bridge.py
        # Repo root is 4 levels up.
        repo_root = Path(__file__).resolve().parents[4]
        if (repo_root / "node_modules" / "@gensyn-ai" / "gensyn-delphi-sdk").exists():
            return repo_root
        # Fall back to the package directory (if installed standalone with
        # node_modules there).
        pkg_root = Path(__file__).resolve().parents[2]
        if (pkg_root / "node_modules" / "@gensyn-ai" / "gensyn-delphi-sdk").exists():
            return pkg_root
        # Last resort: current working directory
        return Path.cwd()


# ---------------------------------------------------------------------------
# JSON helpers (tolerant of trailing whitespace / newlines)
# ---------------------------------------------------------------------------

def _json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"), default=str)


def _json_loads(s: str) -> Any:
    import json
    return json.loads(s)


__all__ = ["Bridge", "BRIDGE_SCRIPT", "BRIDGE_READY_TIMEOUT_SEC", "DEFAULT_CALL_TIMEOUT_SEC"]
