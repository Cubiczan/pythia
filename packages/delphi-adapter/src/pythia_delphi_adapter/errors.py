"""Exception types for the Pythia Delphi adapter.

These are the errors the adapter can raise. They are intentionally narrow
so the executor and observability layer can pattern-match on type rather
than parsing error strings.
"""

from __future__ import annotations

from typing import Any


class DelphiAdapterError(Exception):
    """Base class for all adapter errors."""


class BridgeError(DelphiAdapterError):
    """The Node bridge subprocess failed to start or crashed."""


class BridgeNotReadyError(BridgeError):
    """A call was made before ``Bridge.start()`` completed, or after the
    bridge process has exited."""


class DelphiAPIError(DelphiAdapterError):
    """The SDK raised an error while executing a method.

    Mirrors the JSON-RPC error object: ``code`` is the numeric error code
    (negative for JSON-RPC protocol errors, -32000 for SDK errors), and
    ``data`` carries extra SDK detail (``shortMessage``, ``details``,
    ``cause``) when available.
    """

    def __init__(
        self,
        *,
        message: str,
        code: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data

    def __str__(self) -> str:
        parts = [self.message]
        if self.code is not None:
            parts.append(f"code={self.code}")
        if self.data:
            if "shortMessage" in self.data:
                parts.append(f"short={self.data['shortMessage']}")
            if "details" in self.data:
                parts.append(f"details={self.data['details']}")
        return " | ".join(parts)


class MarketNotFoundError(DelphiAPIError):
    """A market with the given address/ID was not found on the active network."""


__all__ = [
    "DelphiAdapterError",
    "BridgeError",
    "BridgeNotReadyError",
    "DelphiAPIError",
    "MarketNotFoundError",
]
