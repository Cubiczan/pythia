"""Audit signing integration for pythia-consensus.

This module is the **only** place that talks to the upstream
`consensus-hardening-protocol`. It is intentionally tiny and isolated: if the
upstream's API changes, only this file needs updating.

Strategy
--------
We `try: import consensus_hardening_protocol`. If it succeeds, we use the
upstream's `SignedDecision` to sign each `ConsensusDecision`. If it fails
(upstream not vendored), we fall back to a deterministic SHA-256 stub so the
rest of the Pythia mesh can still operate — every audit record is clearly
prefixed with ``stub:`` so downstream consumers (and the replay UI) know it
is NOT cryptographically signed.

# VERIFY: the exact upstream API for `SignedDecision` needs to be confirmed
# once the upstream is vendored. The two most plausible shapes are:
#
#   (a) SignedDecision(payload: dict, key: SigningKey) -> str   # returns sig
#   (b) SignedDecision.sign(payload: dict) -> SignedDecision    # classmethod
#
# We try (a) first, then (b). Both are wrapped in try/except so a mismatch
# degrades to the stub instead of crashing the mesh.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .types import ConsensusDecision

# Whether the upstream is importable. Set once at import time.
try:
    from consensus_hardening_protocol import SignedDecision as _UpstreamSignedDecision  # type: ignore[import-not-found]

    _UPSTREAM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when upstream not vendored
    _UpstreamSignedDecision = None  # type: ignore[assignment, misc]
    _UPSTREAM_AVAILABLE = False

def _canonical_payload(decision: ConsensusDecision) -> dict[str, Any]:
    """Canonical, sorted-key dict representation used for hashing / signing.

    Excludes the `timestamp` field from the signature scope? No — we *include*
    it. The timestamp is part of the audit record. If you need to re-sign the
    same logical decision at a different time, you get a different signature,
    which is correct (it is a different audit event).
    """
    # `model_dump(mode="json")` gives us JSON-native types and stable ordering
    # for nested dicts. We then re-serialise with sort_keys for determinism.
    payload = decision.model_dump(mode="json")
    return payload  # type: ignore[return-value]

def _stub_signature(decision: ConsensusDecision) -> str:
    """Deterministic SHA-256 over the canonical JSON of the decision.

    Returns a string prefixed with ``stub:`` so consumers can tell this is
    not a cryptographic signature.
    """
    payload = _canonical_payload(decision)
    # sort_keys + default separators for a stable, compact canonical form.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"stub:sha256:{digest}"

class AuditSigner:
    """Signs `ConsensusDecision` records for the audit trail.

    Delegates to `consensus_hardening_protocol.SignedDecision` when the
    upstream is vendored; otherwise falls back to a SHA-256 stub.

    Parameters
    ----------
    signing_key:
        Optional signing key passed through to the upstream signer. Ignored
        by the stub path. The Pythia mesh typically loads this from
        `DELPHI_SIGNING_KEY` at the executor layer; the consensus signer
        itself is key-agnostic.
    """

    def __init__(self, signing_key: Any | None = None) -> None:
        self._signing_key = signing_key
        self._upstream_available = _UPSTREAM_AVAILABLE

    @property
    def upstream_available(self) -> bool:
        """True if the upstream `consensus_hardening_protocol` is importable."""
        return self._upstream_available

    def sign(self, decision: ConsensusDecision) -> str:
        """Return a signature string for `decision`.

        - Upstream available: delegates to `SignedDecision`.
        - Upstream not available: returns ``stub:sha256:<hex>``.

        The returned string is what gets persisted to the audit log alongside
        the decision payload.
        """
        if not self._upstream_available or _UpstreamSignedDecision is None:
            return _stub_signature(decision)

        payload = _canonical_payload(decision)

        # VERIFY: try constructor shape (a) — `SignedDecision(payload, key)`
        try:
            signed = _UpstreamSignedDecision(payload, self._signing_key)  # type: ignore[misc]
            sig_attr = getattr(signed, "signature", None)
            if callable(sig_attr):
                return str(sig_attr())
            if sig_attr is not None:
                return str(sig_attr)
            return str(signed)
        except TypeError:
            pass
        except Exception:  # pragma: no cover - defensive
            pass

        # VERIFY: try classmethod shape (b) — `SignedDecision.sign(payload)`
        sign_method = getattr(_UpstreamSignedDecision, "sign", None)
        if callable(sign_method):
            try:
                signed = sign_method(payload)  # type: ignore[misc]
                sig_attr = getattr(signed, "signature", None)
                if callable(sig_attr):
                    return str(sig_attr())
                if sig_attr is not None:
                    return str(sig_attr)
                return str(signed)
            except Exception:  # pragma: no cover - defensive
                pass

        # Upstream importable but neither shape worked — fall back to stub,
        # but flag it so the audit log is honest about provenance.
        return _stub_signature(decision)

__all__ = ["AuditSigner"]
