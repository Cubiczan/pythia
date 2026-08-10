"""Configuration loader for the Delphi adapter.

The SDK reads its own configuration from environment variables — see the
``DelphiClientConfig`` interface in
``node_modules/@gensyn-ai/gensyn-delphi-sdk/dist/types/index.d.ts``. The
Python adapter does NOT duplicate that config; instead, this module resolves
the env vars the SDK needs and returns them as a ``DelphiEnv`` dict that
callers can pass to ``DelphiClient(env=...)`` or set in ``os.environ``
before the bridge starts.

Supported env vars (see the SDK README for the full list):

    DELPHI_NETWORK                  "testnet" | "mainnet" | "competition-testnet"
    DELPHI_API_ACCESS_KEY           REST API key (from https://delphi-api-access.gensyn.ai/)
    DELPHI_SIGNER_TYPE              "cdp_server_wallet" (default) | "private_key"

    WALLET_PRIVATE_KEY              0x... (only for private_key signing)

    CDP_API_KEY_ID                  (only for cdp_server_wallet signing)
    CDP_API_KEY_SECRET
    CDP_WALLET_SECRET
    CDP_WALLET_ADDRESS
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover
    import tomli as _toml

from pydantic import BaseModel, Field, field_validator


class DelphiEnv(BaseModel):
    """Resolved environment variables for the Delphi SDK bridge.

    Pass this to ``DelphiClient(env=delphi_env.to_env_dict())`` or merge
    into ``os.environ`` before the bridge starts.
    """

    network: str = Field("testnet", description="testnet | mainnet | competition-testnet")
    api_access_key: str = Field("", description="REST API key (never logged)")
    signer_type: str = Field("cdp_server_wallet", description="cdp_server_wallet | private_key")
    wallet_private_key: str = Field("", description="0x... hex private key (private_key signing only)")
    cdp_api_key_id: str = Field("")
    cdp_api_key_secret: str = Field("")
    cdp_wallet_secret: str = Field("")
    cdp_wallet_address: str = Field("")
    poll_interval_sec: int = Field(60, ge=1, description="Seconds between settlement polls")

    @field_validator("network")
    @classmethod
    def _valid_network(cls, v: str) -> str:
        valid = {"testnet", "mainnet", "competition-testnet"}
        if v not in valid:
            raise ValueError(f"network must be one of {valid}, got {v!r}")
        return v

    @field_validator("signer_type")
    @classmethod
    def _valid_signer(cls, v: str) -> str:
        valid = {"cdp_server_wallet", "private_key"}
        if v not in valid:
            raise ValueError(f"signer_type must be one of {valid}, got {v!r}")
        return v

    def to_env_dict(self) -> dict[str, str]:
        """Convert to a dict of env-var-name -> string-value, ready for ``os.environ``."""
        env: dict[str, str] = {
            "DELPHI_NETWORK": self.network,
            "DELPHI_SIGNER_TYPE": self.signer_type,
        }
        if self.api_access_key:
            env["DELPHI_API_ACCESS_KEY"] = self.api_access_key
        if self.wallet_private_key:
            env["WALLET_PRIVATE_KEY"] = self.wallet_private_key
        if self.cdp_api_key_id:
            env["CDP_API_KEY_ID"] = self.cdp_api_key_id
        if self.cdp_api_key_secret:
            env["CDP_API_KEY_SECRET"] = self.cdp_api_key_secret
        if self.cdp_wallet_secret:
            env["CDP_WALLET_SECRET"] = self.cdp_wallet_secret
        if self.cdp_wallet_address:
            env["CDP_WALLET_ADDRESS"] = self.cdp_wallet_address
        return env


def load_config(
    *,
    toml_path: str | None = None,
    require_key: bool = True,
) -> DelphiEnv:
    """Build a ``DelphiEnv`` from env vars + optional TOML config.

    Resolution order for each field:
      1. TOML ``api_key_env`` indirection (look up the named env var)
      2. TOML literal value
      3. Standard SDK env var (``DELPHI_API_ACCESS_KEY`` etc.)

    Parameters
    ----------
    toml_path:
        Optional path to a TOML config file. If the file doesn't exist,
        falls back to env vars only.
    require_key:
        If True (default), raise ValueError when no API key is resolvable.
        Set to False for read-only market listing (which still needs a key
        in production, but tests may want to construct an env without one).
    """
    toml_data: dict[str, Any] = {}
    if toml_path is not None:
        path = Path(toml_path)
        if path.is_file():
            with path.open("rb") as f:
                toml_data = _toml.load(f)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "TOML config %s does not exist; falling back to env vars only.",
                toml_path,
            )

    # Network
    network = (
        toml_data.get("network")
        or toml_data.get("delphi", {}).get("network")
        or os.environ.get("DELPHI_NETWORK")
        or "testnet"
    )

    # API key: TOML api_key_env indirection > TOML api_access_key > env var
    api_key = ""
    api_key_env_name = (
        toml_data.get("api_key_env")
        or toml_data.get("delphi", {}).get("api_key_env")
    )
    if api_key_env_name:
        api_key = os.environ.get(str(api_key_env_name), "")
    if not api_key:
        api_key = (
            toml_data.get("api_access_key")
            or toml_data.get("delphi", {}).get("api_access_key")
            or os.environ.get("DELPHI_API_ACCESS_KEY")
            or ""
        )

    # Signer type
    signer_type = (
        toml_data.get("signer_type")
        or toml_data.get("delphi", {}).get("signer_type")
        or os.environ.get("DELPHI_SIGNER_TYPE")
        or "cdp_server_wallet"
    )

    # Private key
    wallet_private_key = os.environ.get("WALLET_PRIVATE_KEY", "")

    # CDP creds
    cdp_api_key_id = os.environ.get("CDP_API_KEY_ID", "")
    cdp_api_key_secret = os.environ.get("CDP_API_KEY_SECRET", "")
    cdp_wallet_secret = os.environ.get("CDP_WALLET_SECRET", "")
    cdp_wallet_address = os.environ.get("CDP_WALLET_ADDRESS", "")

    # Poll interval
    poll_raw = (
        toml_data.get("poll_interval_sec")
        or toml_data.get("delphi", {}).get("poll_interval_sec")
        or os.environ.get("DELPHI_POLL_INTERVAL_SEC")
        or 60
    )
    try:
        poll_interval_sec = int(poll_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"poll_interval_sec must be an integer, got {poll_raw!r}"
        ) from exc

    if require_key and not api_key:
        raise ValueError(
            "No Delphi API key found. Set DELPHI_API_ACCESS_KEY or provide "
            "api_key_env / api_access_key in the TOML config."
        )

    return DelphiEnv(
        network=network,
        api_access_key=api_key,
        signer_type=signer_type,
        wallet_private_key=wallet_private_key,
        cdp_api_key_id=cdp_api_key_id,
        cdp_api_key_secret=cdp_api_key_secret,
        cdp_wallet_secret=cdp_wallet_secret,
        cdp_wallet_address=cdp_wallet_address,
        poll_interval_sec=poll_interval_sec,
    )


__all__ = ["DelphiEnv", "load_config"]
