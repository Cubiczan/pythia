"""Configuration loader for the Delphi adapter.

Loads ``DelphiConfig`` from, in order of precedence:

1. An explicit TOML file path passed to ``load_config(toml_path=...)``.
2. Environment variables (``DELPHI_API_KEY``, ``DELPHI_ENDPOINT``,
   ``DELPHI_POLL_INTERVAL_SEC``).

The TOML file may use *either* a literal ``api_key`` field or the
``api_key_env`` indirection used in the parent Pythia monorepo's
``configs/live-mvp.toml``. The indirection form is preferred — it keeps
the secret out of git.

ATT docs: https://docs.gensyn.ai/tech/agentic-trading
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover — pre-3.11 fallback
    import tomli as _toml

from pydantic import BaseModel, Field, field_validator

from pythia_delphi_adapter.client import DEFAULT_ENDPOINT


class DelphiConfig(BaseModel):
    """Resolved configuration for ``DelphiClient`` + ``SettlementListener``."""

    api_key: str = Field(..., description="ATT API key (never logged).")
    endpoint: str = Field(
        DEFAULT_ENDPOINT,
        description="ATT base URL. Defaults to the public Gensyn endpoint.",
    )
    poll_interval_sec: int = Field(
        60,
        ge=1,
        description="Seconds between settlement polls.",
    )

    @field_validator("api_key")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("api_key must not be empty")
        return v.strip()


def load_config(
    env_var: str = "DELPHI_API_KEY",
    toml_path: str | None = None,
) -> DelphiConfig:
    """Build a ``DelphiConfig`` from env vars + optional TOML.

    Resolution order for each field:

    1. If ``toml_path`` is given and the field is present in the TOML,
       use the TOML value (with ``api_key_env`` indirection resolved
       against the environment).
    2. Otherwise fall back to the standard env vars:
       ``DELPHI_API_KEY``, ``DELPHI_ENDPOINT``, ``DELPHI_POLL_INTERVAL_SEC``.

    Parameters
    ----------
    env_var:
        Name of the environment variable holding the ATT API key. Defaults
        to ``DELPHI_API_KEY``.
    toml_path:
        Optional path to a TOML config file. If the file does not exist,
        a warning is logged and env vars are used.

    Raises
    ------
    ValueError
        If no API key is resolvable from any source.
    """
    toml_data: dict[str, Any] = {}
    if toml_path is not None:
        path = Path(toml_path)
        if path.is_file():
            with path.open("rb") as f:
                toml_data = _toml.load(f)
        else:
            # Don't hard-fail — env vars might still resolve everything.
            import logging

            logging.getLogger(__name__).warning(
                "TOML config %s does not exist; falling back to env vars only.",
                toml_path,
            )

    # Resolve API key: TOML ``api_key_env`` > TOML ``api_key`` > env var.
    api_key = ""
    if "api_key_env" in toml_data:
        api_key = os.environ.get(str(toml_data["api_key_env"]), "")
    if not api_key and "api_key" in toml_data:
        api_key = str(toml_data["api_key"])
    if not api_key:
        api_key = os.environ.get(env_var, "")

    # Resolve endpoint.
    endpoint = (
        toml_data.get("endpoint")
        or os.environ.get("DELPHI_ENDPOINT")
        or DEFAULT_ENDPOINT
    )

    # Resolve poll interval.
    poll_raw = (
        toml_data.get("poll_interval_sec")
        or os.environ.get("DELPHI_POLL_INTERVAL_SEC")
        or 60
    )
    try:
        poll_interval_sec = int(poll_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"poll_interval_sec must be an integer, got {poll_raw!r}"
        ) from exc

    if not api_key:
        raise ValueError(
            f"No Delphi API key found. Set ${env_var} or provide "
            f"api_key / api_key_env in {toml_path or '<toml_path>'}."
        )

    return DelphiConfig(
        api_key=api_key,
        endpoint=str(endpoint),
        poll_interval_sec=poll_interval_sec,
    )


__all__ = ["DelphiConfig", "load_config"]
