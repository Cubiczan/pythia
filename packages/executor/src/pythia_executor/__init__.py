"""pythia-executor — trade-orchestration CLI for the Pythia mesh.

Wraps icohangar-ops/metabocommand and stitches the four sibling Pythia
sub-repos into a single end-to-end pipeline:

    Delphi market → analyst-mesh → consensus → risk → sign → SDK buy_shares → TradeReceipt
                                                                                │
                                                                                ▼
                                                                       signed JSONL audit log

Public API
----------
- ``PythiaExecutor``  — the orchestrator class with ``run_for_market``.
- ``ExecutorConfig``  — pydantic config model (mode, signing key env, retries).
- ``run_pipeline``    — convenience: build a default executor from env + run one market.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .pipeline import PythiaExecutor
from .types import ExecutorConfig, ExecutorMode, PipelineResult

__version__ = "0.1.0"

__all__ = [
    "ExecutorConfig",
    "ExecutorMode",
    "PipelineResult",
    "PythiaExecutor",
    "__version__",
    "run_pipeline",
]

async def run_pipeline(
    market_id: str,
    *,
    mode: ExecutorMode = "paper",
    analysts: list[str] | None = None,
    consensus_threshold: float = 0.65,
    max_stake_usd: float = 50.0,
    audit_log_path: str | Path = "./logs/audit.jsonl",
    config_path: str | None = None,
) -> PipelineResult:
    """Convenience: build a default executor from env + TOML and run one market.

    This is the shortest path from "I have a market id" to "I have a
    PipelineResult". It mirrors what the ``pythia executor delphi paper-trade``
    CLI does — useful for scripts, notebooks, and tests.

    Parameters
    ----------
    market_id:
        Delphi market to evaluate.
    mode:
        ``"paper"`` (default) or ``"live"``. Live mode requires
        ``DELPHI_SIGNING_KEY``.
    analysts:
        List of analyst slugs. Default ``["politics", "crypto"]``.
    consensus_threshold:
        ``agreement_score`` cutoff for ``gate="trade"``. Default 0.65.
    max_stake_usd:
        Per-market stake cap. Default 50.0.
    audit_log_path:
        Where to append the JSONL audit record. Default ``./logs/audit.jsonl``.
    config_path:
        Optional path to a TOML config (e.g. ``configs/live-mvp.toml``).
        If provided, settings from the TOML override the defaults above.
    """
    from pythia_analyst_mesh import AnalystRegistry, LLMConfig
    from pythia_consensus import ConsensusConfig, ConsensusEngine
    from pythia_delphi_adapter import DelphiClient
    from pythia_risk import MarketTypeRules, RiskConfig, RiskEngine

    # Load TOML if provided.
    toml_cfg: dict[str, Any] = {}
    if config_path is not None:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as f:
            toml_cfg = tomllib.load(f)

    delphi_cfg = toml_cfg.get("delphi", {})
    mesh_cfg = toml_cfg.get("mesh", {})

    api_key = os.environ.get(
        delphi_cfg.get("api_key_env", "DELPHI_API_ACCESS_KEY"), ""
    )

    delphi_client = DelphiClient()
    try:
        analyst_slugs = analysts or mesh_cfg.get("analysts", ["politics", "crypto"])
        llm_config = LLMConfig(
            provider=mesh_cfg.get("llm_provider", "openai"),
            model=mesh_cfg.get("llm_model", "gpt-4o-mini"),
            api_key=os.environ.get(
                mesh_cfg.get("llm_api_key_env", "LLM_API_KEY"), ""
            ),
            temperature=float(mesh_cfg.get("llm_temperature", 0.2)),
            max_tokens=int(mesh_cfg.get("llm_max_tokens", 800)),
        )
        registry = AnalystRegistry()
        mesh = registry.build_mesh(analyst_slugs, llm_config)

        consensus_engine = ConsensusEngine(
            ConsensusConfig(
                method="logit-mean",
                agreement_threshold=consensus_threshold,
                min_analysts=2,
            )
        )

        risk_engine = RiskEngine(
            RiskConfig(
                sizing="kelly-fractional",
                kelly_fraction=0.25,
                max_stake_per_market_usd=max_stake_usd,
                max_total_exposure_usd=500.0,
                max_drawdown_pct=5.0,
                cool_down_min_after_loss=30.0,
                market_type_rules={
                    "politics": MarketTypeRules(max_stake_usd=max_stake_usd, allowed=True),
                    "crypto": MarketTypeRules(max_stake_usd=max_stake_usd, allowed=True),
                    "sports": MarketTypeRules(max_stake_usd=max_stake_usd, allowed=True),
                    "niche": MarketTypeRules(max_stake_usd=max_stake_usd, allowed=True),
                    "subjective": MarketTypeRules(max_stake_usd=max_stake_usd, allowed=True),
                },
            )
        )

        executor = PythiaExecutor(
            delphi_client=delphi_client,
            mesh=mesh,
            consensus_engine=consensus_engine,
            risk_engine=risk_engine,
            config=ExecutorConfig(
                mode=mode,
                signing_key_env="DELPHI_SIGNING_KEY",
                idempotency_enabled=True,
                retry_max=3,
                retry_backoff_sec=5,
            ),
            audit_log_path=Path(audit_log_path),
        )

        return await executor.run_for_market(market_id)
    finally:
        await delphi_client.stop()
