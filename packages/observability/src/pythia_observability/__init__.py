"""pythia-observability — replay UI + achievements layer for the Pythia mesh.

Wraps `icohangar-ops/agent-observability` (audit trail) and
`icohangar-ops/achievements` (milestone tracking), and adds:

- a JSONL audit-log reader (`AuditLogReader`) with derived analytics
  (cumulative P&L curve, per-analyst Brier scores, aggregate stats),
- a FastAPI replay server (`ReplayServer`) that surfaces the audit log +
  achievements to a polished dark-themed dashboard UI,
- an achievements evaluator (`AchievementsEvaluator`) that runs the upstream
  achievements.toml conditions against the audit log.

Public API
----------
- `AuditLogReader`          — read + slice + analyse the JSONL audit log.
- `ReplayServer`            — FastAPI app + `run()` for the dashboard UI.
- `AchievementsEvaluator`   — load achievements.toml + evaluate every condition.
- `AuditEntry`              — pydantic model, one audit-log record.
- `PnLMilestone`            — pydantic model, one point on the P&L curve.
- `Achievement`             — pydantic model, one milestone (with unlocked state).
- `AchievementCondition`    — pydantic model, one evaluable condition inside an Achievement.

Run the dashboard:

    pythia-replay serve --log ./logs/audit.jsonl \\
        --achievements-config configs/achievements.toml --port 8088

See `pythia_observability.cli` for the full CLI surface.
"""

from __future__ import annotations

from .achievements import AchievementsEvaluator
from .audit_reader import AuditLogReader
from .server import ReplayServer
from .types import Achievement, AchievementCondition, AuditEntry, PnLMilestone

__version__ = "0.1.0"

__all__ = [
    "Achievement",
    "AchievementCondition",
    "AchievementsEvaluator",
    "AuditEntry",
    "AuditLogReader",
    "PnLMilestone",
    "ReplayServer",
    "__version__",
]
