"""FastAPI replay server — surfaces the audit log + achievements to judges.

The server is intentionally *read-only*. It never writes to the audit log; it
only parses it on each request (the log is append-only and small enough that
re-reading per request is fine for typical workloads; for production use we'd
cache + invalidate on file mtime).

Endpoints
---------
- `GET /`                  — HTML dashboard (Jinja2 template, dark oracle theme)
- `GET /api/stats`         — aggregate stats (P&L, win rate, bankroll, drawdown)
- `GET /api/trades`        — paginated list of recent audit entries (newest first)
- `GET /api/trades/{id}`   — full decision chain for one market_id
- `GET /api/pnl-series`    — list[PnLMilestone] for the P&L chart
- `GET /api/achievements`  — list[Achievement] with unlocked status

Run via the CLI (`pythia-replay serve --log ./logs/audit.jsonl`) or directly:

    uvicorn pythia_observability.server:make_app --factory
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .achievements import AchievementsEvaluator
from .audit_reader import AuditLogReader
from .types import Achievement, AuditEntry, PnLMilestone

_log = logging.getLogger(__name__)

# Templates ship inside the package — discover via __file__ so the server
# works regardless of the install location / editable vs. wheel.
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    enable_async=False,
)

class ReplayServer:
    """Holds the audit log path + optional achievements config and exposes
    a configured FastAPI app.

    Parameters
    ----------
    log_path:
        Path to the audit JSONL.
    achievements_config_path:
        Optional path to an achievements.toml. If provided, `/api/achievements`
        evaluates against the log; if not, the endpoint returns an empty list
        (the dashboard still renders, just without the achievements grid).
    """

    def __init__(
        self,
        log_path: Path,
        achievements_config_path: Path | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.achievements_config_path = (
            Path(achievements_config_path) if achievements_config_path else None
        )

    # ------------------------------------------------------------------ #
    # Reader / evaluator helpers.
    # ------------------------------------------------------------------ #

    def _reader(self) -> AuditLogReader:
        """Fresh reader per request. The reader caches per-instance, so we
        could keep one alive — but a fresh reader per request picks up newly
        appended audit lines without manual cache invalidation. Good for the
        demo (judges can watch trades land live)."""
        return AuditLogReader(self.log_path)

    def _evaluator(self) -> AchievementsEvaluator | None:
        if self.achievements_config_path is None:
            return None
        try:
            return AchievementsEvaluator(self.achievements_config_path)
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------ #
    # App factory.
    # ------------------------------------------------------------------ #

    def app(self) -> FastAPI:
        """Build a fresh FastAPI app with all routes wired up."""
        application = FastAPI(
            title="Pythia Replay",
            description="Replay UI + achievements for the Pythia multi-agent trading mesh.",
            version="0.1.0",
        )
        self._register_routes(application)
        return application

    def _register_routes(self, application: FastAPI) -> None:
        """Attach all routes to `application`."""
        log_path = self.log_path
        cfg_path = self.achievements_config_path

        # --- HTML dashboard ---------------------------------------------- #
        @application.get("/", response_class=HTMLResponse)
        def dashboard() -> HTMLResponse:
            """Render the dashboard HTML. Stats + trades are fetched client-side."""
            # We pass minimal context — everything else is fetched via /api/*.
            template = _JINJA_ENV.get_template("dashboard.html")
            html = template.render(
                log_path=str(log_path),
                achievements_config=str(cfg_path) if cfg_path else "(none)",
            )
            return HTMLResponse(content=html)

        # --- /api/stats -------------------------------------------------- #
        @application.get("/api/stats")
        def get_stats() -> dict[str, Any]:
            reader = self._reader()
            try:
                return reader.compute_stats()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        # --- /api/trades (paginated, newest first) ---------------------- #
        @application.get("/api/trades")
        def get_trades(
            limit: int = Query(default=25, ge=1, le=500),
            offset: int = Query(default=0, ge=0),
        ) -> dict[str, Any]:
            reader = self._reader()
            try:
                entries = reader.read_all()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            # Newest first — audit log is append-only, so reverse chronological.
            entries_sorted = sorted(entries, key=lambda e: e.timestamp, reverse=True)
            page = entries_sorted[offset : offset + limit]
            return {
                "total": len(entries),
                "limit": limit,
                "offset": offset,
                "trades": [self._entry_to_dict(e) for e in page],
            }

        # --- /api/trades/{market_id} ------------------------------------ #
        @application.get("/api/trades/{market_id}")
        def get_trade(market_id: str) -> dict[str, Any]:
            reader = self._reader()
            entries = reader.get_by_market(market_id)
            if not entries:
                raise HTTPException(
                    status_code=404,
                    detail=f"No audit entries for market_id={market_id!r}",
                )
            return {
                "market_id": market_id,
                "entries": [self._entry_to_dict(e) for e in entries],
            }

        # --- /api/pnl-series -------------------------------------------- #
        @application.get("/api/pnl-series")
        def get_pnl_series() -> list[dict[str, Any]]:
            reader = self._reader()
            milestones: list[PnLMilestone] = reader.compute_pnl_series()
            return [m.model_dump(mode="json") for m in milestones]

        # --- /api/achievements ------------------------------------------ #
        @application.get("/api/achievements")
        def get_achievements() -> list[dict[str, Any]]:
            evaluator = self._evaluator()
            if evaluator is None:
                return []
            reader = self._reader()
            unlocked: list[Achievement] = evaluator.evaluate(reader)
            return [a.model_dump(mode="json") for a in unlocked]

    # ------------------------------------------------------------------ #
    # Helpers.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _entry_to_dict(entry: AuditEntry) -> dict[str, Any]:
        """Serialise one AuditEntry for the API.

        Adds a few convenience fields the dashboard wants without having to
        compute them in JS (is_executed, is_skipped, won, realized_pnl_usd).
        """
        base = entry.model_dump(mode="json")
        base["is_executed"] = entry.is_executed
        base["is_skipped"] = entry.is_skipped
        base["is_paper"] = entry.is_paper
        base["won"] = entry.won
        base["realized_pnl_usd"] = entry.realized_pnl_usd
        base["category"] = entry.category
        return base

    # ------------------------------------------------------------------ #
    # Run.
    # ------------------------------------------------------------------ #

    def run(self, host: str = "127.0.0.1", port: int = 8088) -> None:
        """Start uvicorn with the FastAPI app. Blocks until interrupted."""
        # Import here so importing this module doesn't drag in uvicorn
        # transitively for callers that only want the app (e.g. tests).
        import uvicorn

        # We pass a factory so uvicorn rebuilds the app on reload — handy in
        # dev. In production, set reload=False.
        application = self.app()
        _log.info(
            "starting Pythia Replay server on http://%s:%d (log=%s, achievements=%s)",
            host,
            port,
            self.log_path,
            self.achievements_config_path,
        )
        uvicorn.run(application, host=host, port=port, log_level="info")

# Convenience module-level factory so `uvicorn ... --factory` works.
def make_app(
    log_path: Path | None = None,
    achievements_config_path: Path | None = None,
) -> FastAPI:
    """Module-level app factory.

    Reads `PYTHIA_AUDIT_LOG` and `PYTHIA_ACHIEVEMENTS_CONFIG` env vars if the
    args are not provided, so the server can be launched without code:

        PYTHIA_AUDIT_LOG=./logs/audit.jsonl \\
        PYTHIA_ACHIEVEMENTS_CONFIG=./configs/achievements.toml \\
        uvicorn pythia_observability.server:make_app --factory
    """
    import os

    if log_path is None:
        env_log = os.environ.get("PYTHIA_AUDIT_LOG")
        if not env_log:
            raise RuntimeError(
                "make_app: pass log_path=... or set PYTHIA_AUDIT_LOG env var"
            )
        log_path = Path(env_log)
    if achievements_config_path is None:
        env_cfg = os.environ.get("PYTHIA_ACHIEVEMENTS_CONFIG")
        if env_cfg:
            achievements_config_path = Path(env_cfg)
    return ReplayServer(log_path, achievements_config_path).app()

# Re-export JSONResponse for callers that want to monkey-patch / wrap.
__all__ = ["JSONResponse", "ReplayServer", "json", "make_app"]
