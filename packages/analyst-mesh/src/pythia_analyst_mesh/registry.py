"""Pluggable analyst registry.

The registry maps short slugs (e.g. ``"politics"``) to ``BaseAnalyst``
subclasses. New analysts can be registered at runtime without code changes
to this repo — see README §"Pluggability".
"""
from __future__ import annotations

import logging
from typing import overload

from .analysts import BUILTIN_ANALYSTS
from .base import BaseAnalyst
from .types import LLMConfig

logger = logging.getLogger(__name__)

class AnalystRegistry:
    """Singleton-ish registry of analyst classes.

    A single registry instance is usually sufficient per process. Calling
    ``AnalystRegistry()`` multiple times returns fresh registries, but each
    one auto-registers the four built-in analysts on construction.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseAnalyst]] = {}
        self._register_builtins()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _register_builtins(self) -> None:
        """Auto-register the four built-in analysts."""
        for name, cls in BUILTIN_ANALYSTS:
            # Bypass ``register`` to avoid duplicate-key warnings on
            # re-construction of the registry.
            self._registry[name] = cls
        logger.debug(
            "auto-registered %d built-in analysts: %s",
            len(BUILTIN_ANALYSTS),
            ", ".join(name for name, _ in BUILTIN_ANALYSTS),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def register(self, name: str, cls: type[BaseAnalyst]) -> None:
        """Register an analyst class under ``name``.

        Overwrites existing registrations silently — re-registration is
        treated as an upgrade, not an error. Use ``list_known`` to verify.
        """
        if not name or not name.strip():
            raise ValueError("analyst name must be non-empty")
        # Verify it's a BaseAnalyst subclass (not an instance).
        if not (isinstance(cls, type) and issubclass(cls, BaseAnalyst)):
            raise TypeError(
                f"cls must be a subclass of BaseAnalyst, got: {cls!r}"
            )
        # Verify the class has analyst_id / specialty set (sanity check).
        if not getattr(cls, "analyst_id", "") or not getattr(cls, "specialty", ""):
            raise ValueError(
                f"{cls.__name__} must set both `analyst_id` and `specialty` "
                "class attributes."
            )
        if name in self._registry:
            logger.warning(
                "overwriting existing analyst registration for %r (was %s, now %s)",
                name,
                self._registry[name].__name__,
                cls.__name__,
            )
        self._registry[name] = cls

    def get(self, name: str) -> type[BaseAnalyst]:
        """Look up a registered analyst class by slug.

        Raises ``KeyError`` if unknown.
        """
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown analyst: {name!r}. "
                f"Known: {', '.join(sorted(self._registry))}"
            ) from exc

    def list_known(self) -> list[str]:
        """Return sorted list of registered analyst slugs."""
        return sorted(self._registry.keys())

    def build_mesh(
        self, analyst_names: list[str], llm_config: LLMConfig
    ) -> list[BaseAnalyst]:
        """Instantiate each named analyst with a shared ``LLMConfig``.

        Unknown names raise ``KeyError`` immediately (fail fast — better to
        skip a misconfigured mesh at startup than silently run a partial
        one). Use ``list_known`` to discover valid names.
        """
        # Validate first, so we don't half-instantiate.
        missing = [n for n in analyst_names if n not in self._registry]
        if missing:
            raise KeyError(
                f"unknown analysts requested: {missing}. "
                f"Known: {', '.join(sorted(self._registry))}"
            )
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for n in analyst_names:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return [self._registry[n](llm_config) for n in unique]

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def __contains__(self, name: object) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    @overload
    def __getitem__(self, name: str) -> type[BaseAnalyst]: ...
    def __getitem__(self, name: str) -> type[BaseAnalyst]:
        return self.get(name)
