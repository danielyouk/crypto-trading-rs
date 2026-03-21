"""
Exa / LLM fallback contract for S&P 500 symbols.

Important (simulation vs live)
------------------------------
If your stack uses something like ``init.simulation``, search and answers are often
**restricted to a knowledge base**. In that mode you must **not** route through
open-web ``exa_search`` unless your KB is known to contain an S&P 500 snapshot.

Use ``ExaRunMode.SIMULATION`` only with a function that calls your **KB-scoped**
API (e.g. ``exa_answer`` with simulation / KB-only configuration).

Use ``ExaRunMode.LIVE`` for code paths that may query the **open web** (or a
non-KB-restricted search) to recover tickers when Wikipedia fails.

This package does **not** import ``google-agents-api-gen``; you wire that in your
callables.  However, ``default_gemini_backend()`` provides a ready-made LIVE
backend using Gemini + Google Search Grounding (same mechanism as exa_search
in live mode).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable


class ExaRunMode(str, Enum):
    """How the Exa stack is allowed to retrieve information."""

    LIVE = "live"
    """Unrestricted / web-capable search (when your deployment allows it)."""

    SIMULATION = "simulation"
    """KB-only or simulation-scoped retrieval — no silent open-web fallback."""


@runtime_checkable
class Sp500ExaBackend(Protocol):
    """Pluggable backend: implement with your ``exa_answer`` / ``exa_search`` calls."""

    def list_sp500_symbols(self, *, mode: ExaRunMode) -> Sequence[str]:
        """Return current (or KB-snapshot) S&P 500 ticker strings, uppercase, no dots in suffix if possible."""
        ...


def create_exa_backend(
    *,
    fn_live: Callable[[], Sequence[str]],
    fn_simulation: Callable[[], Sequence[str]],
) -> Sp500ExaBackend:
    """
    Build a backend that dispatches on ``mode``.

    - ``fn_live``: call your **web-capable** Exa path (e.g. live ``exa_search``).
    - ``fn_simulation``: call your **KB-only** path (e.g. ``init.simulation`` / no web).

    This keeps simulation from accidentally using the same code path as live web search.
    """

    class _Backend:
        def list_sp500_symbols(self, *, mode: ExaRunMode) -> Sequence[str]:
            if mode == ExaRunMode.SIMULATION:
                return fn_simulation()
            return fn_live()

    return _Backend()


def default_gemini_backend(
    *,
    fn_simulation: Optional[Callable[[], Sequence[str]]] = None,
) -> Sp500ExaBackend:
    """
    Ready-made backend: LIVE uses Gemini Search Grounding (needs GOOGLE_API_KEY).

    For SIMULATION, supply ``fn_simulation`` or an error is raised at call time.
    Requires: ``pip install google-genai`` and ``GOOGLE_API_KEY`` in env.
    """
    from pairs_eda.gemini_search import search_sp500_via_gemini

    def _sim_not_configured() -> Sequence[str]:
        raise RuntimeError(
            "SIMULATION mode not configured. Pass fn_simulation= to default_gemini_backend()."
        )

    return create_exa_backend(
        fn_live=search_sp500_via_gemini,
        fn_simulation=fn_simulation or _sim_not_configured,
    )
