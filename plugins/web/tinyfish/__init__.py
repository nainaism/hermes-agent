"""TinyFish web search + extract plugin — bundled, auto-loaded.

Both search and extract are sync; the dispatcher in :mod:`tools.web_tools`
handles the wrap when the caller is async.
"""

from __future__ import annotations

from plugins.web.tinyfish.provider import TinyFishWebSearchProvider


def register(ctx) -> None:
    """Register the TinyFish provider with the plugin context."""
    ctx.register_web_search_provider(TinyFishWebSearchProvider())
