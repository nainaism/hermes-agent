"""TinyFish web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True (TinyFish Search API)
- ``supports_extract()`` -> True (TinyFish Fetch API)

Both methods are sync — the underlying calls are ``httpx.get/post(...)``.
The dispatcher in :func:`tools.web_tools.web_extract_tool` wraps sync
extracts via ``asyncio.to_thread`` when it needs to keep the event loop
responsive.

Config keys this provider responds to::

    web:
      search_backend: "tinyfish"     # explicit per-capability
      extract_backend: "tinyfish"    # explicit per-capability
      backend: "tinyfish"            # shared fallback for both

Env var::

    TINYFISH_API_KEY=***           # https://tinyfish.ai (free tier available)

Rate limits:
    - Search: 5 requests per minute
    - Fetch: 1 credit = 15 URL fetches (credit consumption varies by plan)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_TINYFISH_SEARCH_ENDPOINT = "https://api.search.tinyfish.ai"
_TINYFISH_FETCH_ENDPOINT = "https://api.fetch.tinyfish.ai"

_MAX_FETCH_URLS = 10  # Fetch API max URLs per request


class TinyFishWebSearchProvider(WebSearchProvider):
    """TinyFish search + extract provider.

    Both methods are sync. The web_extract_tool dispatcher wraps sync
    extracts via ``asyncio.to_thread`` when the caller is async.
    """

    @property
    def name(self) -> str:
        return "tinyfish"

    @property
    def display_name(self) -> str:
        return "TinyFish"

    def is_available(self) -> bool:
        """Return True when ``TINYFISH_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("TINYFISH_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the TinyFish Search API.

        Returns ``{"success": True, "data": {"web": [{...}, ...]}}`` on
        success, ``{"success": False, "error": str}`` on failure (incl.
        missing API key, rate limit, HTTP errors).
        """
        import httpx

        api_key = os.getenv("TINYFISH_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "TINYFISH_API_KEY is not set"}

        safe_limit = max(1, int(limit))

        try:
            resp = httpx.get(
                _TINYFISH_SEARCH_ENDPOINT,
                params={"query": query},
                headers={
                    "X-API-Key": api_key,
                    "Accept": "application/json",
                },
                timeout=15,
            )
            if resp.status_code == 429:
                logger.warning("TinyFish Search rate limited (429)")
                return {
                    "success": False,
                    "error": (
                        "TinyFish Search rate limit exceeded (5 req/min). "
                        "Retry after 15s with exponential backoff."
                    ),
                }
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("TinyFish Search HTTP error: %s", exc)
            status = exc.response.status_code
            detail = _search_error_detail(status)
            return {"success": False, "error": f"TinyFish Search returned HTTP {status}: {detail}"}
        except httpx.RequestError as exc:
            logger.warning("TinyFish Search request error: %s", exc)
            return {"success": False, "error": f"Could not reach TinyFish Search: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("TinyFish Search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse TinyFish Search response as JSON"}

        raw_results = data.get("results", []) or []
        truncated = raw_results[:safe_limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("snippet", "")),
                "position": int(r.get("position", i + 1)),
            }
            for i, r in enumerate(truncated)
        ]

        logger.info(
            "TinyFish Search '%s': %d results (limit %d, total %d)",
            query,
            len(web_results),
            limit,
            data.get("total_results", 0),
        )
        return {"success": True, "data": {"web": web_results}}

    # ── Extract ─────────────────────────────────────────────────────────────

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via the TinyFish Fetch API.

        Returns a list of result dicts shaped for the legacy LLM
        post-processing pipeline. On per-URL or whole-batch failure,
        results carry an ``error`` field rather than raising.
        """
        import httpx

        api_key = os.getenv("TINYFISH_API_KEY", "").strip()
        if not api_key:
            return [{"url": u, "title": "", "content": "", "error": "TINYFISH_API_KEY is not set"} for u in urls]

        if not urls:
            return []

        fetch_urls = urls[:_MAX_FETCH_URLS]

        format = kwargs.get("format", "markdown")
        if format not in ("markdown", "html", "json"):
            format = "markdown"

        payload = {
            "urls": fetch_urls,
            "format": format,
        }

        try:
            resp = httpx.post(
                _TINYFISH_FETCH_ENDPOINT,
                json=payload,
                headers={
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=60,
            )
            if resp.status_code == 429:
                logger.warning("TinyFish Fetch rate limited (429)")
                return [
                    {"url": u, "title": "", "content": "",
                     "error": "TinyFish Fetch rate limit exceeded. Retry with backoff."}
                    for u in urls
                ]
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("TinyFish Fetch HTTP error: %s", exc)
            return [
                {"url": u, "title": "", "content": "",
                 "error": f"TinyFish Fetch returned HTTP {exc.response.status_code}"}
                for u in urls
            ]
        except httpx.RequestError as exc:
            logger.warning("TinyFish Fetch request error: %s", exc)
            return [
                {"url": u, "title": "", "content": "",
                 "error": f"Could not reach TinyFish Fetch: {exc}"}
                for u in urls
            ]

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("TinyFish Fetch response parse error: %s", exc)
            return [
                {"url": u, "title": "", "content": "",
                 "error": "Could not parse TinyFish Fetch response as JSON"}
                for u in urls
            ]

        results: List[Dict[str, Any]] = []
        raw_results = data.get("results", []) or []
        for item in raw_results:
            url = str(item.get("url", ""))
            final_url = str(item.get("final_url", url))
            title = str(item.get("title") or "")
            text = item.get("text", "")

            # text can be string or object depending on format
            if isinstance(text, dict):
                content = str(text)
            else:
                content = str(text or "")

            metadata = {
                "final_url": final_url,
                "description": str(item.get("description") or ""),
                "language": str(item.get("language") or ""),
                "author": str(item.get("author") or ""),
                "published_date": str(item.get("published_date") or ""),
                "latency_ms": item.get("latency_ms"),
            }

            results.append({
                "url": url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": metadata,
            })

        # Report per-URL errors from the API
        fetch_errors = data.get("errors", []) or []
        for err in fetch_errors:
            err_url = err.get("url", "unknown")
            err_type = err.get("error", "unknown")
            logger.info("TinyFish Fetch error for %s: %s", err_url, err_type)
            if not any(r["url"] == err_url for r in results):
                results.append({
                    "url": err_url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"TinyFish Fetch failed: {err_type}",
                    "metadata": {},
                })

        logger.info(
            "TinyFish Fetch: %d/%d URLs extracted successfully",
            len(raw_results),
            len(fetch_urls),
        )
        return results

    # ── Setup schema ────────────────────────────────────────────────────────

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "TinyFish",
            "badge": "free",
            "tag": "Free Search + Fetch APIs — no credit card required.",
            "env_vars": [
                {
                    "key": "TINYFISH_API_KEY",
                    "prompt": "TinyFish API key",
                    "url": "https://tinyfish.ai",
                },
            ],
        }


def _search_error_detail(status: int) -> str:
    """Return a human-readable detail message for known Search API HTTP status codes."""
    details = {
        400: "Missing or invalid query parameter",
        401: "Missing or invalid API key",
        402: "Active subscription or credits needed",
        403: "Search API access not enabled for this account",
        404: "Search API not available",
        429: "Rate limit exceeded (5 req/min)",
        500: "Internal server error",
        503: "Search service unavailable — retry with backoff",
    }
    return details.get(status, "Unknown error")
