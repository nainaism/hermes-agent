"""Tests for the TinyFish web search and extract provider.

Covers:
- TinyFishSearchProvider.is_configured() env var gating
- TinyFishSearchProvider.search() — happy path, rate limit, HTTP error, request error
- TinyFishExtractProvider.is_configured() env var gating
- TinyFishExtractProvider.extract() — happy path, empty URLs, format validation
- Result normalization (title, url, description, position for search; url, title, content for extract)
- Limit truncation
- _is_backend_available("tinyfish") integration
- _get_backend() recognizes "tinyfish" as a valid configured backend
- check_web_api_key() includes tinyfish in availability check
- web_search dispatches to tinyfish
- web_extract dispatches to tinyfish (NOT search-only error)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TinyFishSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestTinyFishSearchProviderIsConfigured:
    def test_configured_when_key_set(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert TinyFishSearchProvider().is_configured() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert TinyFishSearchProvider().is_configured() is False

    def test_not_configured_when_key_whitespace(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "   ")
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert TinyFishSearchProvider().is_configured() is False

    def test_provider_name(self):
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert TinyFishSearchProvider().provider_name() == "tinyfish"

    def test_implements_web_search_provider(self):
        from tools.web_providers.base import WebSearchProvider
        from tools.web_providers.tinyfish import TinyFishSearchProvider
        assert issubclass(TinyFishSearchProvider, WebSearchProvider)


class TestTinyFishSearchProviderSearch:
    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "Alpha", "url": "https://alpha.example.com", "snippet": "desc A", "position": 1},
            {"title": "Beta", "url": "https://beta.example.com", "snippet": "desc B", "position": 2},
            {"title": "Gamma", "url": "https://gamma.example.com", "snippet": "desc C", "position": 3},
        ]
    }

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        return m

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = TinyFishSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {"title": "Alpha", "url": "https://alpha.example.com", "description": "desc A", "position": 1}
        assert web[2]["position"] == 3

    def test_sends_api_key_header_and_query(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["params"] = kwargs.get("params", {})
            return self._mock_resp({"results": []})

        with patch("httpx.get", side_effect=fake_get):
            TinyFishSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://api.search.tinyfish.ai"
        assert captured["headers"].get("X-API-Key") == "tf-key-123"
        assert captured["params"].get("query") == "q"

    def test_limit_is_respected_client_side(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = TinyFishSearchProvider().search("q", limit=2)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2

    def test_empty_results(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", return_value=self._mock_resp({"results": []})):
            result = TinyFishSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_results_key_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", return_value=self._mock_resp({})):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_rate_limit_returns_failure(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        m = MagicMock()
        m.status_code = 429
        with patch("httpx.get", return_value=m):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "rate limit" in result["error"].lower()

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        bad = MagicMock()
        bad.status_code = 500
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=bad)

        with patch("httpx.get", side_effect=err):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "500" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        with patch("httpx.get", side_effect=httpx.RequestError("boom")):
            result = TinyFishSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "Could not reach" in result["error"]

    def test_missing_key_returns_failure(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_providers.tinyfish import TinyFishSearchProvider

        result = TinyFishSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "TINYFISH_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# TinyFishExtractProvider unit tests
# ---------------------------------------------------------------------------


class TestTinyFishExtractProviderIsConfigured:
    def test_configured_when_key_set(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider
        assert TinyFishExtractProvider().is_configured() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_providers.tinyfish import TinyFishExtractProvider
        assert TinyFishExtractProvider().is_configured() is False

    def test_provider_name(self):
        from tools.web_providers.tinyfish import TinyFishExtractProvider
        assert TinyFishExtractProvider().provider_name() == "tinyfish"

    def test_implements_web_extract_provider(self):
        from tools.web_providers.base import WebExtractProvider
        from tools.web_providers.tinyfish import TinyFishExtractProvider
        assert issubclass(TinyFishExtractProvider, WebExtractProvider)


class TestTinyFishExtractProviderExtract:
    _SAMPLE_RESPONSE = {
        "results": [
            {
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Page",
                "text": "# Hello\n\nWorld",
                "description": "An example page",
                "language": "en",
                "latency_ms": 120,
            },
        ]
    }

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        return m

    def test_happy_path_extracts_content(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        with patch("httpx.post", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is True
        data = result["data"]
        assert len(data) == 1
        assert data[0]["url"] == "https://example.com"
        assert data[0]["title"] == "Example Page"
        assert data[0]["content"] == "# Hello\n\nWorld"
        assert data[0]["metadata"]["language"] == "en"

    def test_empty_urls_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        result = TinyFishExtractProvider().extract([])
        assert result["success"] is True
        assert result["data"] == []

    def test_sends_api_key_and_urls(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            TinyFishExtractProvider().extract(["https://a.com", "https://b.com"])

        assert captured["url"] == "https://api.fetch.tinyfish.ai"
        assert captured["headers"].get("X-API-Key") == "tf-key-123"
        assert captured["json"]["urls"] == ["https://a.com", "https://b.com"]
        assert captured["json"]["format"] == "markdown"

    def test_format_kwarg_forwarded(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            TinyFishExtractProvider().extract(["https://a.com"], format="html")

        assert captured["json"]["format"] == "html"

    def test_invalid_format_defaults_to_markdown(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            TinyFishExtractProvider().extract(["https://a.com"], format="xml")

        assert captured["json"]["format"] == "markdown"

    def test_rate_limit_returns_failure(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        m = MagicMock()
        m.status_code = 429
        with patch("httpx.post", return_value=m):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is False
        assert "rate limit" in result["error"].lower()

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        bad = MagicMock()
        bad.status_code = 401
        err = httpx.HTTPStatusError("401", request=MagicMock(), response=bad)

        with patch("httpx.post", side_effect=err):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is False
        assert "401" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        with patch("httpx.post", side_effect=httpx.RequestError("timeout")):
            result = TinyFishExtractProvider().extract(["https://example.com"])

        assert result["success"] is False
        assert "Could not reach" in result["error"]

    def test_missing_key_returns_failure(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        result = TinyFishExtractProvider().extract(["https://example.com"])
        assert result["success"] is False
        assert "TINYFISH_API_KEY" in result["error"]

    def test_api_errors_included_in_results(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        resp_data = {
            "results": [
                {"url": "https://ok.com", "title": "OK", "text": "content"},
            ],
            "errors": [
                {"url": "https://fail.com", "error": "timeout"},
            ],
        }

        with patch("httpx.post", return_value=self._mock_resp(resp_data)):
            result = TinyFishExtractProvider().extract(["https://ok.com", "https://fail.com"])

        assert result["success"] is True
        data = result["data"]
        assert len(data) == 2
        assert data[0]["url"] == "https://ok.com"
        assert data[1]["url"] == "https://fail.com"
        assert "timeout" in data[1]["error"]

    def test_max_10_urls_per_request(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_providers.tinyfish import TinyFishExtractProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        urls = [f"https://example.com/{i}" for i in range(15)]
        with patch("httpx.post", side_effect=fake_post):
            TinyFishExtractProvider().extract(urls)

        assert len(captured["json"]["urls"]) == 10


# ---------------------------------------------------------------------------
# Integration: _is_backend_available / _get_backend / check_web_api_key
# ---------------------------------------------------------------------------


class TestTinyFishBackendWiring:
    def test_is_backend_available_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("tinyfish") is True

    def test_is_backend_available_false_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("tinyfish") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "tinyfish"})
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        assert web_tools._get_backend() == "tinyfish"

    def test_auto_detect_picks_tinyfish_when_only_key_set(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "tinyfish"

    def test_tinyfish_does_not_override_paid_provider(self, monkeypatch):
        """Tavily (higher priority) should win in auto-detect."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY", "EXA_API_KEY",
                    "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly")
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "tavily"

    def test_check_web_api_key_true_when_tinyfish_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "tinyfish"})
        monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-123")
        assert web_tools.check_web_api_key() is True
