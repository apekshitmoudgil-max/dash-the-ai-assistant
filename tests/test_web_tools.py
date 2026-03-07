"""Tests for src/web_tools.py: web_search and web_fetch tools."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.web_tools import web_fetch, web_search


# ---------------------------------------------------------------------------
# Helper: create a fake tavily module so the lazy import inside web_search
# resolves to our mock. The function does `from tavily import TavilyClient`
# at call time, so we inject a module into sys.modules.
# ---------------------------------------------------------------------------

def _make_tavily_module(mock_client_cls: MagicMock) -> ModuleType:
    """Create a fake 'tavily' module whose TavilyClient is the given mock."""
    mod = ModuleType("tavily")
    mod.TavilyClient = mock_client_cls  # type: ignore[attr-defined]
    return mod


class TestWebSearch:
    """Tests for the web_search function."""

    def test_empty_query_returns_error(self) -> None:
        """Empty query should return an error message."""
        result = web_search({"query": ""})
        assert result == "Error: No search query provided."

    @patch("src.web_tools.TAVILY_API_KEY", "")
    def test_missing_api_key_returns_error(self) -> None:
        """Missing TAVILY_API_KEY should return a configuration error."""
        result = web_search({"query": "python testing"})
        assert "TAVILY_API_KEY is not configured" in result

    @patch("src.web_tools.TAVILY_API_KEY", "test-key-123")
    def test_formatted_results(self) -> None:
        """Search results should be formatted as numbered markdown entries."""
        mock_response = {
            "results": [
                {
                    "title": "Python Testing Guide",
                    "url": "https://example.com/testing",
                    "content": "A comprehensive guide to testing in Python.",
                },
                {
                    "title": "Pytest Documentation",
                    "url": "https://docs.pytest.org",
                    "content": "Official pytest documentation and tutorials.",
                },
            ]
        }

        mock_client = MagicMock()
        mock_client.search.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"tavily": _make_tavily_module(mock_cls)}):
            result = web_search({"query": "python testing"})

        mock_cls.assert_called_once_with(api_key="test-key-123")
        assert "Search results for: python testing" in result
        assert "1. **Python Testing Guide**" in result
        assert "https://example.com/testing" in result
        assert "A comprehensive guide to testing in Python." in result
        assert "2. **Pytest Documentation**" in result
        assert "https://docs.pytest.org" in result

    @patch("src.web_tools.TAVILY_API_KEY", "test-key-123")
    def test_no_results(self) -> None:
        """Empty results list should return a 'no results' message."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_cls = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"tavily": _make_tavily_module(mock_cls)}):
            result = web_search({"query": "xyzzy nonexistent"})

        assert "No results found for: xyzzy nonexistent" in result

    @patch("src.web_tools.TAVILY_API_KEY", "test-key-123")
    def test_api_error(self) -> None:
        """An exception from TavilyClient.search should be caught and reported."""
        mock_client = MagicMock()
        mock_client.search.side_effect = RuntimeError("API rate limit exceeded")
        mock_cls = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"tavily": _make_tavily_module(mock_cls)}):
            result = web_search({"query": "test query"})

        assert "Search failed:" in result
        assert "API rate limit exceeded" in result

    @patch("src.web_tools.TAVILY_API_KEY", "test-key-123")
    def test_max_results_passed(self) -> None:
        """max_results=5 should be passed to the TavilyClient.search call."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_cls = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"tavily": _make_tavily_module(mock_cls)}):
            web_search({"query": "test"})

        mock_client.search.assert_called_once_with(
            query="test",
            max_results=5,
            search_depth="basic",
        )

    @patch("src.web_tools.TAVILY_API_KEY", "test-key-123")
    def test_missing_package(self) -> None:
        """ImportError for tavily should return an install instruction."""
        # Remove tavily from sys.modules so the import fails
        saved = sys.modules.pop("tavily", None)
        try:
            import builtins

            original_import = builtins.__import__

            def _import_no_tavily(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
                if name == "tavily":
                    raise ImportError("No module named 'tavily'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=_import_no_tavily):
                result = web_search({"query": "test"})

            assert "tavily-python is not installed" in result
        finally:
            if saved is not None:
                sys.modules["tavily"] = saved


class TestWebFetch:
    """Tests for the web_fetch function."""

    def test_empty_url_returns_error(self) -> None:
        """Empty URL should return an error message."""
        result = web_fetch({"url": ""})
        assert result == "Error: No URL provided."

    @patch("src.web_tools.trafilatura.extract")
    @patch("src.web_tools.requests.get")
    def test_extracted_content(self, mock_get: MagicMock, mock_extract: MagicMock) -> None:
        """Successful fetch should return extracted content with URL header."""
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello world</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_extract.return_value = "Hello world"

        result = web_fetch({"url": "https://example.com"})

        assert "Content from https://example.com" in result
        assert "Hello world" in result

    @patch("src.web_tools.requests.get")
    def test_timeout(self, mock_get: MagicMock) -> None:
        """Timeout should return a clear timeout message."""
        import requests as req_lib

        mock_get.side_effect = req_lib.exceptions.Timeout("Connection timed out")

        result = web_fetch({"url": "https://slow-site.com"})
        assert "Fetch timed out" in result
        assert "https://slow-site.com" in result

    @patch("src.web_tools.requests.get")
    def test_http_error(self, mock_get: MagicMock) -> None:
        """HTTP errors should report the status code."""
        import requests as req_lib

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = req_lib.exceptions.HTTPError(
            "404 Client Error"
        )
        mock_get.return_value = mock_response

        result = web_fetch({"url": "https://example.com/missing"})
        assert "HTTP error 404" in result

    @patch("src.web_tools.trafilatura.extract")
    @patch("src.web_tools.requests.get")
    def test_no_content_extracted(self, mock_get: MagicMock, mock_extract: MagicMock) -> None:
        """When trafilatura returns None, report that content could not be extracted."""
        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_extract.return_value = None

        result = web_fetch({"url": "https://example.com/empty"})
        assert "Could not extract readable content" in result

    @patch("src.web_tools.trafilatura.extract")
    @patch("src.web_tools.requests.get")
    def test_truncation(self, mock_get: MagicMock, mock_extract: MagicMock) -> None:
        """Content longer than 8000 chars should be truncated with a message."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Long content</body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        long_text = "x" * 10000
        mock_extract.return_value = long_text

        result = web_fetch({"url": "https://example.com/long"})
        assert "[Content truncated. Full page is longer than displayed.]" in result
        # Result should contain the truncated content (8000 chars) plus the truncation notice
        assert "x" * 8000 in result

    @patch("src.web_tools.trafilatura.extract")
    @patch("src.web_tools.requests.get")
    def test_auto_prepend_https(self, mock_get: MagicMock, mock_extract: MagicMock) -> None:
        """URL without scheme should get https:// prepended."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Content</body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_extract.return_value = "Some content"

        web_fetch({"url": "example.com/page"})

        # Verify requests.get was called with https:// prepended
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://example.com/page"

    @patch("src.web_tools.trafilatura.extract")
    @patch("src.web_tools.requests.get")
    def test_trafilatura_options(self, mock_get: MagicMock, mock_extract: MagicMock) -> None:
        """Trafilatura should be called with the correct extraction options."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Text</body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_extract.return_value = "Extracted text"

        web_fetch({"url": "https://example.com"})

        mock_extract.assert_called_once_with(
            mock_response.text,
            favor_recall=True,
            include_links=False,
            include_images=False,
            include_tables=True,
        )
