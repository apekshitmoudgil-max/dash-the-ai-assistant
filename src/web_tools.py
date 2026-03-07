"""Dash -- Web search and fetch tools."""

from urllib.parse import urlparse

import requests
import trafilatura

from src.config import (
    TAVILY_API_KEY,
    WEB_FETCH_MAX_LENGTH,
    WEB_FETCH_TIMEOUT,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT,
)


def web_search(input_data: dict) -> str:
    """Search the web using Tavily and return markdown-formatted results."""
    query = input_data.get("query", "").strip()
    if not query:
        return "Error: No search query provided."

    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY is not configured. Add it to your .env file."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=WEB_SEARCH_MAX_RESULTS,
            search_depth="basic",
        )
    except ImportError:
        return "Error: tavily-python is not installed. Run: pip install tavily-python"
    except Exception as e:
        return f"Search failed: {e}"

    results = response.get("results", [])
    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "No snippet available.")
        lines.append(f"{i}. **{title}**")
        lines.append(f"   {url}")
        lines.append(f"   {content}\n")

    return "\n".join(lines)


def web_fetch(input_data: dict) -> str:
    """Fetch a URL and extract readable text content."""
    url = input_data.get("url", "").strip()
    if not url:
        return "Error: No URL provided."

    # Auto-prepend https:// if no scheme
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url

    try:
        resp = requests.get(
            url,
            timeout=WEB_FETCH_TIMEOUT,
            headers={"User-Agent": "Dash/0.3 (Personal Assistant)"},
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return f"Fetch timed out after {WEB_FETCH_TIMEOUT}s: {url}"
    except requests.exceptions.ConnectionError:
        return f"Could not connect to: {url}"
    except requests.exceptions.HTTPError as e:
        return f"HTTP error {resp.status_code}: {url} ({e})"
    except Exception as e:
        return f"Fetch failed: {e}"

    text = trafilatura.extract(
        resp.text,
        favor_recall=True,
        include_links=False,
        include_images=False,
        include_tables=True,
    )

    if not text or not text.strip():
        return f"Could not extract readable content from: {url}"

    if len(text) > WEB_FETCH_MAX_LENGTH:
        text = text[:WEB_FETCH_MAX_LENGTH] + "\n\n[Content truncated. Full page is longer than displayed.]"

    return f"Content from {url}:\n\n{text}"
