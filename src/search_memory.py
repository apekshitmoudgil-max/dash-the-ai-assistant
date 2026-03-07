"""Dash: search_memory tool. Keyword search over archival memory."""

import re
from pathlib import Path
from typing import Optional

from src import config
from src.memory import read_archived_observations
from src.session_memory import load_recent_summaries


# Common English stop words to ignore in queries
_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "was", "were", "are", "been",
    "be", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "it", "its", "this", "that", "and", "or", "but",
    "not", "no", "if", "then", "so", "as", "about", "my", "i",
    "me", "we", "you", "your", "he", "she", "they", "them",
    "what", "when", "where", "how", "why", "which", "who",
}

# Minimum shared prefix length for fuzzy matching
_MIN_PREFIX_LEN: int = 4


def _tokenize_query(query: str) -> list[str]:
    """Split query into lowercase keywords, removing stop words and short words."""
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _tokenize_text(text: str) -> list[str]:
    """Split text into lowercase words for fuzzy matching."""
    return re.findall(r"\w+", text.lower())


def _score_text(text: str, keywords: list[str]) -> float:
    """Score text against keywords using hybrid exact + fuzzy matching.

    Scoring:
    - Exact substring match: 1.0 point per keyword
    - Fuzzy prefix match (4+ chars shared prefix): 0.5 points per keyword
    - Each keyword can score at most once (exact takes priority)
    """
    text_lower = text.lower()
    text_words = _tokenize_text(text)
    score = 0.0

    for kw in keywords:
        # Try exact substring first
        if kw in text_lower:
            score += 1.0
        else:
            # Try fuzzy prefix match: check if any text word shares a prefix
            # of at least _MIN_PREFIX_LEN chars with the keyword
            kw_prefix = kw[:_MIN_PREFIX_LEN] if len(kw) >= _MIN_PREFIX_LEN else kw
            if len(kw_prefix) >= _MIN_PREFIX_LEN:
                for tw in text_words:
                    tw_prefix = tw[:_MIN_PREFIX_LEN]
                    if kw_prefix == tw_prefix:
                        score += 0.5
                        break

    return score


def _snippet(text: str, keywords: list[str], max_len: int = 150) -> str:
    """Extract a relevant snippet around the first keyword match."""
    text_lower = text.lower()
    best_pos = len(text)

    for kw in keywords:
        pos = text_lower.find(kw)
        if 0 <= pos < best_pos:
            best_pos = pos

    # If no exact match found, try prefix match
    if best_pos == len(text):
        text_words_with_pos = [(m.start(), m.group()) for m in re.finditer(r"\w+", text_lower)]
        for kw in keywords:
            kw_prefix = kw[:_MIN_PREFIX_LEN] if len(kw) >= _MIN_PREFIX_LEN else kw
            if len(kw_prefix) >= _MIN_PREFIX_LEN:
                for pos, tw in text_words_with_pos:
                    if tw[:_MIN_PREFIX_LEN] == kw_prefix and pos < best_pos:
                        best_pos = pos
                        break

    # If still no match, start from beginning
    if best_pos >= len(text):
        best_pos = 0

    start = max(0, best_pos - 40)
    end = min(len(text), start + max_len)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def search_memory(
    input_data: dict,
    summaries_file: Optional[Path] = None,
    archive_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> str:
    """Search over archival memory: summaries, archived observations, session logs.

    Returns top 5 matches with date and context snippet.
    """
    query = input_data.get("query", "").strip()
    if not query:
        return "No search query provided."

    keywords = _tokenize_query(query)
    if not keywords:
        return "Query too generic. Try more specific keywords."

    results: list[tuple[float, str, str, str]] = []  # (score, date, source, snippet)

    # 1. Search all session summaries
    all_summaries = load_recent_summaries(
        n=0,
        summaries_file=summaries_file or config.SESSION_SUMMARIES_FILE,
    )
    for s in all_summaries:
        searchable = " ".join([
            str(s.get("summary", "")),
            " ".join(str(t) for t in s.get("tasks_changed", [])),
            " ".join(str(o) for o in s.get("observations_added", [])),
            str(s.get("mood", "")),
        ])
        score = _score_text(searchable, keywords)
        if score > 0:
            results.append((
                score,
                str(s.get("date", "unknown")),
                "session_summary",
                _snippet(searchable, keywords),
            ))

    # 2. Search archived observations
    archived = read_archived_observations(
        archive_file=archive_file or config.OBSERVATIONS_ARCHIVE_FILE,
    )
    for obs in archived:
        text = str(obs.get("observation", ""))
        score = _score_text(text, keywords)
        if score > 0:
            results.append((
                score,
                str(obs.get("date", "unknown")),
                "archived_observation",
                _snippet(text, keywords),
            ))

    # 3. Search session log markdown files
    search_log_dir = log_dir or config.LOG_DIR
    if search_log_dir.exists():
        for md_file in sorted(search_log_dir.glob("session_*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                score = _score_text(content, keywords)
                if score > 0:
                    # Extract date from filename: session_2026-03-07_13-28-20.md
                    date_part = md_file.stem.replace("session_", "").split("_")[0]
                    results.append((
                        score,
                        date_part,
                        "session_log",
                        _snippet(content, keywords),
                    ))
            except (OSError, UnicodeDecodeError):
                continue

    if not results:
        return f"No matches found for: {query}"

    # Sort by score descending, take top 5
    results.sort(key=lambda x: x[0], reverse=True)
    top_5 = results[:5]

    lines = [f"Memory search results for: {query}\n"]
    for score, date, source, snip in top_5:
        source_label = source.replace("_", " ").title()
        lines.append(f"- [{date}] ({source_label}) {snip}")

    return "\n".join(lines)
