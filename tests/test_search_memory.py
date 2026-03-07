"""Tests for search_memory tool."""

import json
from pathlib import Path

import pytest

from src.search_memory import (
    _score_text,
    _snippet,
    _tokenize_query,
    _tokenize_text,
    search_memory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create temp data dir and patch config paths."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    monkeypatch.setattr("src.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.config.SESSION_SUMMARIES_FILE", tmp_path / "session_summaries.json")
    monkeypatch.setattr("src.config.OBSERVATIONS_ARCHIVE_FILE", tmp_path / "observations_archive.json")
    monkeypatch.setattr("src.config.LOG_DIR", log_dir)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: _tokenize_query
# ---------------------------------------------------------------------------

class TestTokenizeQuery:
    def test_splits_into_lowercase_keywords(self):
        result = _tokenize_query("Building AI Agents")
        assert "building" in result
        assert "agents" in result

    def test_removes_stop_words(self):
        result = _tokenize_query("what is the best way to do this")
        assert "best" in result
        assert "way" in result
        assert "the" not in result
        assert "is" not in result
        assert "to" not in result

    def test_removes_short_words(self):
        result = _tokenize_query("I am a builder")
        assert "builder" in result
        # "am" is length 2 but also a stop word concept; "a" is stop word
        assert "a" not in result

    def test_handles_empty_query(self):
        assert _tokenize_query("") == []

    def test_handles_only_stop_words(self):
        assert _tokenize_query("the a an is was") == []


# ---------------------------------------------------------------------------
# Tests: _tokenize_text
# ---------------------------------------------------------------------------

class TestTokenizeText:
    def test_splits_into_words(self):
        result = _tokenize_text("Hello World, this is a test!")
        assert "hello" in result
        assert "world" in result
        assert "test" in result


# ---------------------------------------------------------------------------
# Tests: _score_text
# ---------------------------------------------------------------------------

class TestScoreText:
    def test_exact_substring_match(self):
        score = _score_text("I love building AI agents", ["building", "agents"])
        assert score == 2.0

    def test_case_insensitive_exact(self):
        score = _score_text("BUILDING things is fun", ["building"])
        assert score == 1.0

    def test_zero_for_no_match(self):
        score = _score_text("The weather is nice", ["python", "coding"])
        assert score == 0.0

    def test_fuzzy_prefix_match(self):
        # "built" and "building" share prefix "buil" (4 chars)
        score = _score_text("I was building something", ["built"])
        assert score == 0.5

    def test_exact_takes_priority_over_fuzzy(self):
        # "build" is an exact substring of "I build things"
        score = _score_text("I build things", ["build"])
        assert score == 1.0  # Not 0.5

    def test_fuzzy_needs_min_prefix_length(self):
        # "cats" prefix is "cats" (4 chars), "categorize" prefix is "cate"
        # "cats" != "cate" so no fuzzy match, and "cats" not in text
        score = _score_text("categorize items", ["cats"])
        assert score == 0.0

    def test_multiple_keywords_mixed_scoring(self):
        text = "The user enjoys building and learning new things"
        # "building" = exact match (1.0), "learnt" shares "lear" prefix with "learning" (0.5)
        score = _score_text(text, ["building", "learnt"])
        assert score == 1.5


# ---------------------------------------------------------------------------
# Tests: _snippet
# ---------------------------------------------------------------------------

class TestSnippet:
    def test_extracts_around_first_match(self):
        text = "A" * 200 + " important keyword here " + "B" * 200
        result = _snippet(text, ["important"])
        assert "important" in result

    def test_adds_ellipsis_when_truncated(self):
        text = "X" * 100 + " keyword " + "Y" * 100
        result = _snippet(text, ["keyword"], max_len=50)
        assert result.startswith("...")
        assert result.endswith("...")

    def test_no_leading_ellipsis_at_start(self):
        text = "keyword at the beginning of text"
        result = _snippet(text, ["keyword"])
        assert not result.startswith("...")

    def test_replaces_newlines(self):
        text = "line one\nkeyword\nline three"
        result = _snippet(text, ["keyword"])
        assert "\n" not in result


# ---------------------------------------------------------------------------
# Tests: search_memory (integration)
# ---------------------------------------------------------------------------

class TestSearchMemory:
    def test_finds_match_in_summaries(self, data_dir):
        summaries = [
            {
                "date": "2026-03-07",
                "summary": "User built an AI agent with Python",
                "tasks_changed": [],
                "observations_added": [],
                "mood": "focused",
            },
        ]
        (data_dir / "session_summaries.json").write_text(json.dumps(summaries))

        result = search_memory({"query": "AI agent Python"})
        assert "2026-03-07" in result
        assert "Session Summary" in result

    def test_finds_match_in_archived_observations(self, data_dir):
        archive = [
            {"date": "2026-03-01", "observation": "User prefers evening work sessions"},
        ]
        (data_dir / "observations_archive.json").write_text(json.dumps(archive))
        (data_dir / "session_summaries.json").write_text("[]")

        result = search_memory({"query": "evening work"})
        assert "2026-03-01" in result
        assert "Archived Observation" in result

    def test_finds_match_in_session_logs(self, data_dir):
        log_dir = data_dir / "logs"
        log_file = log_dir / "session_2026-03-05_20-00-00.md"
        log_file.write_text("User discussed building a personal dashboard with React.")
        (data_dir / "session_summaries.json").write_text("[]")

        result = search_memory({"query": "dashboard React"})
        assert "2026-03-05" in result
        assert "Session Log" in result

    def test_returns_top_5_ranked_by_score(self, data_dir):
        summaries = [
            {"date": f"2026-03-0{i}", "summary": f"Session about Python {'Python ' * i}",
             "tasks_changed": [], "observations_added": [], "mood": "ok"}
            for i in range(1, 8)
        ]
        (data_dir / "session_summaries.json").write_text(json.dumps(summaries))

        result = search_memory({"query": "Python"})
        # Should have at most 5 results
        assert result.count("Session Summary") <= 5

    def test_returns_no_matches_message(self, data_dir):
        (data_dir / "session_summaries.json").write_text("[]")
        result = search_memory({"query": "nonexistent topic xyz"})
        assert "No matches found" in result

    def test_handles_empty_query(self, data_dir):
        result = search_memory({"query": ""})
        assert "No search query" in result

    def test_handles_generic_query(self, data_dir):
        result = search_memory({"query": "the a is"})
        assert "too generic" in result.lower()

    def test_handles_missing_files(self, data_dir):
        # No files created, should handle gracefully
        result = search_memory({"query": "something specific"})
        assert "No matches found" in result

    def test_combines_results_across_sources(self, data_dir):
        summaries = [
            {"date": "2026-03-07", "summary": "Worked on machine learning project",
             "tasks_changed": [], "observations_added": [], "mood": "focused"},
        ]
        archive = [
            {"date": "2026-03-01", "observation": "Interested in machine learning"},
        ]
        (data_dir / "session_summaries.json").write_text(json.dumps(summaries))
        (data_dir / "observations_archive.json").write_text(json.dumps(archive))

        result = search_memory({"query": "machine learning"})
        assert "Session Summary" in result
        assert "Archived Observation" in result

    def test_fuzzy_match_finds_related_words(self, data_dir):
        summaries = [
            {"date": "2026-03-07", "summary": "User was building a dashboard",
             "tasks_changed": [], "observations_added": [], "mood": "focused"},
        ]
        (data_dir / "session_summaries.json").write_text(json.dumps(summaries))

        # "builder" shares "buil" prefix with "building"
        result = search_memory({"query": "builder dashboard"})
        assert "2026-03-07" in result
