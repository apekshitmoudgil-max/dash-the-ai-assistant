"""Tests for src/session_memory.py: session summaries."""

import json
from pathlib import Path

import pytest

from src.session_memory import (
    format_summaries_for_prompt,
    generate_session_summary,
    load_recent_summaries,
    save_session_summary,
)


# ---------------------------------------------------------------------------
# Fake Anthropic client for generate tests
# ---------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.stop_reason = "end_turn"
        self.usage = type("Usage", (), {"input_tokens": 100, "output_tokens": 50})()


class FakeMessages:
    @staticmethod
    def create(**kwargs: object) -> FakeMessage:
        return FakeMessage(
            json.dumps(
                {
                    "summary": "User added tasks and discussed priorities.",
                    "tasks_changed": ["groceries (added)", "email (completed)"],
                    "observations_added": ["prefers morning focus time"],
                    "mood": "productive",
                }
            )
        )


class FakeClient:
    messages = FakeMessages()


class FakeBadMessages:
    @staticmethod
    def create(**kwargs: object) -> FakeMessage:
        return FakeMessage("This is not valid JSON at all!!!")


class FakeBadClient:
    messages = FakeBadMessages()


# ---------------------------------------------------------------------------
# TestSaveSessionSummary
# ---------------------------------------------------------------------------


class TestSaveSessionSummary:
    """Tests for save_session_summary()."""

    def test_saves_to_new_file(self, tmp_path: Path) -> None:
        summaries_file = tmp_path / "session_summaries.json"
        summary = {"summary": "First session", "date": "2026-02-28", "turns": 3}

        save_session_summary(summary, summaries_file=summaries_file)

        data = json.loads(summaries_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["summary"] == "First session"

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        summaries_file = tmp_path / "session_summaries.json"
        first = {"summary": "First session", "date": "2026-02-27", "turns": 2}
        second = {"summary": "Second session", "date": "2026-02-28", "turns": 5}

        save_session_summary(first, summaries_file=summaries_file)
        save_session_summary(second, summaries_file=summaries_file)

        data = json.loads(summaries_file.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["summary"] == "First session"
        assert data[1]["summary"] == "Second session"

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        summaries_file = tmp_path / "nested" / "dir" / "session_summaries.json"
        summary = {"summary": "Test", "date": "2026-02-28", "turns": 1}

        save_session_summary(summary, summaries_file=summaries_file)

        assert summaries_file.exists()
        data = json.loads(summaries_file.read_text(encoding="utf-8"))
        assert len(data) == 1


# ---------------------------------------------------------------------------
# TestLoadRecentSummaries
# ---------------------------------------------------------------------------


class TestLoadRecentSummaries:
    """Tests for load_recent_summaries()."""

    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.json"
        result = load_recent_summaries(n=5, summaries_file=missing)
        assert result == []

    def test_returns_last_n_summaries(self, tmp_path: Path) -> None:
        summaries_file = tmp_path / "session_summaries.json"
        data = [{"summary": f"Session {i}", "date": f"2026-02-{i:02d}"} for i in range(1, 11)]
        summaries_file.write_text(json.dumps(data), encoding="utf-8")

        result = load_recent_summaries(n=5, summaries_file=summaries_file)
        assert len(result) == 5
        assert result[0]["summary"] == "Session 6"
        assert result[-1]["summary"] == "Session 10"

    def test_returns_all_if_fewer_than_n(self, tmp_path: Path) -> None:
        summaries_file = tmp_path / "session_summaries.json"
        data = [
            {"summary": "A", "date": "2026-02-01"},
            {"summary": "B", "date": "2026-02-02"},
            {"summary": "C", "date": "2026-02-03"},
        ]
        summaries_file.write_text(json.dumps(data), encoding="utf-8")

        result = load_recent_summaries(n=5, summaries_file=summaries_file)
        assert len(result) == 3

    def test_handles_corrupted_json(self, tmp_path: Path) -> None:
        summaries_file = tmp_path / "session_summaries.json"
        summaries_file.write_text("{{{not json at all", encoding="utf-8")

        result = load_recent_summaries(n=5, summaries_file=summaries_file)
        assert result == []


# ---------------------------------------------------------------------------
# TestFormatSummariesForPrompt
# ---------------------------------------------------------------------------


class TestFormatSummariesForPrompt:
    """Tests for format_summaries_for_prompt()."""

    def test_formats_single_summary(self) -> None:
        summaries = [
            {
                "summary": "Added grocery tasks.",
                "tasks_changed": ["groceries (added)"],
                "observations_added": ["likes lists"],
                "mood": "focused",
                "date": "2026-02-28",
            }
        ]
        result = format_summaries_for_prompt(summaries)
        assert "2026-02-28" in result
        assert "Added grocery tasks." in result
        assert "groceries (added)" in result
        assert "likes lists" in result
        assert "focused" in result

    def test_formats_multiple_summaries_newest_first(self) -> None:
        summaries = [
            {
                "summary": "Older session.",
                "tasks_changed": [],
                "observations_added": [],
                "mood": "calm",
                "date": "2026-02-27",
            },
            {
                "summary": "Newer session.",
                "tasks_changed": [],
                "observations_added": [],
                "mood": "energetic",
                "date": "2026-02-28",
            },
        ]
        result = format_summaries_for_prompt(summaries)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        # Most recent first
        assert "2026-02-28" in lines[0]
        assert "Newer session." in lines[0]
        assert "2026-02-27" in lines[1]
        assert "Older session." in lines[1]

    def test_empty_summaries_returns_fallback(self) -> None:
        result = format_summaries_for_prompt([])
        assert result == "No previous sessions yet."


# ---------------------------------------------------------------------------
# TestGenerateSessionSummary
# ---------------------------------------------------------------------------


class TestGenerateSessionSummary:
    """Tests for generate_session_summary()."""

    def test_generates_summary_from_convo_file(self, tmp_path: Path) -> None:
        convo = tmp_path / "session_2026-02-28_14-30-00.md"
        convo.write_text(
            "# Dash Session -- 2026-02-28 14:30\n\n"
            "**You:** Add groceries to my list\n\n"
            "**Dash:** Done! Added groceries.\n\n---\n\n"
            "**You:** What's on my plate?\n\n"
            "**Dash:** You've got groceries.\n\n---\n\n",
            encoding="utf-8",
        )

        client = FakeClient()
        result = generate_session_summary(convo, client)

        assert result["summary"] == "User added tasks and discussed priorities."
        assert "groceries (added)" in result["tasks_changed"]
        assert result["mood"] == "productive"
        assert result["date"] == "2026-02-28 14:30"
        assert result["turns"] == 2

    def test_handles_empty_convo(self, tmp_path: Path) -> None:
        convo = tmp_path / "session_empty.md"
        convo.write_text("", encoding="utf-8")

        client = FakeClient()
        result = generate_session_summary(convo, client)

        assert result["turns"] == 0
        assert "Empty session" in result["summary"]

    def test_handles_missing_convo_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.md"

        client = FakeClient()
        result = generate_session_summary(missing, client)

        assert result["turns"] == 0
        assert "Empty session" in result["summary"]

    def test_handles_bad_json_from_claude(self, tmp_path: Path) -> None:
        convo = tmp_path / "session_2026-02-28_15-00-00.md"
        convo.write_text(
            "# Dash Session -- 2026-02-28 15:00\n\n"
            "**You:** Hello\n\n"
            "**Dash:** Hi there!\n\n---\n\n",
            encoding="utf-8",
        )

        client = FakeBadClient()
        result = generate_session_summary(convo, client)

        assert result["turns"] == 1
        assert "could not be parsed" in result["summary"]
        assert result["date"] == "2026-02-28 15:00"
