"""Tests for src/memory.py: JSON persistence for tasks and user context."""

import json
from pathlib import Path

import pytest

from src.memory import read_tasks, read_user_context, write_tasks, write_user_context


class TestReadTasks:
    """Tests for read_tasks()."""

    def test_returns_empty_list_when_file_missing(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "tasks.json"
        result = read_tasks(tasks_file=missing_file)
        assert result == []

    def test_returns_empty_list_when_file_empty(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "tasks.json"
        empty_file.write_text("", encoding="utf-8")
        result = read_tasks(tasks_file=empty_file)
        assert result == []

    def test_returns_empty_list_when_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "tasks.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        result = read_tasks(tasks_file=bad_file)
        assert result == []

    def test_returns_empty_list_when_json_is_not_array(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "tasks.json"
        bad_file.write_text('{"key": "value"}', encoding="utf-8")
        result = read_tasks(tasks_file=bad_file)
        assert result == []

    def test_returns_tasks_from_valid_file(self, tmp_path: Path) -> None:
        tasks_file = tmp_path / "tasks.json"
        tasks = [{"id": "123", "title": "Test task", "status": "active"}]
        tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
        result = read_tasks(tasks_file=tasks_file)
        assert len(result) == 1
        assert result[0]["title"] == "Test task"


class TestWriteTasks:
    """Tests for write_tasks()."""

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        tasks_file = tmp_path / "data" / "tasks.json"
        tasks = [
            {"id": "abc-123", "title": "Buy groceries", "status": "active"},
            {"id": "def-456", "title": "Read a book", "status": "completed"},
        ]
        write_tasks(tasks, tasks_file=tasks_file)
        result = read_tasks(tasks_file=tasks_file)
        assert len(result) == 2
        assert result[0]["id"] == "abc-123"
        assert result[1]["title"] == "Read a book"

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        tasks_file = tmp_path / "nested" / "dir" / "tasks.json"
        write_tasks([], tasks_file=tasks_file)
        assert tasks_file.exists()

    def test_writes_with_indent(self, tmp_path: Path) -> None:
        tasks_file = tmp_path / "tasks.json"
        write_tasks([{"id": "1", "title": "Test"}], tasks_file=tasks_file)
        content = tasks_file.read_text(encoding="utf-8")
        assert "\n" in content  # indented JSON has newlines
        assert "  " in content  # indent=2


class TestReadUserContext:
    """Tests for read_user_context()."""

    def test_returns_default_when_file_missing(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "user_context.json"
        result = read_user_context(context_file=missing_file)
        assert result == {
            "preferences": {},
            "priorities": {},
            "observations": [],
            "updated_at": None,
        }

    def test_returns_default_when_file_empty(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "user_context.json"
        empty_file.write_text("", encoding="utf-8")
        result = read_user_context(context_file=empty_file)
        assert result["preferences"] == {}
        assert result["observations"] == []

    def test_returns_default_when_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "user_context.json"
        bad_file.write_text("not json at all", encoding="utf-8")
        result = read_user_context(context_file=bad_file)
        assert result["updated_at"] is None

    def test_returns_default_when_json_is_not_dict(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "user_context.json"
        bad_file.write_text("[1, 2, 3]", encoding="utf-8")
        result = read_user_context(context_file=bad_file)
        assert "preferences" in result

    def test_reads_valid_context(self, tmp_path: Path) -> None:
        ctx_file = tmp_path / "user_context.json"
        ctx = {
            "preferences": {"style": "hands-on"},
            "priorities": {"focus": "agents"},
            "observations": [{"date": "2026-02-26", "observation": "test"}],
            "updated_at": "2026-02-26T10:00:00+00:00",
        }
        ctx_file.write_text(json.dumps(ctx), encoding="utf-8")
        result = read_user_context(context_file=ctx_file)
        assert result["preferences"]["style"] == "hands-on"
        assert len(result["observations"]) == 1


class TestWriteUserContext:
    """Tests for write_user_context()."""

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        ctx_file = tmp_path / "data" / "user_context.json"
        ctx = {
            "preferences": {"learning": "by doing"},
            "priorities": {},
            "observations": [],
            "updated_at": "2026-02-26T10:00:00+00:00",
        }
        write_user_context(ctx, context_file=ctx_file)
        result = read_user_context(context_file=ctx_file)
        assert result["preferences"]["learning"] == "by doing"

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        ctx_file = tmp_path / "deep" / "nested" / "user_context.json"
        write_user_context({"preferences": {}, "priorities": {}, "observations": [], "updated_at": None}, context_file=ctx_file)
        assert ctx_file.exists()
