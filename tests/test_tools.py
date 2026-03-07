"""Tests for src/tools.py: tool definitions, implementations, and dispatcher."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import tools
from src.memory import read_tasks, read_user_context, write_tasks, write_user_context
from src.tools import TOOL_DEFINITIONS, TOOL_DEFINITIONS_CACHED, execute_tool


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp data directory and patch config paths to use it."""
    tasks_file = tmp_path / "tasks.json"
    context_file = tmp_path / "user_context.json"

    # Patch config module attributes. memory.py looks these up at call time via `config.TASKS_FILE`
    monkeypatch.setattr("src.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.config.TASKS_FILE", tasks_file)
    monkeypatch.setattr("src.config.USER_CONTEXT_FILE", context_file)

    return tmp_path


class TestAddTask:
    """Tests for add_task tool."""

    def test_creates_task_with_uuid(self, data_dir: Path) -> None:
        result = execute_tool("add_task", {"title": "Buy groceries"})
        assert "Task created" in result
        assert "Buy groceries" in result

        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Buy groceries"
        assert len(tasks[0]["id"]) == 36  # UUID format
        assert tasks[0]["status"] == "active"

    def test_creates_task_with_all_fields(self, data_dir: Path) -> None:
        result = execute_tool("add_task", {
            "title": "Deploy app",
            "context": "Production release needed by Friday",
            "priority": "high",
        })
        assert "Deploy app" in result
        assert "high" in result

        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        assert tasks[0]["priority"] == "high"
        assert tasks[0]["context"] == "Production release needed by Friday"
        assert tasks[0]["created_at"] is not None
        assert tasks[0]["updated_at"] is not None
        assert tasks[0]["agent_notes"] is None

    def test_default_priority_is_medium(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Something"})
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        assert tasks[0]["priority"] == "medium"

    def test_appends_to_existing_tasks(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Task 1"})
        execute_tool("add_task", {"title": "Task 2"})
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        assert len(tasks) == 2
        assert tasks[0]["title"] == "Task 1"
        assert tasks[1]["title"] == "Task 2"


class TestListTasks:
    """Tests for list_tasks tool."""

    def test_empty_returns_no_tasks_message(self, data_dir: Path) -> None:
        result = execute_tool("list_tasks", {})
        assert "No active tasks found" in result

    def test_active_filter(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Active task"})
        execute_tool("add_task", {"title": "Another active"})

        # Complete one
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        task_id = tasks[0]["id"]
        execute_tool("complete_task", {"id": str(task_id)})

        result = execute_tool("list_tasks", {"filter": "active"})
        assert "Another active" in result
        assert "Active task" not in result

    def test_completed_filter(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Done task"})
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        execute_tool("complete_task", {"id": str(tasks[0]["id"])})

        result = execute_tool("list_tasks", {"filter": "completed"})
        assert "Done task" in result

    def test_all_filter(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Active one"})
        execute_tool("add_task", {"title": "Completed one"})
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        execute_tool("complete_task", {"id": str(tasks[1]["id"])})

        result = execute_tool("list_tasks", {"filter": "all"})
        assert "Active one" in result
        assert "Completed one" in result

    def test_default_filter_is_active(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Active task"})
        execute_tool("add_task", {"title": "Completed task"})
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        execute_tool("complete_task", {"id": str(tasks[1]["id"])})

        # No filter specified. Should default to active
        result = execute_tool("list_tasks", {})
        assert "Active task" in result
        assert "Completed task" not in result


class TestUpdateTask:
    """Tests for update_task tool."""

    def test_updates_title(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Old title"})
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        task_id = str(tasks[0]["id"])

        result = execute_tool("update_task", {"id": task_id, "title": "New title"})
        assert "Task updated" in result

        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        assert tasks[0]["title"] == "New title"

    def test_updates_multiple_fields(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Task"})
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        task_id = str(tasks[0]["id"])
        original_updated_at = tasks[0]["updated_at"]

        result = execute_tool("update_task", {
            "id": task_id,
            "context": "New context",
            "priority": "high",
            "agent_notes": "Important task",
        })
        assert "Task updated" in result

        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        assert tasks[0]["context"] == "New context"
        assert tasks[0]["priority"] == "high"
        assert tasks[0]["agent_notes"] == "Important task"
        assert tasks[0]["updated_at"] != original_updated_at

    def test_nonexistent_task_returns_error(self, data_dir: Path) -> None:
        result = execute_tool("update_task", {"id": "nonexistent-id", "title": "X"})
        assert "Task not found" in result


class TestCompleteTask:
    """Tests for complete_task tool."""

    def test_marks_task_completed(self, data_dir: Path) -> None:
        execute_tool("add_task", {"title": "Finish this"})
        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        task_id = str(tasks[0]["id"])

        result = execute_tool("complete_task", {"id": task_id})
        assert "Task completed" in result
        assert "Finish this" in result

        tasks = read_tasks(tasks_file=data_dir / "tasks.json")
        assert tasks[0]["status"] == "completed"

    def test_nonexistent_task_returns_error(self, data_dir: Path) -> None:
        result = execute_tool("complete_task", {"id": "bad-id"})
        assert "Task not found" in result


class TestGetUserContext:
    """Tests for get_user_context tool."""

    def test_returns_default_structure(self, data_dir: Path) -> None:
        result = execute_tool("get_user_context", {})
        parsed = json.loads(result)
        assert "preferences" in parsed
        assert "priorities" in parsed
        assert "observations" in parsed
        assert parsed["updated_at"] is None

    def test_returns_stored_context(self, data_dir: Path) -> None:
        ctx = {
            "preferences": {"style": "direct"},
            "priorities": {"focus": "agents"},
            "observations": [],
            "updated_at": "2026-02-26T10:00:00+00:00",
        }
        write_user_context(ctx, context_file=data_dir / "user_context.json")

        result = execute_tool("get_user_context", {})
        parsed = json.loads(result)
        assert parsed["preferences"]["style"] == "direct"


class TestUpdateUserContext:
    """Tests for update_user_context tool."""

    def test_stores_preference(self, data_dir: Path) -> None:
        result = execute_tool("update_user_context", {
            "key": "learning_style",
            "value": "hands-on",
            "reason": "User said they prefer building over reading",
        })
        assert "Stored" in result

        ctx = read_user_context(context_file=data_dir / "user_context.json")
        assert ctx["preferences"]["learning_style"] == "hands-on"
        assert ctx["updated_at"] is not None

    def test_stores_priority(self, data_dir: Path) -> None:
        execute_tool("update_user_context", {
            "key": "current_focus",
            "value": "building AI agents",
            "reason": "Mentioned agents multiple times",
        })
        ctx = read_user_context(context_file=data_dir / "user_context.json")
        assert ctx["priorities"]["current_focus"] == "building AI agents"

    def test_appends_observation(self, data_dir: Path) -> None:
        execute_tool("update_user_context", {
            "key": "observation",
            "value": "User works best in the morning",
            "reason": "They mentioned morning focus blocks",
        })
        execute_tool("update_user_context", {
            "key": "observation",
            "value": "User dislikes verbose output",
            "reason": "Asked for shorter responses",
        })

        ctx = read_user_context(context_file=data_dir / "user_context.json")
        assert len(ctx["observations"]) == 2
        assert ctx["observations"][0]["observation"] == "User works best in the morning"
        assert ctx["observations"][0]["source"] == "conversation"
        assert ctx["observations"][1]["observation"] == "User dislikes verbose output"


class TestExecuteTool:
    """Tests for the tool dispatcher."""

    def test_dispatches_known_tool(self, data_dir: Path) -> None:
        result = execute_tool("add_task", {"title": "Dispatch test"})
        assert "Task created" in result

    def test_unknown_tool_returns_error(self, data_dir: Path) -> None:
        result = execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_all_tools_in_definitions(self) -> None:
        """Verify every tool in TOOL_DEFINITIONS has a matching handler."""
        tool_names = {str(t["name"]) for t in TOOL_DEFINITIONS}
        expected = {"add_task", "list_tasks", "update_task", "complete_task", "get_user_context", "update_user_context", "web_search", "web_fetch", "search_memory"}
        assert tool_names == expected

    def test_web_search_dispatches(self, data_dir: Path) -> None:
        """execute_tool should dispatch web_search to the web_tools function."""
        mock_ws = MagicMock(return_value="search results")
        with patch.dict(tools._TOOL_MAP, {"web_search": mock_ws}):
            result = execute_tool("web_search", {"query": "test"})
        mock_ws.assert_called_once_with({"query": "test"})
        assert result == "search results"

    def test_web_fetch_dispatches(self, data_dir: Path) -> None:
        """execute_tool should dispatch web_fetch to the web_tools function."""
        mock_wf = MagicMock(return_value="fetched content")
        with patch.dict(tools._TOOL_MAP, {"web_fetch": mock_wf}):
            result = execute_tool("web_fetch", {"url": "https://example.com"})
        mock_wf.assert_called_once_with({"url": "https://example.com"})
        assert result == "fetched content"

    def test_cache_control_on_last_tool(self) -> None:
        """Cache control marker should be on the last tool definition."""
        assert "cache_control" in TOOL_DEFINITIONS_CACHED[-1]
        assert TOOL_DEFINITIONS_CACHED[-1]["cache_control"] == {"type": "ephemeral"}
        # No other tool should have cache_control
        for tool in TOOL_DEFINITIONS_CACHED[:-1]:
            assert "cache_control" not in tool
