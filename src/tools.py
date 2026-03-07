"""Dash: Tool definitions and implementations for the agentic loop."""

import json
import uuid
from datetime import datetime, timezone

from src.memory import (
    prune_observations,
    read_tasks,
    read_user_context,
    write_tasks,
    write_user_context,
)
from src.web_tools import web_fetch, web_search

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic API format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, object]] = [
    {
        "name": "add_task",
        "description": "Create a new task. Use this when the user mentions something they need to do.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short, actionable title for the task",
                },
                "context": {
                    "type": "string",
                    "description": "Why this task matters or additional context",
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Task priority. Defaults to medium.",
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date in YYYY-MM-DD format. Convert relative dates (e.g. 'by Saturday') to absolute dates using today's date.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List tasks. Use to show the user what's on their plate or to check existing tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "enum": ["active", "all", "completed"],
                    "description": "Which tasks to show. Defaults to active.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "update_task",
        "description": "Update an existing task's title, context, priority, or agent notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The task UUID to update",
                },
                "title": {
                    "type": "string",
                    "description": "New title for the task",
                },
                "context": {
                    "type": "string",
                    "description": "New context for the task",
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "New priority for the task",
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date in YYYY-MM-DD format",
                },
                "agent_notes": {
                    "type": "string",
                    "description": "Agent's own notes about this task",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "complete_task",
        "description": "Mark a task as completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The task UUID to mark as completed",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "get_user_context",
        "description": "Read what you know about the user: preferences, priorities, and past observations.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "update_user_context",
        "description": "Store something you learned about the user. Use for preferences, priorities, or observations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "What to store. Use 'observation' to append to observations list, or a key like 'learning_style' for preferences, or 'current_focus' for priorities.",
                },
                "value": {
                    "type": "string",
                    "description": "The value to store",
                },
                "reason": {
                    "type": "string",
                    "description": "Why you're storing this (for auditability)",
                },
            },
            "required": ["key", "value", "reason"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Use when the user asks about recent events, needs facts, or wants to look something up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": "Fetch and read the content of a specific URL. Use when the user shares a link or you need to read a particular web page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and read",
                },
            },
            "required": ["url"],
        },
    },
]

# Tool definitions with cache_control on the last tool.
# Anthropic caches everything up to and including the last cache_control breakpoint.
import copy

TOOL_DEFINITIONS_CACHED: list[dict[str, object]] = copy.deepcopy(TOOL_DEFINITIONS)
TOOL_DEFINITIONS_CACHED[-1]["cache_control"] = {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _add_task(input_data: dict[str, str]) -> str:
    """Create a new task and persist it."""
    tasks = read_tasks()
    task_id = str(uuid.uuid4())
    now = _now_iso()

    task: dict[str, object] = {
        "id": task_id,
        "title": input_data["title"],
        "context": input_data.get("context"),
        "due_date": input_data.get("due_date"),
        "status": "active",
        "priority": input_data.get("priority", "medium"),
        "created_at": now,
        "updated_at": now,
        "agent_notes": None,
    }

    tasks.append(task)
    write_tasks(tasks)
    return f"Task created: \"{task['title']}\" (id: {task_id}, priority: {task['priority']})"


def _list_tasks(input_data: dict[str, str]) -> str:
    """List tasks with optional filtering."""
    tasks = read_tasks()
    task_filter = input_data.get("filter", "active")

    if task_filter == "active":
        filtered = [t for t in tasks if t.get("status") == "active"]
    elif task_filter == "completed":
        filtered = [t for t in tasks if t.get("status") == "completed"]
    else:  # "all"
        filtered = tasks

    if not filtered:
        return f"No {task_filter} tasks found."

    lines: list[str] = []
    for t in filtered:
        priority = t.get("priority", "medium")
        status = t.get("status", "active")
        line = f"- [{priority.upper()}] {t.get('title')} (status: {status}, id: {t.get('id')})"
        if t.get("due_date"):
            line += f"\n  Due: {t.get('due_date')}"
        if t.get("context"):
            line += f"\n  Context: {t.get('context')}"
        if t.get("agent_notes"):
            line += f"\n  Agent notes: {t.get('agent_notes')}"
        lines.append(line)

    return f"Tasks ({task_filter}):\n" + "\n".join(lines)


def _update_task(input_data: dict[str, str]) -> str:
    """Update fields on an existing task."""
    tasks = read_tasks()
    task_id = input_data["id"]

    for task in tasks:
        if task.get("id") == task_id:
            if "title" in input_data:
                task["title"] = input_data["title"]
            if "context" in input_data:
                task["context"] = input_data["context"]
            if "priority" in input_data:
                task["priority"] = input_data["priority"]
            if "due_date" in input_data:
                task["due_date"] = input_data["due_date"]
            if "agent_notes" in input_data:
                task["agent_notes"] = input_data["agent_notes"]
            task["updated_at"] = _now_iso()
            write_tasks(tasks)
            return f"Task updated: \"{task.get('title')}\" (id: {task_id})"

    return f"Task not found: {task_id}"


def _complete_task(input_data: dict[str, str]) -> str:
    """Mark a task as completed."""
    tasks = read_tasks()
    task_id = input_data["id"]

    for task in tasks:
        if task.get("id") == task_id:
            task["status"] = "completed"
            task["updated_at"] = _now_iso()
            write_tasks(tasks)
            return f"Task completed: \"{task.get('title')}\" (id: {task_id})"

    return f"Task not found: {task_id}"


def _get_user_context(input_data: dict[str, str]) -> str:
    """Read the full user context."""
    context = read_user_context()
    return json.dumps(context, indent=2, default=str)


def _update_user_context(input_data: dict[str, str]) -> str:
    """Store a learning about the user."""
    context = read_user_context()
    key = input_data["key"]
    value = input_data["value"]
    reason = input_data["reason"]
    now = _now_iso()

    if key == "observation":
        observations = context.get("observations", [])
        if not isinstance(observations, list):
            observations = []
        observations.append(
            {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "observation": value,
                "source": "conversation",
            }
        )
        context["observations"] = observations
    elif key in ("current_focus", "reason", "top_goal"):
        priorities = context.get("priorities", {})
        if not isinstance(priorities, dict):
            priorities = {}
        priorities[key] = value
        context["priorities"] = priorities
    else:
        preferences = context.get("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
        preferences[key] = value
        context["preferences"] = preferences

    context["updated_at"] = now
    write_user_context(context)

    # Prune observations if over limit
    pruned = prune_observations(context)
    if len(pruned.get("observations", [])) < len(context.get("observations", [])):
        write_user_context(pruned)

    return f"Stored {key} = \"{value}\" (reason: {reason})"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_MAP: dict[str, object] = {
    "add_task": _add_task,
    "list_tasks": _list_tasks,
    "update_task": _update_task,
    "complete_task": _complete_task,
    "get_user_context": _get_user_context,
    "update_user_context": _update_user_context,
    "web_search": web_search,
    "web_fetch": web_fetch,
}


def execute_tool(name: str, input_data: dict[str, str]) -> str:
    """Dispatch a tool call by name and return the result string."""
    handler = _TOOL_MAP.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    if callable(handler):
        return handler(input_data)
    return f"Tool '{name}' is not callable"
