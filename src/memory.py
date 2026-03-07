"""Dash: JSON file read/write for persistent memory (tasks + user context)."""

import json
from pathlib import Path
from typing import Optional, Union

from src import config

DEFAULT_USER_CONTEXT: dict[str, object] = {
    "preferences": {},
    "priorities": {},
    "observations": [],
    "updated_at": None,
}


def _ensure_data_dir(data_dir: Path) -> None:
    """Create the data directory if it doesn't exist."""
    data_dir.mkdir(parents=True, exist_ok=True)


def read_tasks(tasks_file: Optional[Path] = None) -> list[dict[str, object]]:
    """Read tasks from JSON file. Returns empty list if file missing, empty, or invalid."""
    if tasks_file is None:
        tasks_file = config.TASKS_FILE
    try:
        content = tasks_file.read_text(encoding="utf-8").strip()
        if not content:
            return []
        data = json.loads(content)
        if not isinstance(data, list):
            return []
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def write_tasks(
    tasks: list[dict[str, object]], tasks_file: Optional[Path] = None
) -> None:
    """Write tasks to JSON file. Creates data directory if needed."""
    if tasks_file is None:
        tasks_file = config.TASKS_FILE
    _ensure_data_dir(tasks_file.parent)
    tasks_file.write_text(
        json.dumps(tasks, indent=2, default=str) + "\n", encoding="utf-8"
    )


def read_user_context(
    context_file: Optional[Path] = None,
) -> dict[str, object]:
    """Read user context from JSON file. Returns default structure if missing, empty, or invalid."""
    if context_file is None:
        context_file = config.USER_CONTEXT_FILE
    try:
        content = context_file.read_text(encoding="utf-8").strip()
        if not content:
            return _default_context()
        data = json.loads(content)
        if not isinstance(data, dict):
            return _default_context()
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return _default_context()


def write_user_context(
    context: dict[str, object], context_file: Optional[Path] = None
) -> None:
    """Write user context to JSON file. Creates data directory if needed."""
    if context_file is None:
        context_file = config.USER_CONTEXT_FILE
    _ensure_data_dir(context_file.parent)
    context_file.write_text(
        json.dumps(context, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _default_context() -> dict[str, object]:
    """Return a fresh copy of the default user context structure."""
    return {
        "preferences": {},
        "priorities": {},
        "observations": [],
        "updated_at": None,
    }


# ---------------------------------------------------------------------------
# Observation pruning
# ---------------------------------------------------------------------------


def prune_observations(
    context: dict[str, object],
    max_observations: Optional[int] = None,
    archive_file: Optional[Path] = None,
) -> dict[str, object]:
    """If observations exceed max, keep most recent, archive the rest.

    Returns the pruned context dict (always returns context, even if no pruning needed).
    Archives evicted observations by APPENDING to archive file.
    """
    if max_observations is None:
        max_observations = config.MAX_OBSERVATIONS
    if archive_file is None:
        archive_file = config.OBSERVATIONS_ARCHIVE_FILE

    observations = context.get("observations")
    if not isinstance(observations, list) or len(observations) <= max_observations:
        return context

    # Observations are ordered oldest-first (appended chronologically).
    # Keep the most recent (last N), evict the oldest (first entries).
    evicted = observations[:len(observations) - max_observations]
    kept = observations[len(observations) - max_observations:]

    # Archive evicted observations
    _ensure_data_dir(archive_file.parent)
    existing_archive = read_archived_observations(archive_file=archive_file)
    existing_archive.extend(evicted)
    archive_file.write_text(
        json.dumps(existing_archive, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    context["observations"] = kept
    return context


def read_archived_observations(
    archive_file: Optional[Path] = None,
) -> list[dict[str, object]]:
    """Read the observation archive. Returns empty list if file missing."""
    if archive_file is None:
        archive_file = config.OBSERVATIONS_ARCHIVE_FILE
    try:
        content = archive_file.read_text(encoding="utf-8").strip()
        if not content:
            return []
        data = json.loads(content)
        if not isinstance(data, list):
            return []
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return []
