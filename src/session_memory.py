"""Dash: Session memory. Summarize conversations, persist across sessions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

from src import config


# ---------------------------------------------------------------------------
# Summarization prompt
# ---------------------------------------------------------------------------

_SUMMARIZATION_PROMPT = """Summarize this conversation between a user and their AI assistant Dash.

Return a JSON object with:
{
  "summary": "One-sentence overview of the session",
  "tasks_changed": ["task1 (added)", "task2 (completed)"],
  "observations_added": ["observation1", "observation2"],
  "mood": "one or two words describing user's vibe"
}

If nothing happened in a category, use an empty array.
Respond with ONLY the JSON, no markdown fences or extra text."""


# ---------------------------------------------------------------------------
# Generate summary from conversation file
# ---------------------------------------------------------------------------


def generate_session_summary(
    convo_file: Path, client: anthropic.Anthropic
) -> dict[str, object]:
    """Read a session conversation .md file and summarize it via Claude Haiku.

    Returns a structured dict with summary, tasks_changed, observations_added,
    mood, date, and turns.
    """
    # Read the conversation file
    try:
        content = convo_file.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        content = ""

    if not content:
        return _minimal_summary()

    # Count turns by counting "**You:**" markers
    turns = content.count("**You:**")
    if turns == 0:
        return _minimal_summary()

    # Extract date from the header line: "# Dash Session -- 2026-02-28 14:30"
    date_str = _extract_date_from_header(content)

    # Call Claude to summarize
    try:
        response = client.messages.create(
            model=config.MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": f"{_SUMMARIZATION_PROMPT}\n\n---\n\n{content}",
                }
            ],
        )
        raw_text = response.content[0].text.strip()
        # Strip markdown fences if Haiku wraps the JSON
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[: -3].strip()
        summary = json.loads(raw_text)
    except (json.JSONDecodeError, IndexError, AttributeError, KeyError):
        summary = {
            "summary": "Session occurred but summary could not be parsed.",
            "tasks_changed": [],
            "observations_added": [],
            "mood": "unknown",
        }
    except Exception:
        summary = {
            "summary": "Session occurred but summary could not be generated.",
            "tasks_changed": [],
            "observations_added": [],
            "mood": "unknown",
        }

    summary["date"] = date_str
    summary["turns"] = turns
    return summary


def _extract_date_from_header(content: str) -> str:
    """Pull date and time from the first line of a session log.

    Expected format: '# Dash Session -- 2026-02-28 14:30'
    Returns 'YYYY-MM-DD HH:MM' if time is present, otherwise 'YYYY-MM-DD'.
    Falls back to current timestamp.
    """
    first_line = content.split("\n")[0]
    parts = first_line.split()
    for i, part in enumerate(parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            # Check if next part is a time (HH:MM)
            if i + 1 < len(parts) and len(parts[i + 1]) == 5 and parts[i + 1][2] == ":":
                return f"{part} {parts[i + 1]}"
            return part
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _minimal_summary() -> dict[str, object]:
    """Return a minimal summary for empty or missing conversations."""
    return {
        "summary": "Empty session, no conversation recorded.",
        "tasks_changed": [],
        "observations_added": [],
        "mood": "n/a",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "turns": 0,
    }


# ---------------------------------------------------------------------------
# Save / load summaries
# ---------------------------------------------------------------------------


def save_session_summary(
    summary: dict[str, object],
    summaries_file: Optional[Path] = None,
) -> None:
    """Append a session summary to the summaries JSON array file."""
    if summaries_file is None:
        summaries_file = config.SESSION_SUMMARIES_FILE

    summaries_file.parent.mkdir(parents=True, exist_ok=True)

    existing = load_recent_summaries(n=0, summaries_file=summaries_file)
    existing.append(summary)

    summaries_file.write_text(
        json.dumps(existing, indent=2, default=str) + "\n", encoding="utf-8"
    )


def load_recent_summaries(
    n: int = 5,
    summaries_file: Optional[Path] = None,
) -> list[dict[str, object]]:
    """Load the last N session summaries from disk.

    If n=0, returns ALL summaries (used internally by save).
    Returns empty list if file missing or corrupted.
    """
    if summaries_file is None:
        summaries_file = config.SESSION_SUMMARIES_FILE

    try:
        content = summaries_file.read_text(encoding="utf-8").strip()
        if not content:
            return []
        data = json.loads(content)
        if not isinstance(data, list):
            return []
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    if n == 0:
        return data
    return data[-n:]


# ---------------------------------------------------------------------------
# Format for prompt injection
# ---------------------------------------------------------------------------


def format_summaries_for_prompt(summaries: list[dict[str, object]]) -> str:
    """Format session summaries into concise text for the system prompt.

    Most recent first. Target ~100 tokens per summary.
    """
    if not summaries:
        return "No previous sessions yet."

    # Most recent first
    ordered = list(reversed(summaries))
    lines: list[str] = []

    for s in ordered:
        date = s.get("date", "unknown")
        summary = s.get("summary", "No summary")
        tasks = s.get("tasks_changed", [])
        observations = s.get("observations_added", [])
        mood = s.get("mood", "")

        parts: list[str] = [summary]
        if tasks:
            parts.append(f"Tasks: {', '.join(str(t) for t in tasks)}.")
        if observations:
            parts.append(f"Learned: {', '.join(str(o) for o in observations)}.")
        if mood:
            parts.append(f"Mood: {mood}.")

        lines.append(f"- {date}: {' '.join(parts)}")

    return "\n".join(lines)
