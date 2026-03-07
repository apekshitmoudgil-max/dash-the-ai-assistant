"""Dash: Pattern synthesis. Infers behavioral patterns from session data."""

import json
from typing import Optional

import anthropic

from src import config
from src.memory import read_user_context, write_user_context
from src.session_memory import load_recent_summaries


MAX_PATTERNS: int = 10

_SYNTHESIS_PROMPT: str = """You are analyzing a user's behavioral patterns from their AI assistant interactions.

Given:
1. Their current observations (things the agent noticed about them)
2. Their last 5 session summaries (what happened in recent sessions)
3. Their existing inferred patterns (if any)

Your job: produce an UPDATED list of 2-10 behavioral patterns.

Rules:
- Each pattern should be something actionable for a personal assistant to know.
- If an existing pattern is still supported by evidence, keep it and update last_confirmed.
- If an existing pattern contradicts new evidence, revise it or lower confidence.
- Add new patterns only when you see clear signals (not speculation).
- Categories: work_habits, motivation, energy, communication, priorities, emotional, scheduling.
- Pay attention to mood fields in session summaries. Look for energy and mood trends across sessions (e.g., "tired on weeknights", "energized after shipping", "frustrated by planning without building").
- Confidence levels: low (single data point), medium (2-3 signals), high (consistent pattern).
- Cap at 10 patterns total. Merge similar ones.
- Do not use em dashes. Use periods, commas, or colons instead.

Return ONLY a JSON array of pattern objects:
[
  {
    "category": "work_habits",
    "pattern": "Description of the pattern",
    "confidence": "high",
    "first_noticed": "YYYY-MM-DD",
    "last_confirmed": "YYYY-MM-DD"
  }
]

No markdown fences, no extra text. Just the JSON array."""


def build_synthesis_input(
    user_context: dict,
    summaries: list[dict],
) -> str:
    """Build the input text for the synthesis prompt."""
    observations = user_context.get("observations", [])
    existing_patterns = user_context.get("inferred_patterns", [])

    parts: list[str] = [_SYNTHESIS_PROMPT, "\n\n---\n\n"]

    parts.append("## Current observations\n")
    if observations:
        for obs in observations:
            parts.append(f"- [{obs.get('date', '?')}] {obs.get('observation', '')}\n")
    else:
        parts.append("No observations yet.\n")

    parts.append("\n## Recent session summaries\n")
    if summaries:
        for s in summaries:
            parts.append(
                f"- [{s.get('date', '?')}] {s.get('summary', '')} "
                f"(mood: {s.get('mood', 'unknown')})\n"
            )
    else:
        parts.append("No session summaries yet.\n")

    parts.append("\n## Existing inferred patterns\n")
    if existing_patterns:
        parts.append(json.dumps(existing_patterns, indent=2, default=str))
    else:
        parts.append("None yet (first synthesis).\n")

    return "".join(parts)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (possibly with language tag like ```json)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def synthesize_patterns(
    client: anthropic.Anthropic,
    user_context: Optional[dict] = None,
    summaries: Optional[list[dict]] = None,
) -> list[dict]:
    """Run pattern synthesis via a single Haiku call.

    Returns the list of inferred pattern dicts.
    Writes updated patterns back to user_context.json.
    """
    if user_context is None:
        user_context = read_user_context()
    if summaries is None:
        summaries = load_recent_summaries(n=5)

    # Don't synthesize if there's nothing to work with
    if not user_context.get("observations") and not summaries:
        return user_context.get("inferred_patterns", [])

    input_text = build_synthesis_input(user_context, summaries)

    response = client.messages.create(
        model=config.MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": input_text}],
    )

    raw_text = response.content[0].text.strip()
    raw_text = _strip_markdown_fences(raw_text)

    try:
        patterns = json.loads(raw_text)
        if not isinstance(patterns, list):
            patterns = []
    except json.JSONDecodeError:
        # If parsing fails, keep existing patterns
        patterns = user_context.get("inferred_patterns", [])

    # Cap at MAX_PATTERNS
    patterns = patterns[:MAX_PATTERNS]

    # Write back to user_context
    user_context["inferred_patterns"] = patterns
    write_user_context(user_context)

    return patterns
