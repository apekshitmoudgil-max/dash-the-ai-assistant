"""Dash: Morning briefing. One-shot proactive summary, no interactive loop."""

from __future__ import annotations

import asyncio
import json
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import anthropic

from src.config import ANTHROPIC_API_KEY, MODEL
from src.memory import read_tasks, read_user_context
from src.session_memory import load_recent_summaries, format_summaries_for_prompt

# ---------------------------------------------------------------------------
# Terminal colors (matching agent.py)
# ---------------------------------------------------------------------------

CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Streaming markdown renderer (same logic as agent.py)
# ---------------------------------------------------------------------------

class StreamingMarkdownRenderer:
    """Converts **bold** markers to ANSI bold during streaming output."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_bold = False

    def feed(self, text: str) -> str:
        """Process a streaming chunk. Returns text ready to print."""
        output = ""
        self._buffer += text

        while "**" in self._buffer:
            idx = self._buffer.index("**")
            output += self._buffer[:idx]
            self._buffer = self._buffer[idx + 2:]
            if self._in_bold:
                output += RESET + MAGENTA
                self._in_bold = False
            else:
                output += BOLD
                self._in_bold = True

        if self._buffer.endswith("*"):
            output += self._buffer[:-1]
            self._buffer = "*"
        else:
            output += self._buffer
            self._buffer = ""

        return output

    def flush(self) -> str:
        """Flush any remaining buffered text."""
        result = self._buffer
        self._buffer = ""
        if self._in_bold:
            result += RESET
            self._in_bold = False
        return result


# ---------------------------------------------------------------------------
# Briefing prompt builder
# ---------------------------------------------------------------------------

def build_briefing_prompt(
    tasks: list[dict[str, object]],
    user_context: dict[str, object],
    session_summaries: list[dict[str, object]],
) -> str:
    """Construct the one-shot briefing prompt from current state.

    This is a pure function (no side effects) so it can be tested independently.
    """
    current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    # Format tasks
    if tasks:
        active_tasks = [t for t in tasks if t.get("status") == "active"]
        completed_tasks = [t for t in tasks if t.get("status") == "completed"]

        task_lines: list[str] = []
        for t in active_tasks:
            line = f"- [{t.get('priority', 'medium').upper()}] {t.get('title')}"
            if t.get("due_date"):
                line += f" (due: {t.get('due_date')})"
            if t.get("context"):
                line += f" . {t.get('context')}"
            task_lines.append(line)
        active_tasks_str = "\n".join(task_lines) if task_lines else "No active tasks."

        completed_count = len(completed_tasks)
        tasks_section = f"Active tasks:\n{active_tasks_str}\n\nCompleted tasks: {completed_count} total."
    else:
        tasks_section = "No tasks yet."

    # Format user context
    if user_context.get("preferences") or user_context.get("priorities") or user_context.get("observations"):
        user_context_str = json.dumps(user_context, indent=2, default=str)
    else:
        user_context_str = "No user context stored yet."

    # Format session summaries
    sessions_str = format_summaries_for_prompt(session_summaries)

    return f"""You are Dash, a personal task assistant. The user has just started their day and wants a quick morning briefing.

Current date and time: {current_datetime}

## User context
{user_context_str}

## Tasks
{tasks_section}

## Recent sessions
{sessions_str}

## Your job

Give a concise, helpful morning briefing with these 5 sections:

1. **Overdue or Urgent**: Any tasks past their deadline or flagged as high priority. If none, say so briefly. IMPORTANT: Calculate overdue duration precisely. Today is {current_datetime}. If a task was due 2026-02-28 and today is 2026-03-07, it is 7 days overdue, not "yesterday."
2. **Today's Priorities**: Based on task status, priority, and deadlines, what should the user focus on today?
3. **Upcoming Deadlines**: Tasks with deadlines in the next few days. If none, say so.
4. **Patterns Noticed**: Based on recent sessions, any observations about work habits, productivity, or recurring themes.
5. **One Suggestion**: A single proactive recommendation based on everything you know.

Keep it direct and actionable. No fluff. Use the user's name if you know it.
If there are no tasks, sessions, or context yet, acknowledge that and give a brief encouraging note about getting started.
Do not use em dashes anywhere in your response. Use periods, commas, colons, or parentheses instead."""


# ---------------------------------------------------------------------------
# Briefing runner
# ---------------------------------------------------------------------------

async def run_briefing() -> None:
    """Main orchestrator: load data, build prompt, stream response, exit."""
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set. Set it in your environment or .env file.")
        sys.exit(1)

    # Load data using the same modules as agent.py
    tasks = read_tasks()
    user_context = read_user_context()
    session_summaries = load_recent_summaries(n=5)

    # Build the prompt
    prompt = build_briefing_prompt(tasks, user_context, session_summaries)

    # Print header (matching agent.py branding style)
    W = "\033[97m"   # bright white
    GR = "\033[90m"  # dark gray
    CY = "\033[96m"  # bright cyan
    print()
    print(f"  {W}\u2584\u2588\u2588\u2588\u2588\u2588\u2584{RESET}")
    print(f"  {W}\u2588{GR}\u2588{CY}\u2588{GR}\u2588{CY}\u2588{GR}\u2588{W}\u2588{RESET}       {BOLD}{MAGENTA}Dash{RESET} {DIM}Morning Briefing{RESET}")
    print(f"  {W}\u2580\u2580\u2588\u2588\u2588\u2580\u2580{RESET}       {DIM}{datetime.now().strftime('%A, %B %d')}{RESET}")
    print(f"    {W}\u2588 \u2588{RESET}")
    print()

    # Stream the response
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    md_renderer = StreamingMarkdownRenderer()

    print(f"{MAGENTA}{BOLD}Dash:{RESET} ", end="", flush=True)

    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    rendered = md_renderer.feed(event.delta.text)
                    if rendered:
                        print(f"{MAGENTA}{rendered}{RESET}", end="", flush=True)
            elif event.type == "content_block_stop":
                remaining = md_renderer.flush()
                if remaining:
                    print(f"{MAGENTA}{remaining}{RESET}", end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_briefing())
