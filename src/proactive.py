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

    # Time context detection
    day_of_week = datetime.now().strftime("%A")
    hour = datetime.now().hour
    is_weekend = day_of_week in ("Saturday", "Sunday")

    # >>> CUSTOMIZE: Adjust time context to match the user's schedule.
    # >>> These defaults assume a typical 9-5 worker. Edit to fit your routine.
    if is_weekend:
        time_context = "It's the weekend. The user has more flexible time for bigger tasks or planning."
    elif hour >= 18:
        time_context = "It's evening. The user may have limited time. Keep suggestions focused."
    else:
        time_context = "It's daytime on a weekday. Keep suggestions light, the user is likely at work."

    # Last session context for emotional continuity
    last_session_summary = ""
    if session_summaries:
        last = session_summaries[-1]
        last_session_summary = str(last.get("summary", ""))

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

    instructions = f"""## Your job

Give a concise, personalized morning briefing. You know this person. Show it.

Time context: {time_context}
Last session: {last_session_summary}

## How to open
Start by picking up the thread from last session. Reference something SPECIFIC that happened, not a mood label.
- Good: "Last time you were grinding on that search feature. Did it hold up?"
- Good: "You shipped three features last session. Riding that wave?"
- Bad: "You're feeling reflective and curious." (Don't tell users how they feel.)
- If the last session was rough, acknowledge it lightly: "That was a long one last time." Don't dwell.
- Always end the opener with a question. Let the user set today's tone.
- If there are no previous sessions, just say hello warmly and ask what they want to tackle.

## Then give 3 sections (keep each concise)
1. **Top Priority**: The ONE most important thing for today's available time. Use behavioral patterns and time context to shape this (don't show your reasoning, just let it influence what you recommend).
2. **Also On Your Plate**: 2-3 other items, briefly, low pressure.
3. **One Suggestion**: A single proactive recommendation. Make it genuinely novel, not just restating a task.

If it's a weeknight, scope to the 2-hour window. Don't overload.
If it's a weekend, you can suggest something bigger.

Use the user's name if you know it.
If there are no tasks, sessions, or context yet, keep it short and encouraging.
IMPORTANT DATE RULES: Today is {current_datetime}. Check session dates carefully before saying "yesterday" or "last week." If a session date is the same day as today, say "earlier today" not "yesterday." Calculate overdue durations precisely: if a task was due 2026-02-28 and today is 2026-03-07, it is 7 days overdue, not "yesterday."
Do not use em dashes anywhere in your response. Use periods, commas, colons, or parentheses instead."""

    # >>> CUSTOMIZE: Adjust the personality line below to match your agent's voice.
    # >>> This shapes the tone of the morning briefing.
    return f"""You are Dash, a personal task assistant. You're helpful, concise, and you remember context from past sessions.

Current date and time: {current_datetime}

## User context
{user_context_str}

## Tasks
{tasks_section}

## Recent sessions
{sessions_str}

{instructions}"""


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
