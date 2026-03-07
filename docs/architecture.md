# Dash -- Architecture

## System Overview

Dash is a CLI-based agentic personal assistant. The user interacts via terminal. The agent uses Claude Haiku 4.5 with tool use to reason about tasks, user context, and priorities. Dash remembers across sessions and manages its own memory.

## Architecture Diagram (v0.3)

```
┌─────────────────────────────────────────────────┐
│                   CLI Interface                  │
│              (stdin/stdout loop)                  │
└─────────────────┬───────────────────────────────┘
                  │
          ┌───────┴────────┐
          ▼                ▼
┌──────────────────┐ ┌──────────────────────────────┐
│  Agent Loop      │ │  Morning Briefing             │
│  (agent.py)      │ │  (proactive.py)               │
│                  │ │                                │
│  Interactive     │ │  Non-interactive               │
│  REPL mode       │ │  Single output, then exit      │
└────────┬─────────┘ └──────────┬───────────────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│               Shared Infrastructure              │
│                                                  │
│  1. Build system prompt (inject memory + tasks)  │
│  2. Check conversation window (summarize if >24) │
│  3. Call Claude with tools (streaming)           │
│  4. If tool_use → execute locally → loop         │
│  5. If end_turn → stream response to user        │
│  6. Max 5 iterations per turn (safety cap)       │
│  7. On quit → generate session summary           │
└────────┬────────────────────┬───────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌──────────────────────────────┐
│  Claude Haiku    │  │          Tools (8)             │
│  4.5 API         │  │                                │
│  (Anthropic)     │  │  add_task        complete_task  │
│                  │  │  list_tasks      get_user_ctx   │
│  - streaming     │  │  update_task     update_user_ctx│
│  - tool use      │  │  web_search      web_fetch      │
│  - multi-turn    │  │                                │
│                  │  │  update_user_ctx triggers       │
│                  │  │  observation pruning (max 30)   │
└─────────────────┘  └──────────┬───────────────────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
  ┌──────────────────────────────┐  ┌─────────────────┐
  │   Local JSON Files (data/)    │  │  External APIs   │
  │                              │  │                  │
  │  CORE MEMORY (in prompt):     │  │  Tavily (search) │
  │    tasks.json                 │  │  HTTP (fetch)    │
  │    user_context.json          │  │                  │
  │    session_summaries.json     │  │                  │
  │                              │  │                  │
  │  ARCHIVAL (not in prompt):    │  │                  │
  │    observations_archive.json  │  │                  │
  │    logs/session_*.md          │  │                  │
  │    logs/session_*.jsonl       │  │                  │
  └──────────────────────────────┘  └─────────────────┘
```

## Memory Model (3-Tier, Letta-Inspired)

| Tier | What | Where | Loaded Into Prompt? | Lifespan |
|------|------|-------|--------------------:|----------|
| **Core** | User preferences, priorities, observations (max 30), active tasks, last 5 session summaries | `user_context.json`, `tasks.json`, `session_summaries.json` | Yes, every turn | Permanent |
| **Working** | Current conversation messages | `messages` list in memory | Yes, sliding window (last 20 messages) | Single session |
| **Archival** | Full conversation logs, evicted observations | `logs/session_*.md`, `observations_archive.json` | No | Permanent |

### How memory flows through a session:

```
Session Start
  → Load user_context.json (preferences, priorities, observations)
  → Load tasks.json (active tasks)
  → Load session_summaries.json (last 5 summaries)
  → All injected into system prompt

During Session
  → Agent calls update_user_context → writes to user_context.json
  → Agent calls add_task/complete_task → writes to tasks.json
  → Observations auto-pruned at 30 (oldest → observations_archive.json)
  → After 24 messages, old messages summarized into 1 message (sliding window)

Session End (quit/exit)
  → Read the .md conversation log
  → Send to Haiku: "Summarize this session as JSON"
  → Append structured summary to session_summaries.json
```

## Core Components

### 1. Agent Loop (`src/agent.py`)
The main REPL. Reads user input, builds the messages array with system prompt and conversation history, calls Claude with tool definitions, executes any tool calls, loops until Claude returns a text response.

- Max 5 tool-call iterations per user message (safety cap)
- Streaming: tokens print to terminal as they arrive via `client.messages.stream()`
- Conversation window: after 24 messages, older messages are summarized via Haiku
- Session summary: generated at quit, saved for next session

### 2. Tools (`src/tools.py`)
Eight tools:

| Tool | Input | Output | Side Effects |
|------|-------|--------|-------------|
| `add_task` | `{title, context?, priority?, due_date?}` | Confirmation + task ID | Writes to tasks.json |
| `list_tasks` | `{filter?: "active"\|"all"\|"completed"}` | Task list | None (read-only) |
| `update_task` | `{id, title?, context?, priority?, due_date?}` | Confirmation | Writes to tasks.json |
| `complete_task` | `{id}` | Confirmation | Writes to tasks.json |
| `get_user_context` | `{}` | Full user context JSON | None (read-only) |
| `update_user_context` | `{key, value, reason}` | Confirmation | Writes to user_context.json + triggers pruning |
| `web_search` | `{query, num_results?}` | Search results (title, URL, snippet) | None (read-only, calls Tavily API) |
| `web_fetch` | `{url}` | Extracted page content (text) | None (read-only, calls URL via requests) |

### 3. Memory (`src/memory.py`)
JSON file read/write + observation pruning.

**tasks.json** -- Array of task objects:
```json
{
  "id": "uuid",
  "title": "Finish the report",
  "context": "Deadline Thursday",
  "due_date": "2026-03-05",
  "status": "active",
  "priority": "high",
  "created_at": "2026-02-28T10:00:00",
  "updated_at": "2026-02-28T10:00:00",
  "agent_notes": null
}
```

**user_context.json** -- Agent's accumulated knowledge:
```json
{
  "preferences": {"work_style": "Morning person"},
  "priorities": {"current_focus": "Ship MVP by Friday"},
  "observations": [
    {"date": "2026-02-28", "observation": "Procrastinates on writing", "source": "conversation"}
  ],
  "updated_at": "2026-02-28T10:00:00"
}
```

**Observation pruning:** When observations exceed 30, the oldest are moved to `observations_archive.json`. Triggered automatically on every `update_user_context` call.

### 4. Session Memory (`src/session_memory.py`)
Cross-session continuity.

**session_summaries.json** -- Array of structured summaries:
```json
{
  "date": "2026-02-28",
  "turns": 3,
  "summary": "User focused on MVP deadline Friday.",
  "tasks_changed": ["Fix login bug (added)"],
  "observations_added": ["morning person", "MVP deadline"],
  "mood": "Urgent, focused"
}
```

- Generated at session end via one Haiku API call (~$0.003)
- Last 5 loaded into system prompt on next session start
- Older summaries stay in file but aren't loaded

### 5. Context Manager (`src/context_manager.py`)
Conversation window management.

- Threshold: 24 messages → trigger summarization
- Window: keep last 20 messages verbatim
- Older messages compressed into one synthetic user/assistant pair via Haiku
- Transparent to user -- happens before the next API call

### 6. Web Tools (`src/web_tools.py`)
Two tools for accessing external information:

- **`web_search`** -- Calls Tavily API with the query, returns top 5 results (title, URL, snippet, optional content). Max 3,000 chars of content per result. Requires `TAVILY_API_KEY`.
- **`web_fetch`** -- Fetches a URL with `requests`, extracts readable content with `trafilatura`. Returns cleaned text, truncated to 5,000 chars. No API key required.

### 7. Proactive Briefing (`src/proactive.py`)
A separate CLI entry point for non-interactive use:

- Run via `python3 -m src.proactive`
- Loads core memory (tasks, user context, session summaries) from the same `data/` directory
- Sends a single prompt to Claude: "Given what you know about the user and their tasks, generate a morning briefing"
- Prints the briefing to stdout and exits
- Does not modify any state (no session summary, no observation updates)

### 8. System Prompt Design (`src/config.py`)
The system prompt is assembled every turn from a template + 3 injected sections:

```
[Personality + tool instructions + learning instructions + behavior rules]
  +
## Right now
  → current date/time (rebuilt every turn)
  +
## What you know about the user
  → from user_context.json
  +
## Current tasks
  → from tasks.json (active only, with due dates)
  +
## Recent sessions
  → from session_summaries.json (last 5)
```

**Token budget:** ~3270 tokens worst case (hard-capped by observation limit of 30 and summary limit of 5).

## Tech Stack
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.9+ | Familiar, good for prototyping |
| LLM | Claude Haiku 4.5 | Best personality-to-cost ratio, ~$5/month (see ADR-0002) |
| Agent pattern | Raw tool-use loop | Maximum learning, no framework abstraction |
| Persistence | Local JSON files | Zero infrastructure, debuggable |
| Web search | Tavily API (tavily-python) | Free tier (1,000/month), pre-cleaned results (see ADR-0004) |
| Content extraction | trafilatura | Purpose-built for extracting article text from HTML |
| UI | CLI (terminal) | Focus on agent logic, not frontend |

## Version History

### v0.1 -- Raw Agentic Loop
- 6 tools, JSON persistence, streaming CLI, personality
- 37 unit tests
- Key learning: the prompt IS the product

### v0.2 -- Persistent Memory
- 3-tier memory model (core + working + archival)
- Observation pruning (max 30, archive evicted)
- Session summaries (auto-generate at quit, load last 5)
- Conversation window (sliding window of 20, summarize older)
- Prompt tuning for proactive context learning
- 69 tests (37 v0.1 + 32 v0.2)
- Key learning: forgetting is a feature

### v0.3 -- Web Search + Proactive Briefing
- 2 new tools: web_search (Tavily) and web_fetch (trafilatura)
- Morning briefing command (`python3 -m src.proactive`)
- Tavily API integration (1,000 free searches/month)
- 92 tests (69 v0.2 + 23 v0.3)
- Key learning: separate entry points keep the agent loop clean

## Roadmap
| Version | Focus | New Capabilities |
|---------|-------|-------------------|
| ~~v0.1~~ | ~~Raw agentic loop~~ | ~~6 tools, JSON memory, CLI~~ |
| ~~v0.2~~ | ~~Persistent memory~~ | ~~Cross-session context, session summaries, pruning~~ |
| ~~v0.3~~ | ~~Web search + proactive~~ | ~~web_search, web_fetch, morning briefing~~ |
| v0.4 | search_memory + personality | Archival memory search, inferred preferences |
| v0.5 | PWA frontend | Next.js UI, Supabase migration |
| v1.0 | Calendar + autonomy | Google Calendar, scheduled agent runs |
