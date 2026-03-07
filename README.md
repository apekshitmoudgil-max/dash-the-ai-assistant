# Dash: An Agentic Personal Task Tracker

An AI agent that doesn't just track tasks. It understands what matters to you and why.

Built with Claude Haiku 4.5 using a raw tool-use loop (no framework). Dash manages tasks, searches the web, learns your work patterns across sessions, and remembers context so every conversation picks up where you left off.

## Why I Built This

Most to-do apps treat tasks as isolated items. But tasks have context: why they matter, how they connect to your goals, and what you were thinking when you added them. I wanted an assistant that captures all of that automatically.

This is also a learning project. Instead of using LangChain or similar frameworks, I built the entire agent loop from scratch to understand exactly how tool-use agents work: message construction, tool execution, streaming, memory management, all of it.

## What It Does

- **8 tools** the agent calls autonomously: task management, user context learning, web search, and URL content extraction
- **Morning briefing** streams a daily summary of priorities, overdue items, and patterns when you start the agent
- **3-tier memory** (core / working / archival) so the agent remembers across sessions
- **Session summaries** auto-generated when you quit, loaded into the next conversation
- **Observation pruning** caps stored observations at 30, archiving older ones
- **Conversation window** keeps the last 20 messages, summarizes older ones to stay within context limits
- **Streaming** responses print to your terminal as they arrive

## Architecture

```
User starts agent
  -> Morning briefing (one-shot, streamed)
  -> Interactive REPL begins
      -> Build system prompt (inject memory + tasks + session history)
      -> Call Claude with tools
      -> LOOP (max 5 iterations):
          tool_use? -> execute locally -> send result -> call Claude again
          end_turn? -> stream response to terminal
      -> Save memory updates
  -> On quit: generate and save session summary
```

All persistence is local JSON files. No database, no infrastructure. See [docs/architecture.md](docs/architecture.md) for the full system design.

### Memory Model

| Tier | What | In Prompt? |
|------|------|:----------:|
| **Core** | User preferences, observations (max 30), active tasks, last 5 session summaries | Yes |
| **Working** | Current conversation (sliding window of 20 messages) | Yes |
| **Archival** | Full conversation logs, evicted observations | No |

Why this model? Inspired by [Letta's architecture](https://docs.letta.com/). The key insight: an agent that stuffs everything into the prompt eventually hits context limits and gets expensive. Instead, Dash keeps a bounded core (capped at ~3,270 tokens) and lets older data age out gracefully to disk. No vector DB needed at personal scale.

## Getting Started

```bash
# 1. Clone and install
git clone https://github.com/apekshitmoudgil-max/dash-agent.git
cd dash-agent
pip install -r requirements.txt

# 2. Add your API keys
cp .env.example .env
# Edit .env: add your Anthropic API key (required) and Tavily API key (optional, for web search)

# 3. Run
python3 -m src.agent
```

The agent starts with a morning briefing, then drops into an interactive conversation. Type `quit` or `exit` to end the session (Dash will auto-generate a session summary).

You can also run the morning briefing standalone: `python3 -m src.proactive`

## Make It Yours

The system prompt in `src/config.py` is where Dash gets its personality. Look for the **`# >>> CUSTOMIZE`** comments. They mark the sections you should edit:

1. **Personality**: Give your agent a voice. Funny? Formal? Direct?
2. **Learning rules**: How proactively should it pick up on your habits?
3. **Behavioral guardrails**: Should it push back when you're overloaded?

The template placeholders (`{user_context}`, `{active_tasks}`, `{recent_sessions}`) are injected automatically every turn. Don't remove those.

## Tests

```bash
python3 -m pytest tests/ -v
```

92 tests covering tool execution, web search/fetch, memory operations, observation pruning, session summaries, conversation window management, and morning briefing prompt construction.

## Design Decisions (and Why)

Every decision here was made deliberately. Full ADRs are in [docs/decisions/](docs/decisions/).

| Decision | Why | What I Considered |
|----------|-----|-------------------|
| **Claude Haiku 4.5** over Sonnet or GPT | Best personality-to-cost ratio. ~$5/month for daily use. An agent you actually use needs to be cheap. Sonnet is 3x more expensive and 2x slower for tasks that don't need deep reasoning. | Sonnet 4.6, GPT-4.1 mini, GPT-4.1 nano, Gemini 2.5 Flash |
| **Raw API loop** over LangChain/LangGraph | Maximum learning value. Frameworks abstract away exactly the parts I wanted to understand. The full loop is ~200 lines of code. | LangChain, LangGraph, CrewAI |
| **Local JSON** over a database | Zero infrastructure. `data/` is auto-created on first run. You can open `user_context.json` and see exactly what the agent knows about you. Transparency over scale. | SQLite, Supabase, pgvector |
| **3-tier memory** over flat history | Keeps costs bounded. Core memory (~3,270 tokens) is always in the prompt. Working memory slides. Archival sits on disk. The agent forgets gracefully instead of hitting context limits. | Full history (expensive), vector search (overkill at personal scale) |
| **Observation cap at 30** | Keeps the system prompt predictable. FIFO pruning (oldest out) is simple and sufficient. LLM-based relevance pruning is planned for v0.4 but not needed yet. | No cap (prompt grows unbounded), relevance scoring (adds cost per observation) |
| **Tavily** for web search | 1,000 free searches/month, no credit card required, pre-cleaned results (no HTML parsing needed). | SerpAPI (needs credit card), DuckDuckGo (no official API, unreliable), Jina Reader (URL-only) |
| **Morning briefing on startup** | The agent should be proactive, not just reactive. Showing priorities when you open Dash means you don't have to ask "what's on my plate?" every time. | Cron job (overkill for CLI), in-agent command (muddies the reactive loop) |
| **Session summaries via Haiku** | One API call at quit (~$0.003). Structured JSON with summary, tasks changed, observations, and mood. Loaded into the next session's system prompt so continuity is automatic. | Manual notes (friction), full log replay (expensive) |

## Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| v0.1 | Raw agentic loop: 6 tools, JSON memory, streaming CLI | Done |
| v0.2 | Persistent memory: cross-session context, session summaries, pruning | Done |
| v0.3 | Web search, URL fetching, morning briefing | Done |
| v0.4 | Personality modeling, priority reasoning, memory search | Planned |
| v0.5 | PWA frontend | Planned |
| v1.0 | Calendar integration, scheduled runs | Planned |

## Built With

- Python 3.9+
- [Anthropic API](https://docs.anthropic.com) (Claude Haiku 4.5)
- [Tavily](https://tavily.com) (web search)
- [trafilatura](https://trafilatura.readthedocs.io) (web content extraction)
- No agent frameworks. Just the raw API.

---

Built in public as part of my Applied AI journey. Each version ships with a LinkedIn post documenting the build.
