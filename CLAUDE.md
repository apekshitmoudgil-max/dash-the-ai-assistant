# Dash -- Agentic Personal Task Tracker

## What This Is
An agentic personal assistant that turns scattered intentions into focused action. Not a to-do app with AI -- an agent that understands your priorities, patterns, and goals, and uses that understanding to be proactively helpful.

## Current Version
**v0.3** -- Web search, web fetch, and morning briefing.

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| LLM | Claude Haiku 4.5 via Anthropic API |
| Agent pattern | Raw tool-use loop (no framework) |
| Persistence | Local JSON files (`data/`, auto-created on first run) |
| Web search | Tavily API (1,000 free searches/month) |
| Web fetch | requests + trafilatura (content extraction) |
| UI | CLI (terminal) |

## Commands
```bash
# Run the agent
python3 -m src.agent

# Run the morning briefing (non-interactive, prints and exits)
python3 -m src.proactive

# Run tests
python3 -m pytest tests/

# Install dependencies
pip install -r requirements.txt
```

## API Keys
- `ANTHROPIC_API_KEY` -- Get one from [console.anthropic.com](https://console.anthropic.com)
- `TAVILY_API_KEY` -- Get one from [tavily.com](https://tavily.com) (free tier: 1,000 searches/month)
- Copy `.env.example` to `.env` and add your keys

## Project Structure
```
dash/
├── src/
│   ├── agent.py            # Main agent loop (REPL + streaming)
│   ├── config.py           # Configuration + system prompt template
│   ├── tools.py            # 8 tools the agent can call
│   ├── web_tools.py        # Web search (Tavily) + web fetch (trafilatura)
│   ├── proactive.py        # Morning briefing entry point
│   ├── memory.py           # JSON persistence + observation pruning
│   ├── session_memory.py   # Cross-session summaries
│   └── context_manager.py  # Conversation window management
├── tests/                  # 92 tests
├── docs/
│   └── architecture.md     # System design documentation
├── requirements.txt
└── .env.example
```

## Customization
The system prompt in `src/config.py` is designed to be customized. Look for `# >>> CUSTOMIZE` comments -- they mark the sections where you define your agent's personality, learning behavior, and interaction style.
