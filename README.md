# Telegram LLM Chat Bot (Python)

A Telegram bot that forwards user messages to an LLM and streams replies back.

## Features
- Streaming response (edits Telegram message while model is generating)
- SQLite persistent chat history per user
- Persistent runtime LLM config stored in `data/config.json`
- `/new` command to start a new session
- `/model` command to show/switch the global current model
- `/models` command to open a button-based model picker
- `/reset` command to clear user history from SQLite
- User whitelist (optional)
- Per-user rate limiting (sliding window)
- Supports official OpenAI endpoint or OpenAI-compatible endpoint
- Docker deployment support

## Commands
- `/new`: clear current user history and start a new session
- `/model`: show current global model, `.env` default model, endpoint, config path, context, and rate-limit settings
- `/model <model_id>`: switch the global model and persist it to `data/config.json`
- `/models`: open a button menu with available model IDs and switch the global model by tapping
- `/reset`: clear current user history

## 1) Create your Telegram bot token
1. Open Telegram and talk to `@BotFather`
2. Run `/newbot` and finish setup
3. Copy the bot token

## 2) Configure environment variables
Copy `.env.example` to `.env` and fill in values:

- `.env` stores default values used on first start.
- On startup, the bot creates `data/config.json` if it does not exist, then uses that file as the runtime override source on later restarts.
- `TELEGRAM_BOT_TOKEN`: from BotFather
- `OPENAI_API_KEY`: default LLM API key; copied into `data/config.json` on first start
- `OPENAI_MODEL`: default model; used when `data/config.json` is missing or incomplete
- `OPENAI_BASE_URL`: default optional OpenAI-compatible endpoint; used when `data/config.json` is missing or incomplete
- `SYSTEM_PROMPT`: optional assistant system prompt
- `MAX_HISTORY_PAIRS`: history message pairs loaded from SQLite context
- `SQLITE_PATH`: SQLite file path (default `./data/chat_history.db`)
- `WHITELIST_USER_IDS`: optional comma-separated Telegram numeric user IDs
- `RATE_LIMIT_COUNT`: max requests per user in window
- `RATE_LIMIT_WINDOW_SECONDS`: rate limit window size
- `STREAM_EDIT_INTERVAL_SECONDS`: minimum seconds between Telegram edits
- `STREAM_MIN_CHARS_DELTA`: minimum new characters before next edit
- `MODELS_MENU_PAGE_SIZE`: optional; number of model buttons per page

## 3) Install dependencies
### Option A: pip (venv)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Conda
```bash
conda create -n tg-bot python=3.11 -y
conda activate tg-bot
pip install -r requirements.txt
```

## 4) Run locally
```bash
python bot.py
```

## 5) Run with Docker
```powershell
docker compose up -d --build
```

Required files for Docker run:
- `.env` in project root
- `./data` folder will be mounted for SQLite persistence and `config.json`

Stop:
```powershell
docker compose down
```

## Notes
- Whitelist is disabled when `WHITELIST_USER_IDS` is empty.
- Rate limit is applied only to text chat requests.
- SQLite keeps full history; `MAX_HISTORY_PAIRS` controls how much context is loaded into prompts.
- Runtime LLM config is loaded in this order: `.env` defaults, then `data/config.json` overrides.
