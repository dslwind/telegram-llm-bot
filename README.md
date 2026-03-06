# Telegram LLM Chat Bot (Python)

A Telegram bot that forwards user messages to an LLM and streams replies back.

## Features
- Streaming response (edits Telegram message while model is generating)
- SQLite persistent chat history per user
- `/reset` command to clear user history from SQLite
- User whitelist (optional)
- Per-user rate limiting (sliding window)
- Supports official OpenAI endpoint or OpenAI-compatible endpoint
- Docker deployment support

## 1) Create your Telegram bot token
1. Open Telegram and talk to `@BotFather`
2. Run `/newbot` and finish setup
3. Copy the bot token

## 2) Configure environment variables
Copy `.env.example` to `.env` and fill in values:

- `TELEGRAM_BOT_TOKEN`: from BotFather
- `OPENAI_API_KEY`: your LLM API key
- `OPENAI_MODEL`: default `gpt-4.1-mini`
- `OPENAI_BASE_URL`: optional; set only if using OpenAI-compatible provider
- `SYSTEM_PROMPT`: optional assistant system prompt
- `MAX_HISTORY_PAIRS`: history message pairs loaded from SQLite context
- `SQLITE_PATH`: SQLite file path (default `./data/chat_history.db`)
- `WHITELIST_USER_IDS`: optional comma-separated Telegram numeric user IDs
- `RATE_LIMIT_COUNT`: max requests per user in window
- `RATE_LIMIT_WINDOW_SECONDS`: rate limit window size
- `STREAM_EDIT_INTERVAL_SECONDS`: minimum seconds between Telegram edits
- `STREAM_MIN_CHARS_DELTA`: minimum new characters before next edit

## 3) Install dependencies (Conda)
### Option A: Use existing env (`llm`)
```powershell
& 'D:\Programs\miniforge3\Scripts\conda.exe' run -n llm pip install -r requirements.txt
```

### Option B: Create dedicated env
```powershell
& 'D:\Programs\miniforge3\Scripts\conda.exe' create -n tg-bot python=3.11 -y
& 'D:\Programs\miniforge3\Scripts\conda.exe' run -n tg-bot pip install -r requirements.txt
```

## 4) Run locally
```powershell
& 'D:\Programs\miniforge3\Scripts\conda.exe' run -n llm python bot.py
```

If you used `tg-bot`, replace `llm` with `tg-bot`.

## 5) Run with Docker
```powershell
docker compose up -d --build
```

Required files for Docker run:
- `.env` in project root
- `./data` folder will be mounted for SQLite persistence

Stop:
```powershell
docker compose down
```

## Notes
- Whitelist is disabled when `WHITELIST_USER_IDS` is empty.
- Rate limit is applied only to text chat requests.
- SQLite keeps full history; `MAX_HISTORY_PAIRS` controls how much context is loaded into prompts.
