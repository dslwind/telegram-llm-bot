# Telegram LLM Chat Bot (Python)

A Telegram bot that forwards user messages to an LLM and streams replies back.

## Features
- Streaming response (edits Telegram message while model is generating)
- SQLite persistent chat history per user
- Persistent multi-provider runtime config stored in `data/config.json`
- `/new` command to start a new session
- `/model` command to show/switch the current provider model
- `/models` command to choose a provider and then a model
- `/providers` command to show and switch configured providers
- Telegram-based provider add/edit/delete management
- Telegram command menu is synced automatically on bot startup
- `/reset` command to clear user history from SQLite
- User whitelist (optional)
- Per-user rate limiting for normal chat requests
- Supports official OpenAI endpoint or OpenAI-compatible endpoint
- Docker deployment support

## Commands
- `/new`: clear current user history and start a new session
- `/model`: show current provider, current model, config path, context, and rate-limit settings
- `/model <model_id>`: switch the current provider model and persist it to `data/config.json`
- `/reasoning`: show current provider reasoning effort
- `/reasoning <effort>`: switch the current provider reasoning effort and persist it to `data/config.json`
- `/models`: pick a provider first, then pick a model from that provider
- `/providers`: show provider summary, provider ids, and provider switch buttons
- `/provider_add`: start the provider creation wizard
- `/provider_edit <provider_id>`: start the provider edit wizard
- `/provider_delete <provider_id>`: delete a provider after inline confirmation
- `/provider_cancel`: cancel an active provider wizard
- `/skip`: keep the current field value during provider edit
- `/reset`: clear current user history

## 1) Create your Telegram bot token
1. Open Telegram and talk to `@BotFather`
2. Run `/newbot` and finish setup
3. Copy the bot token

## 2) Configure environment variables
Copy `.env.example` to `.env` and fill in values:

- `.env` stores bootstrap defaults for the first provider only.
- On startup, the bot creates `data/config.json` if it does not exist, then uses that file as the runtime provider source on later restarts.
- Existing flat `data/config.json` files with `openai_api_key/openai_model/openai_base_url` are migrated automatically to the multi-provider format.
- `TELEGRAM_BOT_TOKEN`: from BotFather
- `OPENAI_API_KEY`: bootstrap API key used when creating the initial default provider
- `OPENAI_MODEL`: bootstrap model used for the initial default provider
- `OPENAI_REASONING_EFFORT`: optional bootstrap reasoning effort for the initial default provider
- `OPENAI_BASE_URL`: bootstrap optional OpenAI-compatible endpoint for the initial default provider
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
- Rate limit is applied only to normal text and photo chat requests; provider wizard input bypasses it.
- SQLite keeps full history; `MAX_HISTORY_PAIRS` controls how much context is loaded into prompts.
- Runtime config lives in `data/config.json` version 2 format with one global current provider and one remembered current model per provider.
- API keys are masked in menus and summaries. During provider setup, the bot tries to delete the API key message after it is received.
