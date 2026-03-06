# Telegram LLM Chat Bot (Python)

A minimal Telegram bot that forwards user messages to an LLM and sends the reply back.

## Features
- Telegram polling bot
- Per-user conversation history (in memory)
- `/reset` command to clear history
- Supports official OpenAI endpoint or OpenAI-compatible endpoint

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
- `MAX_HISTORY_PAIRS`: number of user+assistant pairs to keep per user
- `SYSTEM_PROMPT`: optional assistant system prompt

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

## 4) Run
### Run without activation (recommended)
```powershell
& 'D:\Programs\miniforge3\Scripts\conda.exe' run -n llm python bot.py
```

If you used `tg-bot`, replace `llm` with `tg-bot`.

## Notes
- History is kept in memory; restarting the process clears all history.
- For production, add persistence, rate limiting, and logging/monitoring.
