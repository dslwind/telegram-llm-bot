import logging
import os

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from .constants import CONFIG_PATH
from .handlers import (
    handle_photo,
    handle_text,
    help_command,
    model_command,
    models_command,
    new_session_command,
    provider_add_command,
    provider_callback_router,
    provider_cancel_command,
    provider_delete_command,
    provider_edit_command,
    providers_command,
    reasoning_command,
    reset_command,
    stop_command,
    skip_command,
    start_command,
)
from .runtime import SQLITE_PATH, TELEGRAM_BOT_TOKEN, WHITELIST_USER_IDS, chat_store
from .ui import sync_bot_commands


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    if WHITELIST_USER_IDS:
        logging.info("Whitelist enabled for %d users", len(WHITELIST_USER_IDS))
    logging.info("SQLite chat storage path: %s", os.path.abspath(SQLITE_PATH))
    logging.info("Runtime config path: %s", os.path.abspath(CONFIG_PATH))

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(sync_bot_commands).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler(["new", "newchat"], new_session_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("reasoning", reasoning_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("providers", providers_command))
    app.add_handler(CommandHandler("provider_add", provider_add_command))
    app.add_handler(CommandHandler("provider_edit", provider_edit_command))
    app.add_handler(CommandHandler("provider_delete", provider_delete_command))
    app.add_handler(CommandHandler("provider_cancel", provider_cancel_command))
    app.add_handler(CommandHandler("skip", skip_command))
    app.add_handler(
        CallbackQueryHandler(provider_callback_router, pattern=r"^(providers|models):")
    )
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        chat_store.close()
