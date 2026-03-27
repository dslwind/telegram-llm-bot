from dataclasses import dataclass

from telegram import Update


@dataclass(frozen=True)
class ChatSessionKey:
    chat_id: int
    user_id: int
    thread_id: int | None = None

    @property
    def normalized_thread_id(self) -> int:
        return self.thread_id if self.thread_id is not None else -1

    @property
    def supports_legacy_private_history(self) -> bool:
        return self.thread_id is None and self.chat_id == self.user_id


def build_chat_session_key(update: Update) -> ChatSessionKey | None:
    if not update.effective_chat or not update.effective_user:
        return None
    effective_message = update.effective_message
    thread_id = getattr(effective_message, "message_thread_id", None)
    return ChatSessionKey(
        chat_id=update.effective_chat.id,
        user_id=update.effective_user.id,
        thread_id=thread_id,
    )
