import html

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from .session import ChatSessionKey
from .storage import SQLiteChatStore


def build_sessions_text(store: SQLiteChatStore, session: ChatSessionKey) -> str:
    sessions = store.list_managed_sessions(session)
    active = store.get_active_managed_session(session)
    lines = [
        "<b>Sessions</b>",
        f"current: <code>#{active['id']}</code> {html.escape(str(active['title']))}",
        "",
    ]
    for item in sessions[:20]:
        marker = "▶ " if item["active"] else "  "
        status = " archived" if item["status"] == "archived" else ""
        lines.append(
            f"{marker}<code>#{item['id']}</code> {html.escape(str(item['title']))}{status}"
        )
    if len(sessions) > 20:
        lines.append(f"... and {len(sessions) - 20} more")
    lines.append("")
    lines.append("Use /new to create, /stash to archive current, or tap buttons below.")
    return "\n".join(lines)


def build_sessions_keyboard(store: SQLiteChatStore, session: ChatSessionKey) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("+ New session", callback_data="sessions:new")]
    ]
    for item in store.list_managed_sessions(session)[:12]:
        sid = int(item["id"])
        title = str(item["title"])
        label = ("▶ " if item["active"] else "") + title
        if item["status"] == "archived":
            label += " [archived]"
        rows.append(
            [
                InlineKeyboardButton(label[:48], callback_data=f"sessions:switch:{sid}"),
                InlineKeyboardButton("Rename", callback_data=f"sessions:rename:{sid}"),
                InlineKeyboardButton("Archive", callback_data=f"sessions:archive:{sid}"),
                InlineKeyboardButton(
                    "Delete",
                    callback_data=f"sessions:confirm_delete:{sid}",
                ),
            ]
        )
    return InlineKeyboardMarkup(rows)


def build_delete_confirm_keyboard(session_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Delete permanently",
                    callback_data=f"sessions:delete:{session_id}",
                ),
            ],
            [InlineKeyboardButton("Cancel", callback_data="sessions:refresh")],
        ]
    )
