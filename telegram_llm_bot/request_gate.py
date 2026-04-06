import threading
from dataclasses import dataclass

from .session import ChatSessionKey


@dataclass(frozen=True)
class ActiveSessionRequest:
    task: object
    chat_id: int | None = None
    message_id: int | None = None


@dataclass(frozen=True)
class CancelResult:
    active_request: ActiveSessionRequest | None
    cancelled: bool


class SessionRequestGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_sessions: dict[ChatSessionKey, ActiveSessionRequest] = {}

    def try_register(self, session: ChatSessionKey, task: object) -> bool:
        with self._lock:
            if session in self._active_sessions:
                return False
            self._active_sessions[session] = ActiveSessionRequest(task=task)
            return True

    def set_message_ref(
        self,
        session: ChatSessionKey,
        chat_id: int,
        message_id: int,
    ) -> None:
        with self._lock:
            active = self._active_sessions.get(session)
            if active is None:
                return
            self._active_sessions[session] = ActiveSessionRequest(
                task=active.task,
                chat_id=chat_id,
                message_id=message_id,
            )

    def cancel(self, session: ChatSessionKey) -> CancelResult:
        with self._lock:
            active = self._active_sessions.get(session)
        if active is None:
            return CancelResult(active_request=None, cancelled=False)

        done = getattr(active.task, "done", None)
        if callable(done):
            try:
                if done():
                    return CancelResult(active_request=active, cancelled=False)
            except Exception:
                pass

        cancel = getattr(active.task, "cancel", None)
        if callable(cancel):
            cancel()
            return CancelResult(active_request=active, cancelled=True)

        return CancelResult(active_request=active, cancelled=False)

    def release(self, session: ChatSessionKey, task: object | None = None) -> None:
        with self._lock:
            active = self._active_sessions.get(session)
            if active is None:
                return
            if task is not None and active.task is not task:
                return
            self._active_sessions.pop(session, None)
