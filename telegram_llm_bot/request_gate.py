import threading

from .session import ChatSessionKey


class SessionRequestGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_sessions: set[ChatSessionKey] = set()

    def try_acquire(self, session: ChatSessionKey) -> bool:
        with self._lock:
            if session in self._active_sessions:
                return False
            self._active_sessions.add(session)
            return True

    def release(self, session: ChatSessionKey) -> None:
        with self._lock:
            self._active_sessions.discard(session)
