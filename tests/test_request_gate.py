import unittest

from telegram_llm_bot.request_gate import SessionRequestGate
from telegram_llm_bot.session import ChatSessionKey


class SessionRequestGateTests(unittest.TestCase):
    def test_gate_rejects_overlapping_requests_for_same_session(self) -> None:
        gate = SessionRequestGate()
        session = ChatSessionKey(chat_id=1, user_id=1)

        self.assertTrue(gate.try_acquire(session))
        self.assertFalse(gate.try_acquire(session))

        gate.release(session)
        self.assertTrue(gate.try_acquire(session))

    def test_gate_treats_different_sessions_independently(self) -> None:
        gate = SessionRequestGate()
        session_a = ChatSessionKey(chat_id=1, user_id=1)
        session_b = ChatSessionKey(chat_id=1, user_id=1, thread_id=99)

        self.assertTrue(gate.try_acquire(session_a))
        self.assertTrue(gate.try_acquire(session_b))


if __name__ == "__main__":
    unittest.main()
