import unittest
from unittest.mock import Mock

from telegram_llm_bot.request_gate import SessionRequestGate
from telegram_llm_bot.session import ChatSessionKey


class SessionRequestGateTests(unittest.TestCase):
    def test_gate_rejects_overlapping_requests_for_same_session(self) -> None:
        gate = SessionRequestGate()
        session = ChatSessionKey(chat_id=1, user_id=1)
        task = Mock()

        self.assertTrue(gate.try_register(session, task))
        self.assertFalse(gate.try_register(session, Mock()))

        gate.release(session, task)
        self.assertTrue(gate.try_register(session, Mock()))

    def test_gate_treats_different_sessions_independently(self) -> None:
        gate = SessionRequestGate()
        session_a = ChatSessionKey(chat_id=1, user_id=1)
        session_b = ChatSessionKey(chat_id=1, user_id=1, thread_id=99)

        self.assertTrue(gate.try_register(session_a, Mock()))
        self.assertTrue(gate.try_register(session_b, Mock()))

    def test_cancel_returns_active_request_and_invokes_task_cancel(self) -> None:
        gate = SessionRequestGate()
        session = ChatSessionKey(chat_id=1, user_id=1)
        task = Mock()
        task.done.return_value = False

        self.assertTrue(gate.try_register(session, task))
        gate.set_message_ref(session, 10, 20)

        result = gate.cancel(session)
        self.assertTrue(result.cancelled)
        self.assertIsNotNone(result.active_request)
        self.assertEqual(result.active_request.chat_id, 10)
        self.assertEqual(result.active_request.message_id, 20)
        task.cancel.assert_called_once()

    def test_cancel_does_not_report_success_for_completed_task(self) -> None:
        gate = SessionRequestGate()
        session = ChatSessionKey(chat_id=1, user_id=1)
        task = Mock()
        task.done.return_value = True

        self.assertTrue(gate.try_register(session, task))
        gate.set_message_ref(session, 10, 20)

        result = gate.cancel(session)
        self.assertFalse(result.cancelled)
        self.assertIsNotNone(result.active_request)
        task.cancel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
