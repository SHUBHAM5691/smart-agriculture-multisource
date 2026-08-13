import unittest
from unittest.mock import patch

from memory import session_manager as sm


class SessionManagerTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        self.session_patch = patch.object(sm.st, "session_state", self.state)
        self.session_patch.start()

    def tearDown(self):
        self.session_patch.stop()

    def test_memory_contains_only_messages_and_prediction(self):
        sm.initialize_session()
        sm.add_message("user", "hi")
        prediction = {"type": "crop", "crop": "rice"}
        sm.store_prediction(prediction)

        memory = sm.get_memory()

        self.assertEqual(set(memory), {"messages", "prediction"})
        self.assertEqual(memory["messages"], [{"role": "user", "content": "hi"}])
        self.assertIs(memory["prediction"], prediction)

    def test_clear_session_resets_memory(self):
        sm.initialize_session()
        sm.add_message("user", "hi")
        sm.store_prediction({"type": "crop", "crop": "rice"})

        sm.clear_session()

        self.assertEqual(self.state, {"messages": [], "prediction": None})


if __name__ == "__main__":
    unittest.main()
