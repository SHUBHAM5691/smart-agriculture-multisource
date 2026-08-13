import unittest
from unittest.mock import patch

from llm.provider import generate_text


class GroqProviderTests(unittest.TestCase):
    def test_generate_text_returns_groq_response(self):
        with patch("llm.provider.settings") as mock_settings, patch("llm.provider.urlopen") as mock_urlopen:
            mock_settings.groq_api_key = "test-key"
            mock_settings.groq_base_url = "https://api.groq.com/openai/v1"
            mock_settings.groq_model = "llama-3.3-70b-versatile"
            mock_settings.groq_max_retries = 1
            mock_settings.groq_timeout = 10
            mock_settings.groq_retry_backoff = 0
            mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                b'{"choices":[{"message":{"content":"ok"}}]}'
            )

            result = generate_text("Say hello")

        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
