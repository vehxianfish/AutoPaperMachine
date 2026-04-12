import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm import LLMClient, create_openai_compatible_client_from_env, resolve_openai_compatible_runtime_config


class LlmBaseUrlTest(unittest.TestCase):
    def _mock_response(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        return resp

    @patch("llm.requests.post")
    def test_chat_appends_v1_when_base_is_root(self, mock_post):
        mock_post.return_value = self._mock_response()
        client = LLMClient(
            api_key="test-key",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com",
        )

        client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(
            mock_post.call_args.args[0],
            "https://api.openai.com/v1/chat/completions",
        )

    @patch("llm.requests.post")
    def test_chat_keeps_versioned_base(self, mock_post):
        mock_post.return_value = self._mock_response()
        client = LLMClient(
            api_key="test-key",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
        )

        client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(
            mock_post.call_args.args[0],
            "https://api.openai.com/v1/chat/completions",
        )

    @patch("llm.requests.post")
    def test_chat_uses_full_endpoint_directly(self, mock_post):
        mock_post.return_value = self._mock_response()
        client = LLMClient(
            api_key="test-key",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1/chat/completions",
        )

        client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(
            mock_post.call_args.args[0],
            "https://api.openai.com/v1/chat/completions",
        )

    def test_runtime_config_prefers_openai_env(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                "OPENAI_MODEL": "gpt-4.1-mini",
            },
            clear=True,
        ):
            runtime = resolve_openai_compatible_runtime_config(
                api_key_names=["OPENAI_API_KEY"],
                base_url_names=["OPENAI_BASE_URL"],
                model_names=["OPENAI_MODEL"],
            )
        self.assertEqual(runtime["api_key"], "openai-key")
        self.assertEqual(runtime["base_url"], "https://api.openai.com/v1")
        self.assertEqual(runtime["model"], "gpt-4.1-mini")

    def test_create_runtime_client_from_openai_env(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                "OPENAI_MODEL": "gpt-4.1-mini",
            },
            clear=True,
        ):
            client = create_openai_compatible_client_from_env(
                api_key_names=["OPENAI_API_KEY"],
                base_url_names=["OPENAI_BASE_URL"],
                model_names=["OPENAI_MODEL"],
            )
        self.assertIsInstance(client, LLMClient)
        self.assertEqual(client.api_key, "openai-key")
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.model, "gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
