import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_CONFIG_TESTS") == "1", "runtime config test is opt-in")
class RuntimeProviderConfigTests(unittest.TestCase):
    def test_hosted_provider_env_reaches_settings(self):
        from app.config import settings

        self.assertEqual(settings.llm.chat_provider, "nvidia")
        self.assertEqual(settings.llm.context_manager_provider, "nvidia")
        self.assertEqual(settings.memory.nli_provider, "hf_api")
        self.assertEqual(settings.memory.nli_model, "pritamdeka/PubMedBERT-MNLI-MedNLI")
        self.assertTrue(settings.memory.hf_api_base_url.startswith("https://"))
        self.assertTrue(settings.memory.hf_api_token)

    def test_llm_client_uses_configured_nvidia_provider(self):
        from app.clients.llm import LLMClient

        client = LLMClient()
        cfg = client._provider_config("nvidia")

        self.assertTrue(cfg["base_url"].startswith("https://"))
        self.assertTrue(cfg["model"])
        self.assertTrue(cfg["api_key"])
        self.assertNotIn("\n", cfg["api_key"])
