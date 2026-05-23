import os
import unittest
from unittest.mock import Mock, patch


class ZeroShotHfApiTests(unittest.TestCase):
    def test_hf_api_payload_and_response_mapping(self):
        from app.services import zero_shot

        response = Mock()
        response.json.return_value = {
            "sequence": "Aspirin inhibits platelet aggregation.",
            "labels": ["biomedical", "other"],
            "scores": [0.91, 0.09],
        }
        response.raise_for_status.return_value = None

        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.post.return_value = response

        env = {
            "ZERO_SHOT_PROVIDER": "hf_api",
            "HF_API_TOKEN": "hf_test_token",
            "HF_API_BASE_URL": "https://router.huggingface.co/hf-inference/models",
            "ZERO_SHOT_MODEL": "facebook/bart-large-mnli",
        }
        with patch.dict(os.environ, env, clear=False), patch("httpx.Client", return_value=client):
            result = zero_shot.score_labels(
                ["Aspirin inhibits platelet aggregation."],
                ["biomedical", "other"],
            )

        self.assertEqual(result, [{"biomedical": 0.91, "other": 0.09}])
        client.post.assert_called_once()
        _, kwargs = client.post.call_args
        self.assertEqual(
            kwargs["json"],
            {
                "inputs": "Aspirin inhibits platelet aggregation.",
                "parameters": {
                    "candidate_labels": ["biomedical", "other"],
                    "multi_label": True,
                },
            },
        )
        self.assertIn("Authorization", kwargs["headers"])

    def test_local_provider_uses_lazy_local_pipeline(self):
        from app.services import zero_shot

        fake_pipeline = Mock()
        fake_pipeline.return_value = {
            "labels": ["biomedical"],
            "scores": [0.7],
        }

        with patch.dict(os.environ, {"ZERO_SHOT_PROVIDER": "local"}, clear=False), patch.object(
            zero_shot, "_get_nli", return_value=fake_pipeline
        ):
            result = zero_shot.score_labels(["text"], ["biomedical"])

        self.assertEqual(result, [{"biomedical": 0.7}])

    def test_hf_api_parses_element_style_response(self):
        from app.services import zero_shot

        data = [
            {"label": "biomedical", "score": 0.82},
            {"label": "other", "score": 0.18},
        ]

        self.assertEqual(
            zero_shot._parse_zero_shot_response(data),
            {"biomedical": 0.82, "other": 0.18},
        )

    def test_hf_api_parses_nested_element_style_response(self):
        from app.services import zero_shot

        data = [[{"label": "biomedical", "score": 0.82}]]

        self.assertEqual(
            zero_shot._parse_zero_shot_response(data),
            {"biomedical": 0.82},
        )

    def test_hf_api_batches_multiple_texts(self):
        from app.services import zero_shot

        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.json.return_value = [
            {"labels": ["biomedical", "other"], "scores": [0.91, 0.09]},
            {"labels": ["biomedical", "other"], "scores": [0.22, 0.78]},
        ]
        response.raise_for_status.return_value = None

        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.post.return_value = response

        env = {
            "ZERO_SHOT_PROVIDER": "hf_api",
            "HF_API_TOKEN": "hf_test_token",
            "ZERO_SHOT_HF_API_BATCH_SIZE": "8",
        }
        with patch.dict(os.environ, env, clear=False), patch("httpx.Client", return_value=client):
            result = zero_shot.score_labels(
                ["Aspirin inhibits platelet aggregation.", "The market closed higher."],
                ["biomedical", "other"],
            )

        self.assertEqual(
            result,
            [{"biomedical": 0.91, "other": 0.09}, {"biomedical": 0.22, "other": 0.78}],
        )
        client.post.assert_called_once()
        _, kwargs = client.post.call_args
        self.assertEqual(
            kwargs["json"]["inputs"],
            ["Aspirin inhibits platelet aggregation.", "The market closed higher."],
        )

    def test_hf_api_batch_size_one_keeps_multiple_requests(self):
        from app.services import zero_shot

        first = Mock()
        first.status_code = 200
        first.headers = {}
        first.json.return_value = {"labels": ["biomedical"], "scores": [0.9]}
        first.raise_for_status.return_value = None

        second = Mock()
        second.status_code = 200
        second.headers = {}
        second.json.return_value = {"labels": ["biomedical"], "scores": [0.2]}
        second.raise_for_status.return_value = None

        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.post.side_effect = [first, second]

        env = {
            "ZERO_SHOT_PROVIDER": "hf_api",
            "HF_API_TOKEN": "hf_test_token",
            "ZERO_SHOT_HF_API_BATCH_SIZE": "1",
        }
        with patch.dict(os.environ, env, clear=False), patch("httpx.Client", return_value=client):
            result = zero_shot.score_labels(["one", "two"], ["biomedical"])

        self.assertEqual(result, [{"biomedical": 0.9}, {"biomedical": 0.2}])
        self.assertEqual(client.post.call_count, 2)

    def test_hf_api_retries_retryable_status(self):
        from app.services import zero_shot

        retry_response = Mock()
        retry_response.status_code = 503
        retry_response.headers = {}

        ok_response = Mock()
        ok_response.status_code = 200
        ok_response.headers = {}
        ok_response.json.return_value = {"labels": ["biomedical"], "scores": [0.93]}
        ok_response.raise_for_status.return_value = None

        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.post.side_effect = [retry_response, ok_response]

        env = {
            "ZERO_SHOT_PROVIDER": "hf_api",
            "HF_API_TOKEN": "hf_test_token",
            "ZERO_SHOT_HF_API_MAX_RETRIES": "1",
            "ZERO_SHOT_HF_API_RETRY_BACKOFF_SEC": "0",
        }
        with patch.dict(os.environ, env, clear=False), patch("httpx.Client", return_value=client):
            result = zero_shot.score_labels(["text"], ["biomedical"])

        self.assertEqual(result, [{"biomedical": 0.93}])
        self.assertEqual(client.post.call_count, 2)

    def test_hf_api_uses_retry_after_header(self):
        from app.services import zero_shot

        response = Mock()
        response.headers = {"retry-after": "3"}

        self.assertEqual(zero_shot._retry_delay(1, 2.0, response), 3.0)


if __name__ == "__main__":
    unittest.main()
