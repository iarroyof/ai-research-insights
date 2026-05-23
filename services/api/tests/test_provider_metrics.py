import unittest

from app.services.provider_metrics import record_provider_call, reset_provider_metrics, snapshot_provider_metrics


class ProviderMetricsTests(unittest.TestCase):
    def setUp(self):
        reset_provider_metrics()

    def tearDown(self):
        reset_provider_metrics()

    def test_provider_metrics_count_success_failure_and_retries(self):
        record_provider_call("hf_zero_shot", status="success", latency_sec=0.2, retries=1)
        record_provider_call("hf_zero_shot", status="failure", latency_sec=0.4, retries=2, error="timeout")

        metrics = snapshot_provider_metrics()["hf_zero_shot"]

        self.assertEqual(metrics["calls"], 2)
        self.assertEqual(metrics["successes"], 1)
        self.assertEqual(metrics["failures"], 1)
        self.assertEqual(metrics["retries"], 3)
        self.assertEqual(metrics["last_error"], "timeout")
        self.assertAlmostEqual(metrics["avg_latency_sec"], 0.3)


if __name__ == "__main__":
    unittest.main()
