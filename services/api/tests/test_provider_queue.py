import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from app.services.provider_queue import provider_queue, provider_slot, reset_provider_queues


class ProviderQueueTests(unittest.TestCase):
    def setUp(self):
        reset_provider_queues()

    def tearDown(self):
        reset_provider_queues()

    def test_provider_slot_bounds_concurrency(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def worker():
            nonlocal active, max_active
            with provider_slot("hf_test", timeout_sec=1):
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.03)
                with lock:
                    active -= 1

        with patch.dict("os.environ", {"HF_TEST_MAX_CONCURRENCY": "1"}, clear=False):
            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(max_active, 1)

    def test_retry_budget_is_shared_per_provider_queue(self):
        with patch.dict(
            "os.environ",
            {
                "HF_TEST_RETRY_BUDGET": "2",
                "HF_TEST_RETRY_WINDOW_SEC": "60",
            },
            clear=False,
        ):
            queue = provider_queue("hf_test")
            self.assertTrue(queue.consume_retry())
            self.assertTrue(queue.consume_retry())
            self.assertFalse(queue.consume_retry())

    def test_async_provider_slot_releases_after_use(self):
        from app.services.provider_queue import async_provider_slot

        async def run_once():
            async with async_provider_slot("hf_async_test", timeout_sec=1):
                return True

        self.assertTrue(asyncio.run(run_once()))


if __name__ == "__main__":
    unittest.main()
