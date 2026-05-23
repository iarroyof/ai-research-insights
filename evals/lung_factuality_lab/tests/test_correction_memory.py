import tempfile
import unittest
from pathlib import Path

from evals.lung_factuality_lab.src.run_single import run_single


class CorrectionMemoryTests(unittest.TestCase):
    def test_scope_correction_is_reflected_in_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_single(
                scenario_id="correction_scope_tme_only_001",
                assistant_name="dummy",
                dummy_mode="scope_drift",
                out_dir=Path(tmp),
            )

            self.assertEqual(trace.scenario_id, "correction_scope_tme_only_001")
            self.assertTrue(any(j.error_type == "scope_drift" for t in trace.turns for j in t.claim_judgments))


if __name__ == "__main__":
    unittest.main()

