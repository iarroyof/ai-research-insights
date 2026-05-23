# Lung Factuality Large Generated Conversation Corpus v1

This package adds the missing generated corpus layer for `evals/lung_factuality_lab`.

## What is included

- 8 scenario families
- 15 generated variants per family
- 120 generated conversations
- 840 user turns
- generated scenario entries
- generated gold-claim additions
- generated mechanism graph additions
- user false-premise bank
- assistant wrong-answer bank
- three-pass verification summary

## How to add to the repo

Copy the `evals/lung_factuality_lab` directory in this package over the existing package root, preserving existing files.

Example:

```bash
rsync -av /path/to/lung_factuality_large_corpus/evals/lung_factuality_lab/ ./evals/lung_factuality_lab/
```

Then run your existing tests and smoke commands:

```bash
python -m evals.lung_factuality_lab.src.run_batch   --config configs/batch_runs.yaml   --assistant wrong_answer_replay   --out runs/generated_corpus_smoke
```

If your current `run_batch` reads only the hand-authored batch config, add generated scenarios from:

```text
data/scenarios/generated_scenarios.yaml
```

## Corpus format

Each generated conversation file is JSONL with one user turn per line.

Each turn contains:

- `conversation_id`
- `scenario_id`
- `variant_index`
- `turn`
- `role`
- `text`
- `expected_behavior`
- `target_gold_claims`
- `trap_ids`
- `must_mention`
- `must_not_claim`
- `scope`
- `tags`

## Verification

See:

```text
data/verification/verification_report.md
data/verification/verification_summary.json
```

Three verification passes were applied:

1. source-grounded biomedical fact matrix;
2. polarity/false-claim placement audit;
3. schema and trace-observability audit.
