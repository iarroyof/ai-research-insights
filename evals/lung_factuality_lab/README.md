# Lung Cancer Synthetic Conversation Evaluation Lab

This package is an evaluation laboratory for lung-cancer factuality, reward quality, multi-turn memory consistency, and scope-drift behavior.

It is intentionally modular:

- scenario data lives under `data/`;
- reward weights live under `configs/`;
- assistant behavior is isolated behind adapters in `src/assistant_adapters.py`;
- every run writes full traces and reports under `runs/` or a caller-provided output directory.

The lab is meant to remain useful across implementation commits. It evaluates traces, extracted claims, mechanism coverage, failure ownership, recommendations, and regression plans rather than only reporting a scalar score.

## Run One Scenario

```bash
python -m evals.lung_factuality_lab.src.run_single \
  --scenario expert_hgf_met_direction_001 \
  --assistant dummy \
  --out evals/lung_factuality_lab/runs/test_expert_hgf_met
```

## Run Batch

```bash
python -m evals.lung_factuality_lab.src.run_batch \
  --config configs/batch_runs.yaml \
  --assistant dummy \
  --out evals/lung_factuality_lab/runs/batch_001
```

## Live Reward Shaping Ladder

Use this protocol when continuing reward shaping against the live chatbot endpoint:

1. Read `configs/reward_shape_registry.yaml` first. Do not revisit registered rejected shapes.
2. Diagnose one failing live conversation or one saved-live replay trace before editing the evaluator or reward path.
3. Make one narrow change tied to that observed failure and add a regression test.
4. Use saved-answer replay to isolate the evaluator/reward delta.
5. Run the relevant live family microfit before broader validation:
   - `configs/generated_microfit_correction_scope.yaml`
   - `configs/generated_microfit_tam_cd8.yaml`
6. Require a stratified live guard before treating the shape as stable:
   - `configs/generated_sentinel_a.yaml`
   - `configs/generated_sentinel_b.yaml`
   - `configs/generated_sentinel_c.yaml`
7. Keep `configs/generated_semantic_drift_holdout_v1.yaml` protected until the candidate has enough fit and guard evidence.

The unit of diagnosis may be one conversation. The unit of acceptance is not one conversation: require sibling variants plus a stratified guard so the reward shape does not overfit a single generated wording. Replay-only gains are diagnostic; promotion requires live endpoint evidence. Record each candidate, baseline, metric delta, blocked shape, and next allowed action in `configs/reward_shape_registry.yaml`.

## Compare Runs

```bash
python -m evals.lung_factuality_lab.src.compare_runs \
  --before evals/lung_factuality_lab/runs/batch_001 \
  --after evals/lung_factuality_lab/runs/batch_002 \
  --out evals/lung_factuality_lab/runs/comparison_001
```

## Generate Regression Plan

```bash
python -m evals.lung_factuality_lab.src.regression_planner \
  --from evals/lung_factuality_lab/runs/batch_001/failure_board.json \
  --out evals/lung_factuality_lab/runs/batch_001/regression_plan.yaml
```

## Per-Run Outputs

Each scenario run writes:

- `scenario.yaml`
- `generated_conversation.jsonl`
- `assistant_answers.jsonl`
- `extracted_claims.jsonl`
- `claim_judgments.jsonl`
- `turn_scores.jsonl`
- `conversation_trace.json`
- `failure_board.json`
- `simulation_report.md`
- `recommendations.json`
- `recommendations.md`
- `regression_plan.yaml`

## Data Format

The files use `.yaml` extensions but are JSON-compatible YAML. The loader uses PyYAML if installed and otherwise falls back to Python's standard `json` module.

## Asset Separation

The lab keeps three assets separate:

- Scenario: what the simulation tests, including target claims, mechanism graphs, and success criteria.
- Seed conversation: the user turns, expected behavior, scope, traps, must-mention concepts, and must-not-claim concepts.
- Wrong-answer bank: known bad assistant responses used to validate the evaluator/reward layer without depending on the live chatbot.

Seed conversations live in `data/conversations/seed/*.jsonl`. Generated variants can be written to `data/conversations/generated/` or directly to run outputs as `generated_conversation.jsonl`.

## Wrong-Answer Replay

Use the replay adapter to test whether the evaluator catches known bad answers:

```bash
python -m evals.lung_factuality_lab.src.run_single \
  --scenario expert_hgf_met_direction_001 \
  --assistant wrong_answer_replay \
  --out evals/lung_factuality_lab/runs/evaluator_fixture_hgf_met
```

## Generated Variants

Use `--variant-index` to perturb trap turns from `user_false_premise_bank.yaml`:

```bash
python -m evals.lung_factuality_lab.src.run_single \
  --scenario expert_hgf_met_direction_001 \
  --assistant dummy \
  --variant-index 1 \
  --out evals/lung_factuality_lab/runs/hgf_met_variant_001
```
