# grpo-lab

A minimal, readable GRPO (Group Relative Policy Optimization) training setup for small
LLMs on a task with a verifiable reward. It is built on
[TRL](https://github.com/huggingface/trl)'s `GRPOTrainer` + PEFT LoRA, fits on an
8 GB consumer GPU for a ~0.5B model, and scales to an ~8B model with one config change
(QLoRA) on a bigger GPU.

Task: [GSM8K](https://huggingface.co/datasets/openai/gsm8k) grade-school math. The model
reasons freely and ends with `#### <number>`; the reward is 1 if that number matches the
gold answer. No learned reward model, no SFT stage, no human labels.

```
prompt ──► policy samples G=8 completions ──► reward each (correct? formatted?)
                                                       │
              advantage_i = (r_i − mean(r)) / std(r)  ◄┘      (group-relative)
                                                       │
       clipped policy-gradient step on the LoRA weights ◄┘   (no critic, no ref model)
```

## Results

RESULTS_PLACEHOLDER

## Repo layout

```
grpo_lab/
  data.py        GSM8K loading, prompt template, gold-answer parsing
  rewards.py     correctness_reward (1.0 / 0.0) and format_reward (0.2 / 0.0)
  train.py       YAML config -> GRPOConfig + LoRA/QLoRA -> GRPOTrainer -> adapter
  evaluate.py    greedy pass@1 on GSM8K test (before / after), dumps every completion
  plot.py        reward / length curves from trainer_state.json
  summarize.py   markdown table from the two eval files
configs/
  qwen2.5-0.5b-gsm8k.yaml   the run reported above (8 GB GPU)
  qwen2.5-7b-gsm8k.yaml     same recipe, ~8B model, 4-bit QLoRA (needs >=24 GB GPU)
scripts/
  setup.sh                  uv venv + CUDA torch + deps
  run_experiment.sh         baseline eval -> train -> eval -> plots -> summary
results/<run_name>/         eval_before.json, eval_after.json, train_log.csv, curves.png, summary.md
tests/                      unit tests for the answer parser and rewards
```

## Quick start

```bash
git clone https://github.com/gian-g3dai/grpo-lab && cd grpo-lab
scripts/setup.sh                                   # needs uv + an NVIDIA GPU
.venv/bin/python -m pytest tests                   # reward parser sanity checks

# whole pipeline for the small model (baseline eval, GRPO, eval, plots)
scripts/run_experiment.sh configs/qwen2.5-0.5b-gsm8k.yaml

# or step by step
.venv/bin/python -m grpo_lab.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --out results/x/eval_before.json --limit 200
.venv/bin/python -m grpo_lab.train    --config configs/qwen2.5-0.5b-gsm8k.yaml --max-steps 20
.venv/bin/python -m grpo_lab.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen2.5-0.5b-gsm8k/final --out results/x/eval_after.json --limit 200
.venv/bin/python -m grpo_lab.plot     --state outputs/qwen2.5-0.5b-gsm8k/final/trainer_state.json --out results/x/curves.png
```

### Running the ~8B model

```bash
scripts/run_experiment.sh configs/qwen2.5-7b-gsm8k.yaml
```

The 7B config loads [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
in 4-bit (NF4) and trains a LoRA on top. Rough memory budget: ~5 GB weights + KV cache for
8 × 512-token rollouts + LoRA grads/optimizer ≈ 20-24 GB, so an A10G / L4 / 3090 / 4090
works. On a 40 GB+ card, uncomment the `use_vllm` block in the config for much faster
rollouts. Any HF causal LM works in the `model:` field; swap in
[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) or
[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
(gated) and adjust `target_modules` if the projection names differ.

## How the pieces fit

* **Data** (`data.py`): each GSM8K row becomes a chat prompt (system + user) and a gold
  string. The system prompt asks for step-by-step reasoning and a final `#### <number>` line.
* **Rewards** (`rewards.py`): plain Python functions with the TRL signature
  `f(completions, **columns) -> list[float]`. `correctness` parses the last `####` line and
  compares numbers; `format` pays 0.2 for exactly one `####` line at the end (dense-ish
  signal before the model ever gets a problem right, and an anti-answer-spam guard).
  TRL sums the rewards (weights configurable via `reward_weights`).
* **Trainer** (`train.py`): every key under `grpo:` in the YAML is a `trl.GRPOConfig`
  field, so the [TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer) apply
  verbatim. `lora:` builds a `peft.LoraConfig`; `quantization.load_in_4bit` builds a
  `BitsAndBytesConfig`. Output is a LoRA adapter plus `trainer_state.json`.
* **Eval** (`evaluate.py`): merges the adapter, greedy-decodes the test set in batches,
  writes accuracy, format rate and every completion to JSON so you can diff behaviours.

## Design notes / knobs that mattered

* `beta: 0.0` (no KL to a reference policy) is TRL's default and halves memory, since no
  frozen reference model is kept. With LoRA the policy can't drift far in 300 steps anyway.
* `num_generations: 8` with `per_device_train_batch_size: 8` means one prompt's whole group
  per micro-batch; `gradient_accumulation_steps` is then "prompts per optimizer step".
* `scale_rewards: group` is the original GRPO normalisation (divide by group std). Groups
  where every completion gets the same reward have zero advantage and contribute nothing;
  `frac_reward_zero_std` in the log tells you how many prompts are being wasted that way.
* Learning rate: full-parameter GRPO uses ~1e-6; LoRA needs ~10-20x that.
* HF `generate` (no vLLM) is the bottleneck on an 8 GB card. It is what makes the small
  run tractable in a single process; on a bigger GPU switch on `use_vllm`.

## References

* Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO), 2024.
* DeepSeek-AI, *DeepSeek-R1*, 2025.
* [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/grpo_trainer).
