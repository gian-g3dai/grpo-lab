# Paired analysis

`Qwen/Qwen2.5-0.5B-Instruct` (base) vs. + GRPO LoRA (`outputs/qwen2.5-0.5b-gsm8k/final`) on 1319 GSM8K test problems, greedy. Rows split the problems by what the *base* model produced, so both models are scored on the same set.

## Strict scoring: number in the last `\boxed{}` (as in summary.md)

| base model output | n | base acc | GRPO acc | net problems gained |
|---|---|---|---|---|
| numeric `\boxed{}` | 1117 | 46.2% | 50.3% | +46 |
| no box, hit the 512-token limit | 67 | 0.0% | 28.4% | +19 |
| no box, stopped on its own | 135 | 0.0% | 54.8% | +74 |
| all | 1319 | 39.1% | 49.7% | +139 |

**Δ accuracy: +10.5 points**, paired bootstrap 95% CI [+7.7, +13.4] (10000 resamples). 261 problems wrong→right, 122 right→wrong, exact McNemar p = 1.0e-12.

## Lenient scoring: the `\boxed{}` answer if present, else the last number in the completion

| base model output | n | base acc | GRPO acc | net problems gained |
|---|---|---|---|---|
| numeric `\boxed{}` | 1117 | 46.2% | 50.3% | +46 |
| no box, hit the 512-token limit | 67 | 6.0% | 28.4% | +15 |
| no box, stopped on its own | 135 | 44.4% | 54.8% | +14 |
| all | 1319 | 44.0% | 49.7% | +75 |

**Δ accuracy: +5.7 points**, paired bootstrap 95% CI [+3.0, +8.5] (10000 resamples). 208 problems wrong→right, 133 right→wrong, exact McNemar p = 5.8e-05.
