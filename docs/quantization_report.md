## Speed measurement

Speed was measured on fixed real RAG prompts with `temperature=0`, `num_ctx=8192`,
collecting Ollama generation metrics (`eval_count` / `eval_duration`) where available.

| Model | Quant | gen_tps (tok/s) | wall_med | words_med | prompt_s |
|---|---|---:|---:|---:|---:|
| qwen2.5:7b | Q4_K_M (default) | N/A* | 3.32s | 16.0 | 0.81s |
| qwen2.5:7b-instruct-q2_K | Q2_K | N/A* | 3.35s | 10.0 | 1.06s |
| qwen2.5:7b-instruct-q3_K_M | Q3_K_M | 73.4 | 4.05s | 37.0 | 0.91s |
| qwen2.5:7b-instruct-q5_K_M | Q5_K_M | 67.1 | 4.69s | 44.0 | 0.83s |

\* `gen_tps` was not returned by Ollama for the baseline and Q2_K runs (short/truncated
answers, metrics structure differed). This is a measurement gap, not zero throughput;
a direct tok/s comparison of baseline vs Q5_K_M is therefore not available from this run.

† `prompt_s` values in this capture are incorrect due to a unit-conversion bug in the
benchmark harness (nanoseconds were not divided by 1e9 for sub-second values). The field
is excluded from conclusions until re-measured with the fixed converter.

### Reading the numbers

- Among candidates with valid `gen_tps`, **Q3_K_M decodes ~9% faster than Q5_K_M**
  (73.4 vs 67.1 tok/s), consistent with its smaller weight footprint.
- Q5_K_M has higher wall time (4.69s vs 4.05s) but produces longer coherent answers
  (44 vs 37 words median), i.e. it does more useful work per request rather than being
  uniformly slower.
- `wall_tps~` (words/sec) diverges from `gen_tps` because words ≠ tokens and wall time
  includes prompt processing; `gen_tps` is the canonical decode-speed metric here.
- The baseline and Q2_K show short median answers (16 / 10 words) partly because of
  language collapse / truncation observed in the quality gate, so their wall times are
  not comparable as "speed" without the quality context.

Console evidence: `docs/assets/speed_summary.png`. Raw metrics: `data/speed_benchmark.json`.