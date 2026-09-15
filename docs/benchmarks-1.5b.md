# Benchmark results

- engine: `ollama`  model: `qwen2.5:1.5b-instruct-q4_K_M`
- context window: 4096, history budget: 1400 tokens
- forced CPU inference: True
- threads: runtime default
- run at: 2026-09-15 20:01:41

## Single-turn latency

**Uncached first token** (prompt prefix not in the KV cache): TTFT 17.07s -- prompt-eval 12.93s for 1747 tokens (135 tok/s prompt throughput), plus model load 4.12s.

This is the cost the startup warmup and the stable prompt prefix exist to avoid; the warm figures below are what a user actually experiences.

| metric                     | mean | median | p95   | min  | max   |
|----------------------------|------|--------|-------|------|-------|
| time to first token (ms)   | 990  | 277    | 2288  | 213  | 12862 |
| total response (s)         | 4.45 | 3.77   | 12.08 | 0.71 | 16.16 |
| decode throughput (tok/s)  | 21.4 | 21.2   | 23.0  | 20.7 | 23.2  |
| completion length (tokens) | 73   | 70     | 134   | 11   | 230   |

(18 warm generations, 3 rounds over 6 prompts.)
