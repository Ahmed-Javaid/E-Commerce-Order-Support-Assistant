# Benchmark results

- engine: `ollama`  model: `qwen2.5:3b-instruct-q4_K_M`
- context window: 4096, history budget: 1400 tokens
- forced CPU inference: True
- threads: runtime default
- run at: 2026-09-15 19:53:55

## Single-turn latency

**Uncached first token** (prompt prefix not in the KV cache): TTFT 27.49s -- prompt-eval 27.44s for 1747 tokens (64 tok/s prompt throughput), plus model load 0.00s.

This is the cost the startup warmup and the stable prompt prefix exist to avoid; the warm figures below are what a user actually experiences.

| metric                     | mean | median | p95   | min  | max   |
|----------------------------|------|--------|-------|------|-------|
| time to first token (ms)   | 518  | 493    | 738   | 347  | 776   |
| total response (s)         | 6.96 | 6.20   | 12.01 | 2.13 | 16.25 |
| decode throughput (tok/s)  | 12.1 | 12.0   | 13.6  | 9.2  | 13.7  |
| completion length (tokens) | 75   | 68     | 129   | 24   | 154   |

(18 warm generations, 3 rounds over 6 prompts.)

## Prompt length vs time-to-first-token

| extra turns | prompt tokens | TTFT, cache miss (ms) | TTFT, cache hit (ms) | prompt-eval on miss (ms) | decode (tok/s) |
|-------------|---------------|-----------------------|----------------------|--------------------------|----------------|
| 0           | 1743          | 26774                 | 136                  | 26733                    | 9.2            |
| 4           | 1959          | 31340                 | 125                  | 31263                    | 11.3           |
| 8           | 2175          | 33728                 | 129                  | 33676                    | 11.1           |
| 16          | 2613          | 42185                 | 139                  | 42130                    | 11.1           |
| 24          | 3053          | 50352                 | 141                  | 50270                    | 10.7           |

The miss column is linear in prompt length -- that is the cost the 1400-token history budget bounds. The hit column is flat, which is what the stable prompt prefix buys during a conversation.

## Token estimator calibration

| prompt                                 | estimated | actual | error |
|----------------------------------------|-----------|--------|-------|
| My order has not arrived and it has be | 1782      | 1723   | +3.4% |
| How much does express shipping cost?   | 1779      | 1717   | +3.6% |
| Can I return a pair of earbuds I alrea | 1782      | 1723   | +3.4% |
| I need to cancel order NIM-40011234, i | 1786      | 1734   | +3.0% |
| My smart plug stopped working after th | 1787      | 1724   | +3.7% |
| How long do refunds take once you rece | 1784      | 1722   | +3.6% |

CHARS_PER_TOKEN = 4.0; mean error +3.5% (max |error| 3.7%). Implied best-fit value: 4.14.

## 4 concurrent sessions

| metric                       | value         |
|------------------------------|---------------|
| sessions completed           | 4/4           |
| wall-clock for the batch (s) | 54.64         |
| TTFT mean / p95 (ms)         | 28061 / 44676 |
| total mean / p95 (s)         | 34.87 / 52.14 |
| server concurrency cap       | 4             |
