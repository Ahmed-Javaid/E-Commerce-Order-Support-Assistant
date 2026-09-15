# Benchmark results

- engine: `ollama`  model: `qwen2.5:3b-instruct-q4_K_M`
- context window: 4096, history budget: 1400 tokens
- forced CPU inference: True
- threads: runtime default
- run at: 2026-09-15 21:12:04

## Realistic multi-turn conversation

| # | customer says                      | stage                | prompt tok | TTFT (ms) | total (s) |
|---|------------------------------------|----------------------|------------|-----------|-----------|
| 1 | hi                                 | greeting             | 1786       | 4963      | 9.3       |
| 2 | my order still has not turned up a | order identification | 2052       | 9385      | 22.7      |
| 3 | it is NIM-40011234 and I ordered w | policy resolution    | 2214       | 12704     | 29.3      |
| 4 | the tracking page just says label  | confirmation         | 2355       | 14292     | 31.4      |
| 5 | ok, and what happens if it never g | policy resolution    | 2551       | 17897     | 36.1      |
| 6 | actually, separate question: how m | policy resolution    | 2608       | 19059     | 25.9      |
| 7 | right, back to the parcel then. wh | policy resolution    | 2664       | 19760     | 32.0      |
| 8 | thanks, that is all                | closing              | 2776       | 22448     | 25.1      |

Model-answered turns only (guard turns are instant and excluded): TTFT mean **15064 ms**, median 16095 ms, p95 21507 ms. Total response mean 26.5 s.
