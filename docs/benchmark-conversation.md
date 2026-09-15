# Benchmark results

- engine: `ollama`  model: `qwen2.5:3b-instruct-q4_K_M`
- context window: 4096, history budget: 1400 tokens
- forced CPU inference: True
- threads: runtime default
- run at: 2026-09-15 21:35:03

## Realistic multi-turn conversation

| # | customer says                      | stage                | prompt tok | TTFT (ms) | total (s) |
|---|------------------------------------|----------------------|------------|-----------|-----------|
| 1 | hi                                 | greeting             | 1789       | 29867     | 31.6      |
| 2 | my order still has not turned up a | order identification | 1988       | 4446      | 10.9      |
| 3 | it is NIM-40011234 and I ordered w | policy resolution    | 2128       | 6208      | 27.8      |
| 4 | the tracking page just says label  | confirmation         | 2364       | 8632      | 25.8      |
| 5 | ok, and what happens if it never g | policy resolution    | 2595       | 8027      | 31.5      |
| 6 | actually, separate question: how m | policy resolution    | 2779       | 7779      | 12.4      |
| 7 | right, back to the parcel then. wh | policy resolution    | 2828       | 3684      | 28.5      |
| 8 | thanks, that is all                | closing              | 2967       | 22255     | 35.1      |

Model-answered turns only (guard turns are instant and excluded): TTFT mean **11362 ms**, median 7903 ms, p95 27202 ms. Total response mean 25.4 s.
