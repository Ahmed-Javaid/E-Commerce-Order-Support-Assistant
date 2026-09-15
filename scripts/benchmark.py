"""Inference latency benchmarks.

Measures what the README reports, against the real local model:

  --latency     time-to-first-token and decode throughput, cold and warm
  --context     how TTFT scales as the prompt grows (the reason the context
                policy exists at all)
  --concurrent  N simultaneous sessions through the running FastAPI server
  --calibrate   the character-per-token estimator against Ollama's own
                prompt_eval_count

Run everything with ``--all``. Results print as markdown tables, ready to paste.

Usage:
    python scripts/benchmark.py --all
    python scripts/benchmark.py --concurrent 4 --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows defaults redirected stdout to cp1252, which mangles the
# punctuation in these reports. Force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings  # noqa: E402
from app.conversation.tokens import CHARS_PER_TOKEN, estimate_tokens  # noqa: E402
from app.domain.policy import Stage  # noqa: E402
from app.domain.prompts import build_system_prompt  # noqa: E402
from app.llm.base import ChatMessage  # noqa: E402
from app.llm.ollama_engine import OllamaEngine  # noqa: E402

PROMPTS: list[str] = [
    "My order has not arrived and it has been 8 days.",
    "How much does express shipping cost?",
    "Can I return a pair of earbuds I already opened?",
    "I need to cancel order NIM-40011234, it was placed an hour ago.",
    "My smart plug stopped working after three weeks. What are my options?",
    "How long do refunds take once you receive the item back?",
]


@dataclass
class Sample:
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    decode_tps: float
    prompt_eval_ms: float
    load_ms: float


def p(values: list[float], pct: float) -> float:
    """Percentile with linear interpolation; ``statistics.quantiles`` needs n>1."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([line, rule, *body])


async def one_shot(engine: OllamaEngine, messages: list[ChatMessage]) -> Sample:
    ttft = total = 0.0
    prompt_tokens = completion_tokens = 0
    decode_tps = prompt_eval_ms = load_ms = 0.0
    async for chunk in engine.stream_chat(messages):
        if chunk.done and chunk.stats:
            s = chunk.stats
            ttft, total = s.ttft_ms, s.total_ms
            prompt_tokens, completion_tokens = s.prompt_tokens, s.completion_tokens
            decode_tps = s.decode_tps
            prompt_eval_ms = float(s.extra.get("prompt_eval_ms", 0.0))
            load_ms = float(s.extra.get("load_ms", 0.0))
    return Sample(ttft, total, prompt_tokens, completion_tokens, decode_tps, prompt_eval_ms, load_ms)


# --------------------------------------------------------------------------
# 1. latency
# --------------------------------------------------------------------------


async def bench_latency(engine: OllamaEngine, rounds: int) -> None:
    print("\n## Single-turn latency\n")
    system = build_system_prompt(Stage.POLICY_RESOLUTION)

    # A genuine uncached first token. The marker changes the shared prefix, so
    # the runtime cannot reuse any cached KV and has to evaluate the whole
    # prompt -- which is exactly what a first turn costs when the server has NOT
    # pre-warmed the prefix.
    marker = f"[benchmark cold {time.time_ns()}]\n"
    cold = await one_shot(
        engine,
        [ChatMessage("system", marker + system), ChatMessage("user", PROMPTS[0])],
    )
    prompt_eval_tps = (
        cold.prompt_tokens / (cold.prompt_eval_ms / 1000.0) if cold.prompt_eval_ms else 0.0
    )
    print(
        f"**Uncached first token** (prompt prefix not in the KV cache): "
        f"TTFT {cold.ttft_ms / 1000:.2f}s -- prompt-eval {cold.prompt_eval_ms / 1000:.2f}s "
        f"for {cold.prompt_tokens} tokens ({prompt_eval_tps:.0f} tok/s prompt throughput), "
        f"plus model load {cold.load_ms / 1000:.2f}s.\n"
    )
    print(
        "This is the cost the startup warmup and the stable prompt prefix exist to "
        "avoid; the warm figures below are what a user actually experiences.\n"
    )

    samples: list[Sample] = []
    for round_index in range(rounds):
        for prompt in PROMPTS:
            samples.append(
                await one_shot(
                    engine, [ChatMessage("system", system), ChatMessage("user", prompt)]
                )
            )
        print(f"  round {round_index + 1}/{rounds} done", file=sys.stderr)

    ttfts = [s.ttft_ms for s in samples]
    totals = [s.total_ms for s in samples]
    tps = [s.decode_tps for s in samples if s.decode_tps]
    out = [s.completion_tokens for s in samples]

    print(
        table(
            ["metric", "mean", "median", "p95", "min", "max"],
            [
                ["time to first token (ms)", f"{statistics.fmean(ttfts):.0f}",
                 f"{statistics.median(ttfts):.0f}", f"{p(ttfts, 0.95):.0f}",
                 f"{min(ttfts):.0f}", f"{max(ttfts):.0f}"],
                ["total response (s)", f"{statistics.fmean(totals) / 1000:.2f}",
                 f"{statistics.median(totals) / 1000:.2f}", f"{p(totals, 0.95) / 1000:.2f}",
                 f"{min(totals) / 1000:.2f}", f"{max(totals) / 1000:.2f}"],
                ["decode throughput (tok/s)", f"{statistics.fmean(tps):.1f}",
                 f"{statistics.median(tps):.1f}", f"{p(tps, 0.95):.1f}",
                 f"{min(tps):.1f}", f"{max(tps):.1f}"],
                ["completion length (tokens)", f"{statistics.fmean(out):.0f}",
                 f"{statistics.median(out):.0f}", f"{p([float(x) for x in out], 0.95):.0f}",
                 f"{min(out)}", f"{max(out)}"],
            ],
        )
    )
    print(f"\n({len(samples)} warm generations, {rounds} rounds over {len(PROMPTS)} prompts.)")


# --------------------------------------------------------------------------
# 2. context scaling
# --------------------------------------------------------------------------


async def bench_context(engine: OllamaEngine) -> None:
    """TTFT against prompt length, separating KV-cache hits from misses.

    Both columns matter and they say different things:

    * **cache miss** is the true cost of a prompt of that size -- what you pay on
      the first turn of a session, after a restart, or when concurrent sessions
      recycle the runtime's cache slots. It is linear in prompt length, and it is
      the number the context budget exists to bound.
    * **cache hit** is the steady-state cost mid-conversation, when the runtime
      only has to evaluate the tokens appended since the last turn.

    A miss is forced by putting a unique marker at the *front* of the system
    prompt, which changes the shared prefix and invalidates the cached KV.
    """
    print("\n## Prompt length vs time-to-first-token\n")
    system = build_system_prompt(Stage.POLICY_RESOLUTION)
    filler_turn = (
        "I ordered a pair of the wireless earbuds last month and the left one "
        "keeps disconnecting when I walk away from my phone. "
    )

    rows: list[list[str]] = []
    for run_index, extra_turns in enumerate((0, 4, 8, 16, 24)):
        history: list[ChatMessage] = []
        for index in range(extra_turns):
            history.append(ChatMessage("user", f"{filler_turn} (message {index})"))
            history.append(ChatMessage("assistant", "Understood, let me check the policy on that."))

        # Cache MISS: a unique prefix marker defeats prefix reuse.
        marker = f"[benchmark run {run_index} {time.time_ns()}]\n"
        miss_messages = [
            ChatMessage("system", marker + system),
            *history,
            ChatMessage("user", PROMPTS[1]),
        ]
        miss = await one_shot(engine, miss_messages)

        # Cache HIT: identical prompt, second time.
        hit = await one_shot(engine, miss_messages)

        rows.append(
            [
                str(extra_turns),
                str(miss.prompt_tokens),
                f"{miss.ttft_ms:.0f}",
                f"{hit.ttft_ms:.0f}",
                f"{miss.prompt_eval_ms:.0f}",
                f"{miss.decode_tps:.1f}",
            ]
        )
        print(f"  {extra_turns} extra turns -> {miss.prompt_tokens} prompt tokens", file=sys.stderr)

    print(
        table(
            [
                "extra turns",
                "prompt tokens",
                "TTFT, cache miss (ms)",
                "TTFT, cache hit (ms)",
                "prompt-eval on miss (ms)",
                "decode (tok/s)",
            ],
            rows,
        )
    )
    print(
        f"\nThe miss column is linear in prompt length -- that is the cost the "
        f"{settings.history_token_budget}-token history budget bounds. The hit column "
        f"is flat, which is what the stable prompt prefix buys during a conversation."
    )


# --------------------------------------------------------------------------
# 3. concurrency (through the running server)
# --------------------------------------------------------------------------


async def bench_concurrent(url: str, users: int) -> None:
    """Drive N independent WebSocket sessions at once against the live server."""
    import websockets

    print(f"\n## {users} concurrent sessions\n")
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/chat"

    async def one_user(index: int) -> tuple[float, float, int] | None:
        prompt = PROMPTS[index % len(PROMPTS)]
        try:
            async with websockets.connect(ws_url, open_timeout=20, ping_interval=None) as ws:
                await ws.recv()  # ready
                started = time.perf_counter()
                await ws.send(json.dumps({"type": "chat", "message": prompt}))
                ttft = 0.0
                tokens = 0
                while True:
                    frame = json.loads(await ws.recv())
                    if frame["type"] == "token":
                        if tokens == 0:
                            ttft = (time.perf_counter() - started) * 1000.0
                        tokens += 1
                    elif frame["type"] == "done":
                        return ttft, (time.perf_counter() - started) * 1000.0, tokens
                    elif frame["type"] == "error":
                        print(f"  user {index}: error {frame['code']}: {frame['message']}")
                        return None
        except Exception as exc:  # noqa: BLE001 - benchmark, report and continue
            print(f"  user {index}: {type(exc).__name__}: {exc}")
            return None

    wall_start = time.perf_counter()
    results = await asyncio.gather(*(one_user(i) for i in range(users)))
    wall = time.perf_counter() - wall_start

    ok = [r for r in results if r]
    if not ok:
        print("No session completed. Is the server running (`uvicorn app.api.main:app`)?")
        return

    ttfts = [r[0] for r in ok]
    totals = [r[1] for r in ok]
    print(
        table(
            ["metric", "value"],
            [
                ["sessions completed", f"{len(ok)}/{users}"],
                ["wall-clock for the batch (s)", f"{wall:.2f}"],
                ["TTFT mean / p95 (ms)", f"{statistics.fmean(ttfts):.0f} / {p(ttfts, 0.95):.0f}"],
                ["total mean / p95 (s)", f"{statistics.fmean(totals) / 1000:.2f} / {p(totals, 0.95) / 1000:.2f}"],
                ["server concurrency cap", str(settings.max_concurrent_generations)],
            ],
        )
    )


# --------------------------------------------------------------------------
# 4. token-estimator calibration
# --------------------------------------------------------------------------


async def bench_calibrate(engine: OllamaEngine) -> None:
    """Check the chars-per-token constant the memory budget is built on."""
    print("\n## Token estimator calibration\n")
    system = build_system_prompt(Stage.POLICY_RESOLUTION)

    rows: list[list[str]] = []
    errors: list[float] = []
    for prompt in PROMPTS:
        messages = [ChatMessage("system", system), ChatMessage("user", prompt)]
        sample = await one_shot(engine, messages)
        estimated = sum(estimate_tokens(m.content) for m in messages)
        actual = sample.prompt_tokens
        if not actual:
            continue
        error = (estimated - actual) / actual * 100.0
        errors.append(error)
        rows.append([prompt[:38], str(estimated), str(actual), f"{error:+.1f}%"])

    print(table(["prompt", "estimated", "actual", "error"], rows))
    if errors:
        mean_error = statistics.fmean(errors)
        implied = CHARS_PER_TOKEN * (1 + mean_error / 100.0)
        print(
            f"\nCHARS_PER_TOKEN = {CHARS_PER_TOKEN}; mean error {mean_error:+.1f}% "
            f"(max |error| {max(abs(e) for e in errors):.1f}%). "
            f"Implied best-fit value: {implied:.2f}."
        )


# --------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Nimbus latency benchmarks")
    parser.add_argument("--all", action="store_true", help="run every benchmark")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--concurrent", type=int, default=0, metavar="N")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    run_all = args.all or not (args.latency or args.context or args.calibrate or args.concurrent)

    print("# Benchmark results\n")
    print(f"- engine: `{settings.engine}`  model: `{settings.model}`")
    print(f"- context window: {settings.context_window}, history budget: "
          f"{settings.history_token_budget} tokens")
    print(f"- forced CPU inference: {settings.force_cpu}")
    print(f"- threads: {settings.num_threads or 'runtime default'}")
    print(f"- run at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    engine = OllamaEngine()
    info = await engine.health()
    if not info.get("reachable"):
        print("\nOllama is not reachable. Start it, then re-run.")
        return
    if not info.get("model_present"):
        print(f"\nModel {settings.model} is not pulled. Run: ollama pull {settings.model}")
        return

    try:
        if run_all or args.latency:
            await bench_latency(engine, args.rounds)
        if run_all or args.context:
            await bench_context(engine)
        if run_all or args.calibrate:
            await bench_calibrate(engine)
        if args.concurrent or run_all:
            await bench_concurrent(args.url, args.concurrent or 4)
    finally:
        await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
