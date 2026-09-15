"""Deliberately break the running server and check it degrades gracefully.

The pytest suite covers these paths against the mock engine. This script runs
the same abuse against the *live* server with the real model behind it, which
is what Phase VI actually asks for: malformed input, mid-stream disconnects,
several sessions at once, and oversized payloads.

Start the server first, then:

    python scripts/failure_drill.py
    python scripts/failure_drill.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import websockets

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {mark} {name}" + (f"{DIM} -- {detail}{RESET}" if detail else ""))


def ws_url(base: str) -> str:
    return base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/chat"


async def expect_error(ws, name: str, expected_code: str | None = None) -> None:
    """Read one frame, assert it is an error, then prove the socket still works."""
    frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
    if frame.get("type") != "error":
        record(name, False, f"expected an error frame, got {frame.get('type')}")
        return
    if expected_code and frame.get("code") != expected_code:
        record(name, False, f"expected code {expected_code}, got {frame.get('code')}")
        return

    await ws.send(json.dumps({"type": "ping"}))
    pong = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
    alive = pong.get("type") == "pong"
    record(name, alive, f"code={frame.get('code')}, socket alive={alive}")


# --------------------------------------------------------------------------


async def drill_malformed(base: str) -> None:
    print("\nMalformed input (socket must survive every one)")
    async with websockets.connect(ws_url(base), ping_interval=None) as ws:
        await ws.recv()  # ready

        await ws.send("{this is not json")
        await expect_error(ws, "unparseable JSON", "malformed_json")

        await ws.send(json.dumps([1, 2, 3]))
        await expect_error(ws, "JSON array instead of object", "malformed_frame")

        await ws.send(json.dumps({"type": "nonsense"}))
        await expect_error(ws, "unknown frame type", "invalid_frame")

        await ws.send(json.dumps({"type": "chat"}))
        await expect_error(ws, "chat frame with no message", "missing_message")

        await ws.send(json.dumps({"type": "chat", "message": "   "}))
        await expect_error(ws, "whitespace-only message", "empty_message")

        await ws.send(json.dumps({"type": "chat", "message": "x" * 5000}))
        await expect_error(ws, "oversized message (5000 chars)", "invalid_frame")

        await ws.send(json.dumps({"type": "chat", "message": {"nested": "object"}}))
        await expect_error(ws, "non-string message", "invalid_frame")


async def drill_disconnect(base: str) -> None:
    print("\nDisconnect mid-stream")
    ws = await websockets.connect(ws_url(base), ping_interval=None)
    await ws.recv()
    await ws.send(json.dumps({"type": "chat", "message": "How long do refunds take?"}))

    tokens = 0
    while tokens < 3:
        frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
        if frame["type"] == "token":
            tokens += 1
        elif frame["type"] in ("done", "error"):
            break
    await ws.close()
    record("closed socket after 3 tokens", True, f"{tokens} tokens received")

    await asyncio.sleep(0.5)
    async with httpx.AsyncClient(timeout=20) as client:
        body = (await client.get(f"{base}/health")).json()
    record("server healthy after abrupt disconnect", body.get("status") == "ok",
           f"status={body.get('status')}")


async def drill_busy(base: str) -> None:
    print("\nTwo sends on one session")
    async with websockets.connect(ws_url(base), ping_interval=None) as ws:
        ready = json.loads(await ws.recv())
        session_id = ready["session_id"]

        await ws.send(json.dumps({"type": "chat", "message": "How long do refunds take?"}))
        await asyncio.wait_for(ws.recv(), timeout=90)  # start

        async with websockets.connect(
            f"{ws_url(base)}?session_id={session_id}", ping_interval=None
        ) as second:
            await second.recv()
            await second.send(json.dumps({"type": "chat", "message": "and shipping?"}))
            frame = json.loads(await asyncio.wait_for(second.recv(), timeout=30))
            record("second concurrent send is rejected, not queued",
                   frame.get("type") == "error" and frame.get("code") == "busy",
                   f"code={frame.get('code')}")

        # Drain the first generation so the session unlocks cleanly.
        while True:
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            if frame["type"] in ("done", "error"):
                break


async def drill_concurrent(base: str, users: int) -> None:
    print(f"\n{users} simultaneous sessions")
    prompts = [
        "How much is express shipping?",
        "Can I return an opened speaker?",
        "What payment methods do you take?",
        "Do you deliver on Sundays?",
        "How long do refunds take?",
    ]

    async def one(index: int) -> bool:
        try:
            async with websockets.connect(ws_url(base), ping_interval=None, open_timeout=30) as ws:
                await ws.recv()
                await ws.send(json.dumps({"type": "chat", "message": prompts[index % len(prompts)]}))
                while True:
                    frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
                    if frame["type"] == "done":
                        return True
                    if frame["type"] == "error":
                        print(f"{DIM}      user {index}: {frame['code']}{RESET}")
                        return False
        except Exception as exc:  # noqa: BLE001
            print(f"{DIM}      user {index}: {type(exc).__name__}{RESET}")
            return False

    started = time.perf_counter()
    outcomes = await asyncio.gather(*(one(i) for i in range(users)))
    elapsed = time.perf_counter() - started
    record(f"all {users} sessions completed", all(outcomes),
           f"{sum(outcomes)}/{users} in {elapsed:.1f}s")

    async with httpx.AsyncClient(timeout=20) as client:
        stats = (await client.get(f"{base}/api/stats")).json()
    record("in-flight counter returns to zero",
           stats.get("in_flight_generations") == 0,
           f"in_flight={stats.get('in_flight_generations')}, "
           f"sessions={stats.get('active_sessions')}")


async def drill_rest(base: str) -> None:
    print("\nREST misuse")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{base}/api/session/does-not-exist")
        record("unknown session id is 404", r.status_code == 404, f"status={r.status_code}")

        r = await client.get(f"{base}/api/session/../../etc/passwd")
        record("path traversal is rejected", r.status_code in (404, 400, 404),
               f"status={r.status_code}")

        r = await client.post(f"{base}/api/session")
        created = r.json().get("session_id", "")
        r = await client.delete(f"{base}/api/session/{created}")
        record("reset of an empty session succeeds", r.status_code == 200,
               f"status={r.status_code}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Failure-handling drill")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=4)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            health = (await client.get(f"{base}/health")).json()
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}Cannot reach the server at {base}{RESET}: {exc}")
        sys.exit(2)

    print(f"Target: {base} | model: {health.get('model')} | status: {health.get('status')}")

    await drill_malformed(base)
    await drill_rest(base)
    await drill_disconnect(base)
    await drill_busy(base)
    await drill_concurrent(base, args.users)

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 70)
    colour = GREEN if passed == len(results) else RED
    print(f"{colour}{passed}/{len(results)} drills passed{RESET}")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}: {detail}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
