"""Deterministic in-process engine used by the test-suite.

It exists so that the conversation manager, the WebSocket protocol, the session
store and the frontend contract can all be tested without a 1 GB model and
without minutes of CPU decoding. It streams word-by-word with a configurable
per-token delay so that streaming, cancellation and concurrency behave the same
way they do against a real backend.

It is never selected unless ``NIMBUS_ENGINE=mock`` is set explicitly.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from app.llm.base import ChatMessage, GenerationStats, LLMError, StreamChunk


class MockEngine:
    """Echo-style engine with controllable timing and failure injection."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-1b-instruct-q4",
        *,
        token_delay: float = 0.002,
        first_token_delay: float = 0.02,
        canned: str | None = None,
        fail_with: LLMError | None = None,
        fail_after_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.token_delay = token_delay
        self.first_token_delay = first_token_delay
        self.canned = canned
        self.fail_with = fail_with
        self.fail_after_tokens = fail_after_tokens
        #: Every prompt the engine was handed, for prompt-construction assertions.
        self.calls: list[list[ChatMessage]] = []

    def _reply_for(self, messages: list[ChatMessage]) -> str:
        if self.canned is not None:
            return self.canned
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        return f"Acknowledged: {last_user.strip()[:160]}"

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(list(messages))

        if self.fail_with is not None and self.fail_after_tokens is None:
            raise self.fail_with

        started = time.perf_counter()
        words = self._reply_for(messages).split(" ")
        if max_tokens is not None:
            words = words[:max_tokens]

        ttft_ms = 0.0
        emitted = 0
        for index, word in enumerate(words):
            await asyncio.sleep(self.first_token_delay if index == 0 else self.token_delay)
            if index == 0:
                ttft_ms = (time.perf_counter() - started) * 1000.0
            emitted += 1
            yield StreamChunk(text=word if index == 0 else " " + word)
            if self.fail_after_tokens is not None and emitted >= self.fail_after_tokens:
                raise self.fail_with or LLMError("injected mid-stream failure", "upstream_error")

        prompt_chars = sum(len(m.content) for m in messages)
        stats = GenerationStats(
            ttft_ms=ttft_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
            prompt_tokens=max(prompt_chars // 4, 1),
            completion_tokens=emitted,
            model=self.model,
            engine=self.name,
        )
        yield StreamChunk(text="", done=True, stats=stats.finalise())

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.stream_chat(
            messages, temperature=temperature, max_tokens=max_tokens
        ):
            if not chunk.done:
                parts.append(chunk.text)
        return "".join(parts).strip()

    async def health(self) -> dict[str, object]:
        return {
            "engine": self.name,
            "model": self.model,
            "reachable": True,
            "model_present": True,
        }

    async def warmup(self, messages: list[ChatMessage] | None = None) -> None:
        return None

    async def aclose(self) -> None:
        return None
