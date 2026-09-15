"""Ollama backend.

Talks to a local ``ollama serve`` over its HTTP API using ``httpx`` in async
streaming mode. Nothing here blocks the event loop, so N concurrent WebSocket
sessions each get their own in-flight HTTP stream and the server stays
responsive while the model is decoding.

Ollama itself serialises work per model by default (``OLLAMA_NUM_PARALLEL``
controls how many decode slots it runs); the point of the async client is that
*our* process never blocks -- a slow generation for one user cannot stall
another user's handshake, heartbeat, or session reset.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from app.config import settings
from app.llm.base import ChatMessage, GenerationStats, LLMError, StreamChunk

#: Ollama reports durations in nanoseconds.
_NS_PER_MS = 1_000_000


class OllamaEngine:
    """Streaming chat against a local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.model
        self._timeout = timeout or settings.generation_timeout_seconds
        # The connect timeout stays short so a dead daemon fails fast; the read
        # timeout has to cover slow CPU decoding of a long answer.
        self._client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(self._timeout, connect=5.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

    # -- options ---------------------------------------------------------

    def _options(self, temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "temperature": settings.temperature if temperature is None else temperature,
            "top_p": settings.top_p,
            "num_ctx": settings.context_window,
            "num_predict": settings.max_output_tokens if max_tokens is None else max_tokens,
            # Repetition control matters a lot for sub-2B models in long chats;
            # without it they start echoing their own previous reply.
            "repeat_penalty": 1.12,
            "repeat_last_n": 256,
        }
        if settings.num_threads > 0:
            opts["num_thread"] = settings.num_threads
        if settings.force_cpu:
            # 0 layers offloaded == pure CPU inference. Pinned per request so
            # the guarantee does not depend on how the daemon was launched:
            # Ollama otherwise grabs any discrete GPU it finds.
            opts["num_gpu"] = 0
        return opts

    # -- streaming -------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": True,
            "options": self._options(temperature, max_tokens),
            # Hold the model resident between turns so the next turn does not
            # pay the model-load cost again.
            "keep_alive": "10m",
        }
        if stop:
            payload["options"]["stop"] = stop

        stats = GenerationStats(model=self.model, engine=self.name)
        started = time.perf_counter()
        first_token_at: float | None = None
        emitted = 0

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code == 404:
                    await response.aread()
                    raise LLMError(
                        f"Model {self.model!r} is not available in Ollama. "
                        f"Run: ollama pull {self.model}",
                        code="model_not_found",
                    )
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise LLMError(
                        f"Ollama returned HTTP {response.status_code}: {body}",
                        code="upstream_error",
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # A truncated line means the daemon died mid-stream.
                        raise LLMError(
                            "Malformed response from Ollama (stream interrupted).",
                            code="bad_upstream_payload",
                        ) from None

                    if event.get("error"):
                        raise LLMError(str(event["error"]), code="upstream_error")

                    piece = (event.get("message") or {}).get("content", "")
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            stats.ttft_ms = (first_token_at - started) * 1000.0
                        emitted += 1
                        yield StreamChunk(text=piece)

                    if event.get("done"):
                        stats.total_ms = (time.perf_counter() - started) * 1000.0
                        stats.prompt_tokens = int(event.get("prompt_eval_count") or 0)
                        stats.completion_tokens = int(event.get("eval_count") or emitted)
                        stats.extra = {
                            "load_ms": (event.get("load_duration") or 0) / _NS_PER_MS,
                            "prompt_eval_ms": (event.get("prompt_eval_duration") or 0)
                            / _NS_PER_MS,
                            "eval_ms": (event.get("eval_duration") or 0) / _NS_PER_MS,
                            "done_reason": str(event.get("done_reason") or ""),
                        }
                        break

        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Is the Ollama service running?",
                code="engine_unreachable",
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(
                "The model took too long to respond and the request timed out.",
                code="engine_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Transport error talking to Ollama: {exc}", code="transport_error"
            ) from exc

        if stats.total_ms == 0.0:
            stats.total_ms = (time.perf_counter() - started) * 1000.0
        yield StreamChunk(text="", done=True, stats=stats.finalise())

    # -- one-shot --------------------------------------------------------

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

    # -- lifecycle -------------------------------------------------------

    async def health(self) -> dict[str, object]:
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return {
                "engine": self.name,
                "model": self.model,
                "reachable": False,
                "model_present": False,
                "detail": str(exc),
            }
        names = [m.get("name", "") for m in response.json().get("models", [])]
        return {
            "engine": self.name,
            "model": self.model,
            "reachable": True,
            "model_present": any(n == self.model or n.startswith(self.model) for n in names),
            "available_models": names,
        }

    async def warmup(self, messages: list[ChatMessage] | None = None) -> None:
        """Load the model and pre-evaluate the stable prompt prefix.

        Two costs are paid on a cold first turn: loading ~1 GB of weights, and
        evaluating the ~1.5k-token system prompt. Passing the real system prompt
        here pays *both* at startup, because llama.cpp keeps the KV cache for the
        longest common prefix between consecutive requests -- and that prefix is
        exactly the persona + policy block every turn shares. Measured effect on
        the first user turn: ~13 s time-to-first-token down to ~0.2 s.
        """
        probe = messages or [ChatMessage("user", "hi")]
        try:
            async for _ in self.stream_chat(probe, temperature=0.0, max_tokens=1):
                pass
        except LLMError:
            # Warmup is best-effort; /health reports the real state.
            pass

    async def aclose(self) -> None:
        await self._client.aclose()
