"""FastAPI application: REST endpoints, the /ws/chat WebSocket, and the UI.

Concurrency model
-----------------
Everything on the request path is ``async``. The only CPU-bound work -- token
decoding -- happens inside Ollama, in a different process, reached over
non-blocking HTTP. So the event loop is never blocked by generation, and a
second user connecting, resetting, or reading their history while a first user
is mid-answer is served immediately.

Two limits protect the box from itself:

* a process-wide semaphore (``max_concurrent_generations``) bounds how many
  generations are in flight, because a 4-core CPU running eight decodes at once
  makes *everyone* slow rather than sharing fairly;
* a per-session lock means one browser tab cannot start a second generation
  while its first is still running. That request is rejected with ``busy``
  rather than queued, so the client gets an immediate, honest answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError as PydanticValidationError

from app import __version__
from app.config import settings
from app.conversation.manager import ConversationManager, ValidationError
from app.conversation.session import SessionStore
from app.domain.prompts import build_system_prompt
from app.api import schemas
from app.llm import build_engine
from app.llm.base import ChatMessage, LLMError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("nimbus")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


class AppState:
    """Process-wide singletons, created on startup and closed on shutdown."""

    def __init__(self) -> None:
        self.engine = build_engine()
        self.store = SessionStore()
        self.manager = ConversationManager(self.engine)
        self.gen_semaphore = asyncio.Semaphore(settings.max_concurrent_generations)
        self.in_flight = 0


state: AppState


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global state
    state = AppState()
    log.info(
        "engine=%s model=%s ctx=%d max_concurrent=%d",
        state.engine.name,
        state.engine.model,
        settings.context_window,
        settings.max_concurrent_generations,
    )
    # Load the model into RAM *and* pre-evaluate the system-prompt prefix, so
    # the first customer is not the one who pays the ~13 s cold start. Fired as
    # a background task: the server accepts connections immediately and the
    # warmup finishes underneath.
    warmup = getattr(state.engine, "warmup", None)
    if warmup is not None:
        asyncio.create_task(_safe_warmup(warmup))
    try:
        yield
    finally:
        await state.engine.aclose()


async def _safe_warmup(warmup: Any) -> None:
    """Prime the runtime with the exact prefix every real turn will share."""
    probe = [
        ChatMessage("system", build_system_prompt()),
        ChatMessage("user", "hello"),
    ]
    try:
        started = asyncio.get_running_loop().time()
        await warmup(probe)
        elapsed = asyncio.get_running_loop().time() - started
        log.info("model warmup complete in %.1fs (prompt prefix primed)", elapsed)
    except Exception as exc:  # noqa: BLE001 - warmup must never kill startup
        log.warning("model warmup failed (continuing): %s", exc)


app = FastAPI(
    title="Nimbus Order Support Assistant",
    version=__version__,
    description=(
        "A fully local, CPU-only conversational agent for e-commerce order "
        "support. REST for session management, WebSocket for streaming chat."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# REST
# --------------------------------------------------------------------------


@app.get("/health", response_model=schemas.HealthResponse, tags=["ops"])
async def health() -> schemas.HealthResponse:
    """Liveness plus a real check that the model is actually loadable."""
    info = await state.engine.health()
    store_stats = await state.store.stats()
    reachable = bool(info.get("reachable"))
    present = bool(info.get("model_present"))
    return schemas.HealthResponse(
        status="ok" if (reachable and present) else "degraded",
        engine=str(info.get("engine", "")),
        model=str(info.get("model", "")),
        reachable=reachable,
        model_present=present,
        detail=info.get("detail"),  # type: ignore[arg-type]
        active_sessions=int(store_stats["active_sessions"]),
        version=__version__,
    )


@app.get("/api/stats", response_model=schemas.StatsResponse, tags=["ops"])
async def stats() -> schemas.StatsResponse:
    store_stats = await state.store.stats()
    return schemas.StatsResponse(
        active_sessions=int(store_stats["active_sessions"]),
        max_sessions=int(store_stats["max_sessions"]),
        ttl_seconds=int(store_stats["ttl_seconds"]),
        total_turns=int(store_stats["total_turns"]),
        in_flight_generations=state.in_flight,
        max_concurrent_generations=settings.max_concurrent_generations,
        context_window=settings.context_window,
        history_token_budget=settings.history_token_budget,
    )


@app.post("/api/session", response_model=schemas.CreateSessionResponse, tags=["session"])
async def create_session() -> schemas.CreateSessionResponse:
    session = await state.store.get_or_create(None)
    return schemas.CreateSessionResponse(
        session_id=session.session_id, stage=session.stage.value
    )


@app.get("/api/session/{session_id}", response_model=schemas.SessionView, tags=["session"])
async def get_session(session_id: str) -> schemas.SessionView:
    session = await state.store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session.")
    return schemas.SessionView(**session.as_dict())  # type: ignore[arg-type]


@app.delete("/api/session/{session_id}", response_model=schemas.CreateSessionResponse, tags=["session"])
async def reset_session(session_id: str) -> schemas.CreateSessionResponse:
    """Clear a session's history while keeping its id (the UI reset control)."""
    session = await state.store.reset(session_id)
    return schemas.CreateSessionResponse(
        session_id=session.session_id, stage=session.stage.value
    )


# --------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------


async def _send(ws: WebSocket, frame: schemas.ServerFrame) -> None:
    await ws.send_text(frame.model_dump_json())


def _stats_payload(stats_obj: Any) -> dict[str, Any]:
    if stats_obj is None:
        return {}
    return {
        "ttft_ms": round(stats_obj.ttft_ms, 1),
        "total_ms": round(stats_obj.total_ms, 1),
        "prompt_tokens": stats_obj.prompt_tokens,
        "completion_tokens": stats_obj.completion_tokens,
        "decode_tps": round(stats_obj.decode_tps, 2),
        "overall_tps": round(stats_obj.overall_tps, 2),
        "engine": stats_obj.engine,
        "model": stats_obj.model,
    }


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Streaming chat endpoint.

    Client frames: ``{"type":"chat","message":str,"session_id":str?}``,
    ``{"type":"reset"}``, ``{"type":"ping"}``.

    Server frames: ``ready``, ``start``, ``token``*, ``done``, ``context``,
    ``reset_ok``, ``pong``, ``error``.

    The socket survives every recoverable error: bad JSON, an unknown frame
    type, an over-long message, a busy session, and a model failure all produce
    an ``error`` frame and the connection stays open.
    """
    await websocket.accept()

    requested = websocket.query_params.get("session_id")
    session = await state.store.get_or_create(requested)

    await _send(
        websocket,
        schemas.ReadyFrame(
            session_id=session.session_id,
            stage=session.stage.value,
            engine=state.engine.name,
            model=state.engine.model,
            turn_count=len(session.turns),
        ),
    )

    try:
        while True:
            raw = await websocket.receive_text()

            # --- frame parsing ---------------------------------------
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await _send(
                    websocket,
                    schemas.ErrorFrame(
                        code="malformed_json",
                        message="Could not parse that frame as JSON.",
                    ),
                )
                continue

            if not isinstance(payload, dict):
                await _send(
                    websocket,
                    schemas.ErrorFrame(
                        code="malformed_frame",
                        message="A frame must be a JSON object.",
                    ),
                )
                continue

            try:
                frame = schemas.ClientFrame(**payload)
            except PydanticValidationError as exc:
                await _send(
                    websocket,
                    schemas.ErrorFrame(
                        code="invalid_frame",
                        message=_first_error(exc),
                    ),
                )
                continue

            # --- control frames --------------------------------------
            if frame.type == "ping":
                await _send(websocket, schemas.PongFrame())
                continue

            if frame.type == "reset":
                session = await state.store.reset(session.session_id)
                await _send(
                    websocket, schemas.ResetFrame(session_id=session.session_id)
                )
                continue

            # --- chat -------------------------------------------------
            if frame.message is None:
                await _send(
                    websocket,
                    schemas.ErrorFrame(
                        code="missing_message",
                        message="A chat frame must carry a 'message' field.",
                    ),
                )
                continue

            if session.lock.locked():
                await _send(
                    websocket,
                    schemas.ErrorFrame(
                        code="busy",
                        message="Still answering your previous message. "
                        "Wait for it to finish before sending another.",
                    ),
                )
                continue

            async with session.lock:
                await _handle_chat(websocket, session, frame.message)

    except WebSocketDisconnect:
        # Normal: the tab was closed or refreshed, possibly mid-stream. The
        # session stays in the store so a reconnect with the same id resumes it.
        log.info("client disconnected (session=%s)", session.session_id)
    except RuntimeError as exc:
        # Raised by Starlette when we touch a socket the client already dropped.
        log.info("socket closed during send (session=%s): %s", session.session_id, exc)
    except Exception as exc:  # noqa: BLE001 - never let one socket kill the server
        log.exception("unhandled websocket error (session=%s)", session.session_id)
        try:
            await _send(
                websocket,
                schemas.ErrorFrame(
                    code="internal_error",
                    message=f"Internal error: {type(exc).__name__}",
                    fatal=True,
                ),
            )
        except Exception:  # noqa: BLE001 - socket is already gone
            pass


async def _handle_chat(
    websocket: WebSocket, session: Any, message: str | None
) -> None:
    """Run one turn and stream it out. Must not raise."""
    try:
        async with state.gen_semaphore:
            state.in_flight += 1
            try:
                async for event in state.manager.stream_turn(session, message):
                    if event.type == "start":
                        await _send(
                            websocket,
                            schemas.StartFrame(
                                stage=event.stage,
                                intent=event.intent,
                                guarded=event.guarded,
                            ),
                        )
                    elif event.type == "token":
                        await _send(websocket, schemas.TokenFrame(text=event.text))
                    elif event.type == "done":
                        await _send(
                            websocket,
                            schemas.DoneFrame(
                                stage=event.stage,
                                intent=event.intent,
                                guarded=event.guarded,
                                corrected=event.corrected,
                                stats=_stats_payload(event.stats),
                                context=event.context,
                            ),
                        )
                    elif event.type == "correction":
                        log.warning(
                            "output verification replaced a reply (session=%s): %r",
                            session.session_id,
                            event.corrected,
                        )
                        await _send(
                            websocket, schemas.CorrectionFrame(text=event.text)
                        )
                    elif event.type == "context":
                        await _send(
                            websocket, schemas.ContextFrame(context=event.context)
                        )
                    elif event.type == "error":
                        await _send(
                            websocket,
                            schemas.ErrorFrame(code=event.code, message=event.text),
                        )
            finally:
                state.in_flight -= 1

    except ValidationError as exc:
        await _send(websocket, schemas.ErrorFrame(code=exc.code, message=str(exc)))
    except LLMError as exc:
        await _send(websocket, schemas.ErrorFrame(code=exc.code, message=str(exc)))
    except (WebSocketDisconnect, RuntimeError):
        # The client vanished mid-stream. Nothing to report to; let the outer
        # handler unwind. The generator is closed by the async-for teardown,
        # which also aborts the upstream HTTP request to Ollama.
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("turn failed")
        await _send(
            websocket,
            schemas.ErrorFrame(
                code="internal_error",
                message=f"Something went wrong handling that message ({type(exc).__name__}).",
            ),
        )


def _first_error(exc: PydanticValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid frame."
    first = errors[0]
    location = ".".join(str(p) for p in first.get("loc", ())) or "frame"
    return f"{location}: {first.get('msg', 'invalid')}"


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------

if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

else:  # pragma: no cover - only hit if the frontend folder is missing

    @app.get("/", include_in_schema=False)
    async def index() -> JSONResponse:
        return JSONResponse({"detail": "frontend/ not found"}, status_code=404)
