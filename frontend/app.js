/* Nimbus Support chat client.
 *
 * One WebSocket to /ws/chat. Frames in and out are JSON objects with a `type`
 * discriminator (see app/api/schemas.py). Responsibilities:
 *   - keep the socket alive and reconnect with backoff,
 *   - render streamed tokens as they arrive, without reflowing the whole thread,
 *   - keep the session id in sessionStorage so a refresh resumes the chat,
 *   - surface errors inline instead of failing silently.
 */

(() => {
  "use strict";

  const SESSION_KEY = "nimbus.session_id";
  const MAX_BACKOFF_MS = 8000;

  const el = {
    thread: document.getElementById("thread"),
    welcome: document.getElementById("welcome"),
    suggestions: document.getElementById("suggestions"),
    composer: document.getElementById("composer"),
    input: document.getElementById("input"),
    send: document.getElementById("send-btn"),
    stop: document.getElementById("stop-btn"),
    reset: document.getElementById("reset-btn"),
    details: document.getElementById("details-btn"),
    status: document.getElementById("status"),
    statusText: document.getElementById("status-text"),
    banner: document.getElementById("banner"),
    modelBadge: document.getElementById("model-badge"),
  };

  const state = {
    ws: null,
    sessionId: null,
    streaming: false,
    showTelemetry: false,
    backoff: 500,
    current: null, // { bubble, meta, text }
    pingTimer: null,
    manualClose: false,
  };

  /* ----------------------------------------------------------- utilities */

  function setStatus(kind, text) {
    el.status.dataset.state = kind;
    el.statusText.textContent = text;
  }

  function showBanner(message, tone = "error") {
    el.banner.textContent = message;
    el.banner.dataset.tone = tone;
    el.banner.hidden = false;
  }

  function hideBanner() {
    el.banner.hidden = true;
  }

  function atBottom() {
    const { scrollTop, scrollHeight, clientHeight } = el.thread;
    return scrollHeight - scrollTop - clientHeight < 120;
  }

  function scrollToBottom(force = false) {
    if (force || atBottom()) {
      el.thread.scrollTop = el.thread.scrollHeight;
    }
  }

  function hideWelcome() {
    if (el.welcome && !el.welcome.hidden) el.welcome.hidden = true;
  }

  /* ------------------------------------------------------------ rendering */

  function addMessage(role, text, opts = {}) {
    hideWelcome();

    const wrap = document.createElement("div");
    wrap.className = `msg ${role}` + (opts.guard ? " guard" : "");

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "You" : "Ava";
    avatar.setAttribute("aria-hidden", "true");

    const col = document.createElement("div");
    col.className = "bubble-col";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text || "";
    if (!text) bubble.classList.add("empty");

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.hidden = true;

    col.append(bubble, meta);
    wrap.append(avatar, col);
    el.thread.appendChild(wrap);
    scrollToBottom(true);

    return { wrap, bubble, meta };
  }

  function chip(text, cls = "") {
    const span = document.createElement("span");
    span.className = "chip " + cls;
    span.textContent = text;
    return span;
  }

  function renderMeta(meta, frame) {
    meta.replaceChildren();

    if (frame.stage) meta.appendChild(chip(frame.stage.replace(/_/g, " "), "stage"));
    if (frame.guarded) meta.appendChild(chip("guard: " + frame.guarded, "guard"));
    if (frame.corrected) {
      meta.appendChild(chip("corrected: invented order data", "guard"));
    }

    const s = frame.stats || {};
    const c = frame.context || {};
    if (state.showTelemetry) {
      if (s.ttft_ms != null) meta.appendChild(chip(`ttft ${Math.round(s.ttft_ms)}ms`));
      if (s.total_ms != null) meta.appendChild(chip(`total ${(s.total_ms / 1000).toFixed(1)}s`));
      if (s.decode_tps) meta.appendChild(chip(`${s.decode_tps.toFixed(1)} tok/s`));
      if (s.prompt_tokens) meta.appendChild(chip(`prompt ${s.prompt_tokens}t`));
      if (s.completion_tokens) meta.appendChild(chip(`out ${s.completion_tokens}t`));
      if (c.turns_verbatim != null) {
        meta.appendChild(chip(`ctx ${c.turns_verbatim}/${c.turns_total} turns`));
      }
      if (c.turns_evicted_total) meta.appendChild(chip(`evicted ${c.turns_evicted_total}`));
      if (c.has_summary) meta.appendChild(chip("summary on"));
      if (c.pinned_facts && c.pinned_facts.length) {
        meta.appendChild(chip("pinned: " + c.pinned_facts.join(", ")));
      }
    }

    meta.hidden = meta.childElementCount === 0;
  }

  /* -------------------------------------------------------------- sending */

  function setStreaming(on) {
    state.streaming = on;
    el.input.disabled = on;
    el.send.hidden = on;
    el.send.disabled = on;
    el.stop.hidden = !on;
    setStatus(on ? "thinking" : "online", on ? "Ava is typing" : "Connected");
  }

  function send() {
    const text = el.input.value.trim();
    if (!text || state.streaming) return;

    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      showBanner("Not connected. Reconnecting…");
      connect();
      return;
    }

    hideBanner();
    addMessage("user", text);
    el.input.value = "";
    autosize();

    state.current = addMessage("assistant", "");
    state.current.text = "";
    setStreaming(true);

    state.ws.send(JSON.stringify({ type: "chat", message: text }));
  }

  /* ---------------------------------------------------------- ws handling */

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const id = state.sessionId ? `?session_id=${encodeURIComponent(state.sessionId)}` : "";
    return `${proto}//${location.host}/ws/chat${id}`;
  }

  function connect() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    setStatus("connecting", "Connecting");
    state.manualClose = false;

    let ws;
    try {
      ws = new WebSocket(wsUrl());
    } catch (err) {
      scheduleReconnect();
      return;
    }
    state.ws = ws;

    ws.onopen = () => {
      state.backoff = 500;
      hideBanner();
      setStatus("online", "Connected");
      clearInterval(state.pingTimer);
      // Keeps intermediaries from idling the socket shut during a long think.
      state.pingTimer = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
      }, 25000);
    };

    ws.onmessage = (event) => {
      let frame;
      try {
        frame = JSON.parse(event.data);
      } catch {
        return;
      }
      handleFrame(frame);
    };

    ws.onclose = () => {
      clearInterval(state.pingTimer);
      if (state.streaming) {
        finishStream("Connection closed before the reply finished.");
      }
      if (!state.manualClose) {
        setStatus("offline", "Reconnecting");
        scheduleReconnect();
      }
    };

    ws.onerror = () => {
      // onclose always follows; reconnect logic lives there.
    };
  }

  function scheduleReconnect() {
    const delay = state.backoff;
    state.backoff = Math.min(state.backoff * 2, MAX_BACKOFF_MS);
    setTimeout(connect, delay);
  }

  function finishStream(errorText) {
    if (state.current) {
      if (!state.current.text && errorText) {
        state.current.bubble.textContent = errorText;
      }
      state.current.bubble.classList.remove("empty");
    }
    state.current = null;
    setStreaming(false);
  }

  function handleFrame(frame) {
    switch (frame.type) {
      case "ready": {
        state.sessionId = frame.session_id;
        try { sessionStorage.setItem(SESSION_KEY, frame.session_id); } catch {}
        el.modelBadge.textContent = `${frame.engine}:${frame.model}`;
        setStatus("online", "Connected");
        break;
      }

      case "start": {
        if (state.current) {
          state.current.stage = frame.stage;
          state.current.guarded = frame.guarded;
          if (frame.guarded) state.current.wrap.classList.add("guard");
        }
        break;
      }

      case "token": {
        if (!state.current) break;
        state.current.text += frame.text;
        state.current.bubble.classList.remove("empty");
        // appendChild on a text node avoids re-laying-out the whole bubble on
        // every token, which matters at 15-20 tokens/second.
        state.current.bubble.appendChild(document.createTextNode(frame.text));
        scrollToBottom();
        break;
      }

      case "correction": {
        // Output verification caught the reply claiming live order visibility.
        // The streamed text is already on screen, so replace it wholesale.
        if (state.current) {
          state.current.text = frame.text;
          state.current.bubble.textContent = frame.text;
          state.current.bubble.classList.remove("empty");
          state.current.wrap.classList.add("corrected");
          state.current.corrected = true;
        }
        scrollToBottom();
        break;
      }

      case "done": {
        if (state.current) renderMeta(state.current.meta, frame);
        finishStream();
        scrollToBottom();
        break;
      }

      case "context": {
        // Rolling-summary refresh finished after the answer was displayed.
        const last = el.thread.querySelector(".msg.assistant:last-of-type .meta");
        if (last && state.showTelemetry && frame.context && frame.context.summary) {
          last.appendChild(chip("summary refreshed"));
          last.hidden = false;
        }
        break;
      }

      case "error": {
        showBanner(frame.message || "Something went wrong.");
        if (state.current) {
          // Drop the empty placeholder bubble; the banner carries the message.
          if (!state.current.text) state.current.wrap.remove();
          finishStream();
        }
        if (frame.fatal) setStatus("offline", "Disconnected");
        break;
      }

      case "reset_ok": {
        el.thread.querySelectorAll(".msg").forEach((n) => n.remove());
        if (el.welcome) el.welcome.hidden = false;
        hideBanner();
        finishStream();
        setStatus("online", "Connected");
        break;
      }

      case "pong":
        break;

      default:
        break;
    }
  }

  /* ---------------------------------------------------------------- input */

  function autosize() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 180) + "px";
  }

  el.input.addEventListener("input", autosize);

  el.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  el.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    send();
  });

  el.stop.addEventListener("click", () => {
    // The server has no cancel frame: stopping closes the socket, which aborts
    // the upstream generation, then we reconnect to the same session.
    state.manualClose = true;
    if (state.ws) state.ws.close();
    finishStream();
    state.manualClose = false;
    connect();
  });

  el.reset.addEventListener("click", () => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "reset" }));
    } else {
      el.thread.querySelectorAll(".msg").forEach((n) => n.remove());
      if (el.welcome) el.welcome.hidden = false;
    }
    el.input.focus();
  });

  el.details.addEventListener("click", () => {
    state.showTelemetry = !state.showTelemetry;
    el.details.setAttribute("aria-pressed", String(state.showTelemetry));
    // Telemetry chips are rendered at `done` time, so toggling only affects
    // future turns; say so rather than silently doing nothing.
    showBanner(
      state.showTelemetry
        ? "Telemetry on — latency and context stats will show under each new reply."
        : "Telemetry off.",
      "info"
    );
    setTimeout(hideBanner, 3200);
  });

  if (el.suggestions) {
    el.suggestions.addEventListener("click", (event) => {
      const button = event.target.closest(".suggestion");
      if (!button) return;
      el.input.value = button.dataset.text || "";
      autosize();
      send();
    });
  }

  /* ------------------------------------------------------------ bootstrap */

  try {
    state.sessionId = sessionStorage.getItem(SESSION_KEY);
  } catch {
    state.sessionId = null;
  }

  connect();
  el.input.focus();
})();
