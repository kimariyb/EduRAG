// Manual acceptance: desktop rail + workspace; mobile drawer; light contrast;
// keyboard send; clear focus ring; demo warning; suggested prompts; source chips;
// streaming response; sync fallback; session history; FAQ prompt insertion.
(() => {
  "use strict";

  const SESSION_STORAGE_KEY = "edurag.session-metadata.v1";
  const DEFAULT_SOURCE_FILTER = "ai";
  const $ = (selector) => document.querySelector(selector);
  const messagesEl = $("#messages");
  const inputEl = $("#input");
  const formEl = $("#question-form");
  const sendEl = $("#send");
  const sourceFilterEl = $("#source-filter");
  const historyListEl = $("#history-list");
  const faqListEl = $("#faq-list");
  const sessionHintEl = $("#session-hint");
  const sessionCountEl = $("#session-count");
  const statusDotEl = $("#status-dot");
  const runtimeStatusEl = $("#runtime-status");
  const navEl = $("#workspace-nav");
  const navToggleEl = $("#nav-toggle");
  const drawerBackdropEl = $("#drawer-backdrop");

  let sessionId = newSessionId();
  let isSending = false;
  const sessions = new Map(loadSessionMetadata().map((session) => [session.id, session]));

  const SOURCE_META = {
    sql: { label: "FAQ match", className: "source-sql" },
    rag: { label: "Knowledge retrieval", className: "source-rag" },
    mock: { label: "Demo response", className: "source-mock" },
    history: { label: "Saved response", className: "source-history" },
    err: { label: "Server error", className: "source-error" },
  };

  function newSessionId() {
    return (crypto.randomUUID && crypto.randomUUID()) || `s-${Math.random().toString(36).slice(2)}`;
  }

  function makeElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setSession(id) {
    sessionId = id;
    const saved = sessions.get(id);
    sessionHintEl.textContent = saved ? `Session: ${saved.title}` : "New conversation";
    renderHistory();
  }

  function createWelcomeState(message) {
    const state = makeElement("div", "empty-state");
    const mark = makeElement("span", "empty-state-mark", "✦");
    mark.setAttribute("aria-hidden", "true");
    state.append(mark, makeElement("h2", "", "What would you like to explore?"));
    state.append(makeElement("p", "", message || "Ask a focused question, choose a starter prompt, or open a saved session."));
    return state;
  }

  function clearMessages(message) {
    messagesEl.replaceChildren(createWelcomeState(message));
  }

  function removeWelcomeState() {
    if (messagesEl.querySelector(".empty-state")) messagesEl.replaceChildren();
  }

  function addUserMessage(text) {
    removeWelcomeState();
    const message = makeElement("article", "message message-user");
    message.setAttribute("aria-label", "Your question");
    const label = makeElement("p", "message-label", "You");
    const bubble = makeElement("div", "message-bubble");
    bubble.textContent = text;
    message.append(label, bubble);
    messagesEl.append(message);
    scrollToBottom();
  }

  function addAssistantPlaceholder() {
    removeWelcomeState();
    const message = makeElement("article", "message message-assistant");
    message.setAttribute("aria-label", "EduRAG response");
    const header = makeElement("div", "assistant-header");
    header.append(makeElement("p", "message-label", "EduRAG"));
    const bubble = makeElement("div", "message-bubble");
    const typing = makeElement("div", "typing-indicator");
    typing.setAttribute("aria-label", "Thinking");
    for (let index = 0; index < 3; index += 1) typing.append(makeElement("span"));
    bubble.append(typing);
    message.append(header, bubble);
    messagesEl.append(message);
    scrollToBottom();
    return message;
  }

  function getAssistantContent(message) {
    const bubble = message.querySelector(".message-bubble");
    let content = bubble.querySelector(".message-content");
    if (!content) {
      content = makeElement("div", "message-content");
      bubble.querySelector(".typing-indicator")?.remove();
      bubble.append(content);
    }
    return content;
  }

  function setSourceChip(message, source) {
    const meta = SOURCE_META[source] || SOURCE_META.err;
    const header = message.querySelector(".assistant-header");
    header.querySelector(".source-chip")?.remove();
    const chip = makeElement("span", `source-chip ${meta.className}`, meta.label);
    chip.title = `Answer source: ${meta.label}`;
    header.append(chip);
  }

  function appendToken(message, token) {
    const content = getAssistantContent(message);
    content.textContent += String(token || "");
    scrollToBottom();
  }

  function finishAssistant(message, text, source) {
    setSourceChip(message, source);
    const content = getAssistantContent(message);
    content.textContent = String(text || "");
    scrollToBottom();
  }

  function appendAssistantError(message) {
    setSourceChip(message, "err");
    const bubble = message.querySelector(".message-bubble");
    const notice = makeElement("p", "stream-error", "Answer generation failed.");
    bubble.append(notice);
    scrollToBottom();
  }

  function showError(message, text) {
    finishAssistant(message, text || "We could not complete that request. Please try again later.", "err");
  }

  function truncateTitle(text) {
    const compact = String(text).replace(/\s+/g, " ").trim();
    return compact.length > 36 ? `${compact.slice(0, 33)}…` : compact || "Untitled conversation";
  }

  function loadSessionMetadata() {
    try {
      const stored = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || "[]");
      if (!Array.isArray(stored)) return [];
      return stored
        .filter((session) => session && typeof session.id === "string" && typeof session.title === "string")
        .map((session) => ({ id: session.id, title: session.title, updatedAt: Number(session.updatedAt) || 0 }))
        .slice(0, 20);
    } catch (_) {
      return [];
    }
  }

  function persistSessions() {
    const metadata = [...sessions.values()]
      .sort((left, right) => right.updatedAt - left.updatedAt)
      .slice(0, 20);
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(metadata));
    } catch (_) {
      // Browsing can continue if storage is unavailable.
    }
  }

  function rememberSession(id, title) {
    if (!id) return;
    const current = sessions.get(id);
    sessions.set(id, {
      id,
      title: title ? truncateTitle(title) : current?.title || "Untitled conversation",
      updatedAt: Date.now(),
    });
    persistSessions();
    renderHistory();
  }

  function renderHistory() {
    historyListEl.replaceChildren();
    const ordered = [...sessions.values()].sort((left, right) => right.updatedAt - left.updatedAt);
    sessionCountEl.textContent = String(ordered.length);
    sessionCountEl.setAttribute("aria-label", `${ordered.length} saved sessions`);
    if (!ordered.length) {
      const item = makeElement("li", "empty-list", "No saved sessions yet.");
      historyListEl.append(item);
      return;
    }
    for (const session of ordered) {
      const item = document.createElement("li");
      const button = makeElement("button", "nav-list-button", session.title);
      button.type = "button";
      button.title = `Open session ${session.title}`;
      if (session.id === sessionId) button.setAttribute("aria-current", "page");
      button.addEventListener("click", () => loadSession(session.id));
      item.append(button);
      historyListEl.append(item);
    }
  }

  async function loadSession(id) {
    setSession(id);
    clearMessages("Loading this saved conversation…");
    closeNavigation();
    try {
      const response = await fetch(`/api/qa/sessions/${encodeURIComponent(id)}/history`);
      if (!response.ok) throw new Error("history unavailable");
      const data = await response.json();
      const history = Array.isArray(data.history) ? data.history : [];
      if (!history.length) {
        clearMessages("This session has no messages yet.");
        return;
      }
      messagesEl.replaceChildren();
      for (const turn of history) {
        addUserMessage(turn.question || "");
        const message = addAssistantPlaceholder();
        finishAssistant(message, turn.answer || "", "history");
      }
    } catch (_) {
      clearMessages("This saved session could not be loaded. Please try again.");
    }
  }

  async function loadFaq() {
    faqListEl.replaceChildren();
    try {
      const response = await fetch("/api/faq?limit=30");
      if (!response.ok) throw new Error("FAQ unavailable");
      const data = await response.json();
      const items = Array.isArray(data.items) ? data.items : [];
      if (!items.length) {
        faqListEl.append(makeElement("li", "empty-list", "No FAQ prompts available."));
        return;
      }
      for (const faq of items) {
        const item = document.createElement("li");
        const button = makeElement("button", "nav-list-button faq-button");
        const question = makeElement("span", "faq-question", String(faq.question || "Untitled question"));
        const subject = makeElement("span", "faq-subject", String(faq.subject || "General"));
        button.type = "button";
        button.title = "Place this question in the composer";
        button.append(question, subject);
        button.addEventListener("click", () => {
          inputEl.value = question.textContent;
          autoGrow();
          inputEl.focus();
          closeNavigation();
        });
        item.append(button);
        faqListEl.append(item);
      }
    } catch (_) {
      faqListEl.append(makeElement("li", "empty-list", "FAQ prompts are unavailable."));
    }
  }

  function updateRuntimeStatus(text, status) {
    runtimeStatusEl.textContent = text;
    runtimeStatusEl.className = `runtime-status ${status}`;
  }

  async function checkHealth() {
    try {
      const response = await fetch("/health");
      const data = await response.json();
      const healthy = response.ok && data.ready && data.status === "ok";
      statusDotEl.className = `status-dot ${healthy ? "is-ready" : "is-unavailable"}`;
      if (data.mock) {
        updateRuntimeStatus("Demo mode is active. Answers use sample data and server-side chat history resets when the server restarts.", "is-demo");
      } else if (healthy) {
        updateRuntimeStatus("Live learning service connected. Your request will use the configured FAQ and knowledge sources.", "is-ready");
      } else {
        updateRuntimeStatus("The learning service is unavailable. You can still review locally saved session titles.", "is-unavailable");
      }
    } catch (_) {
      statusDotEl.className = "status-dot is-unavailable";
      updateRuntimeStatus("The learning service is unavailable. Check that the server is running and try again.", "is-unavailable");
    }
  }

  function requestBody(text) {
    return JSON.stringify({
      query: text,
      session_id: sessionId,
      source_filter: sourceFilterEl.value.trim() || DEFAULT_SOURCE_FILTER,
    });
  }

  async function sendMessage(text) {
    addUserMessage(text);
    rememberSession(sessionId, text);
    setSession(sessionId);
    const message = addAssistantPlaceholder();
    const body = requestBody(text);
    const streamState = { tokenReceived: false, done: false, serverError: false };

    try {
      const response = await fetch("/api/qa/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!response.ok || !response.body) throw new Error("stream unavailable");
      await consumeStream(response.body, message, streamState);
      if (streamState.serverError) return;
      if (!streamState.done) throw new Error("stream ended unexpectedly");
    } catch (_) {
      if (streamState.tokenReceived) {
        appendAssistantError(message);
        return;
      }
      await requestSyncFallback(body, message, text);
    }
  }

  async function requestSyncFallback(body, message, title) {
    try {
      const response = await fetch("/api/qa/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!response.ok) throw new Error("fallback unavailable");
      const data = await response.json();
      setSession(data.session_id || sessionId);
      rememberSession(data.session_id || sessionId, title);
      finishAssistant(message, data.answer || "", data.source || "rag");
    } catch (_) {
      showError(message, "We could not complete that request. Please try again later.");
    }
  }

  async function consumeStream(body, message, state) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const event = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        handleSseEvent(event, message, state);
        boundary = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) handleSseEvent(buffer, message, state);
  }

  function handleSseEvent(event, message, state) {
    const dataLine = event.split("\n").find((line) => line.startsWith("data:"));
    if (!dataLine) return;
    let payload;
    try {
      payload = JSON.parse(dataLine.slice(5).trim());
    } catch (_) {
      return;
    }
    switch (payload.type) {
      case "meta":
        if (payload.session_id) {
          setSession(payload.session_id);
          rememberSession(payload.session_id);
        }
        setSourceChip(message, payload.source || "rag");
        break;
      case "token":
        state.tokenReceived = true;
        appendToken(message, payload.content);
        break;
      case "done":
        state.done = true;
        break;
      case "error":
        state.serverError = true;
        if (state.tokenReceived) appendAssistantError(message);
        else showError(message, "Answer generation failed.");
        break;
      default:
        break;
    }
  }

  function setSending(sending) {
    isSending = sending;
    sendEl.disabled = sending;
    sendEl.setAttribute("aria-busy", String(sending));
    sendEl.querySelector(".send-label").textContent = sending ? "Sending" : "Send";
  }

  function submit() {
    const text = inputEl.value.trim();
    if (!text || isSending) return;
    inputEl.value = "";
    autoGrow();
    setSending(true);
    sendMessage(text).finally(() => setSending(false));
  }

  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
  }

  function setNavigation(open) {
    navEl.classList.toggle("is-open", open);
    drawerBackdropEl.hidden = !open;
    navToggleEl.setAttribute("aria-expanded", String(open));
    navToggleEl.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    document.body.classList.toggle("drawer-open", open);
  }

  function closeNavigation() {
    setNavigation(false);
  }

  formEl.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });
  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  });
  inputEl.addEventListener("input", autoGrow);
  $("#new-chat").addEventListener("click", () => {
    setSession(newSessionId());
    clearMessages();
    closeNavigation();
    inputEl.focus();
  });
  $("#clear-chat").addEventListener("click", async () => {
    try {
      await fetch(`/api/qa/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    } catch (_) {
      // The local UI can still return to its empty state.
    }
    clearMessages("This conversation has been cleared. Ask a new question when you are ready.");
    inputEl.focus();
  });
  $(".suggestion-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-prompt]");
    if (!button) return;
    inputEl.value = button.dataset.prompt || "";
    autoGrow();
    submit();
  });
  navToggleEl.addEventListener("click", () => setNavigation(!navEl.classList.contains("is-open")));
  drawerBackdropEl.addEventListener("click", closeNavigation);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && navEl.classList.contains("is-open")) closeNavigation();
  });

  renderHistory();
  loadFaq();
  checkHealth();
  inputEl.focus();
})();
