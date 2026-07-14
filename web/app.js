/* EduRAG frontend: chat UI with streaming answers and FAQ management. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const messagesEl = $("#messages");
  const inputEl = $("#input");
  const sendEl = $("#send");
  const historyListEl = $("#history-list");
  const faqListEl = $("#faq-list");
  const sessionHintEl = $("#session-hint");
  const statusDotEl = $("#status-dot");
  const modeLabelEl = $("#mode-label");

  let sessionId = newSessionId();
  const sessions = new Map(); // id -> { id, title }

  const SOURCE_META = {
    sql: { label: "FAQ 命中", cls: "sql" },
    rag: { label: "知识库检索", cls: "rag" },
    mock: { label: "演示模式", cls: "mock" },
    err: { label: "出错了", cls: "err" },
  };

  function newSessionId() {
    return (crypto.randomUUID && crypto.randomUUID()) || "s-" + Math.random().toString(36).slice(2);
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setSession(id) {
    sessionId = id;
    sessionHintEl.textContent = "会话 ID：" + id;
  }

  function clearMessages() {
    messagesEl.innerHTML = "";
  }

  function addUserMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg user";
    wrap.innerHTML =
      '<div class="avatar">🧑</div><div class="bubble"></div>';
    wrap.querySelector(".bubble").textContent = text;
    messagesEl.appendChild(wrap);
    scrollToBottom();
  }

  function addAssistantPlaceholder() {
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    wrap.innerHTML =
      '<div class="avatar">🤖</div>' +
      '<div class="bubble"><div class="meta"></div>' +
      '<div class="typing"><span></span><span></span><span></span></div></div>';
    messagesEl.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function setBadge(wrap, source) {
    const meta = SOURCE_META[source] || SOURCE_META.err;
    const metaEl = wrap.querySelector(".meta");
    metaEl.innerHTML = `<span class="badge ${meta.cls}">${meta.label}</span>`;
  }

  function finishAssistant(wrap, text, source) {
    const bubble = wrap.querySelector(".bubble");
    setBadge(wrap, source);
    let content = bubble.querySelector(".content");
    if (!content) {
      content = document.createElement("div");
      content.className = "content";
      bubble.appendChild(content);
    }
    bubble.querySelector(".typing")?.remove();
    content.textContent = text;
    scrollToBottom();
  }

  function appendToken(wrap, token) {
    const bubble = wrap.querySelector(".bubble");
    bubble.querySelector(".typing")?.remove();
    let content = bubble.querySelector(".content");
    if (!content) {
      content = document.createElement("div");
      content.className = "content";
      bubble.appendChild(content);
    }
    content.textContent += token;
    scrollToBottom();
  }

  function showError(wrap, message) {
    finishAssistant(wrap, message, "err");
  }

  function rememberSession(id, title) {
    if (!sessions.has(id)) {
      sessions.set(id, { id, title: title || "新对话" });
      renderHistory();
    } else if (title) {
      sessions.get(id).title = title;
      renderHistory();
    }
  }

  function renderHistory() {
    historyListEl.innerHTML = "";
    if (sessions.size === 0) {
      historyListEl.innerHTML = '<li class="empty">暂无历史会话</li>';
      return;
    }
    for (const s of sessions.values()) {
      const li = document.createElement("li");
      li.textContent = s.title;
      li.title = s.id;
      li.addEventListener("click", () => loadSession(s.id));
      historyListEl.appendChild(li);
    }
  }

  async function loadSession(id) {
    setSession(id);
    clearMessages();
    try {
      const res = await fetch(`/api/qa/sessions/${encodeURIComponent(id)}/history`);
      if (!res.ok) return;
      const data = await res.json();
      if (!data.history || data.history.length === 0) return;
      for (const turn of data.history) {
        addUserMessage(turn.question);
        const wrap = addAssistantPlaceholder();
        finishAssistant(wrap, turn.answer, "sql");
      }
    } catch (_) {
      /* ignore */
    }
  }

  async function loadFaq() {
    try {
      const res = await fetch("/api/faq?limit=30");
      if (!res.ok) return;
      const data = await res.json();
      faqListEl.innerHTML = "";
      const items = data.items || [];
      if (items.length === 0) {
        faqListEl.innerHTML = '<li class="empty">暂无常见问题</li>';
        return;
      }
      for (const f of items) {
        const li = document.createElement("li");
        li.innerHTML = `${escapeHtml(f.question)}<span class="sub">${
          escapeHtml(f.subject || "通用")
        }</span>`;
        li.title = "点击填入输入框";
        li.addEventListener("click", () => {
          inputEl.value = f.question;
          inputEl.focus();
        });
        faqListEl.appendChild(li);
      }
    } catch (_) {
      /* ignore */
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      const ok = data.ready && data.status === "ok";
      statusDotEl.className = "status-dot " + (ok ? "ok" : "bad");
      modeLabelEl.textContent = data.mock
        ? "演示模式（内存存储，重启清空）"
        : "已连接真实后端";
      modeLabelEl.title = data.mock
        ? "未检测到 MySQL/Redis/Milvus/LLM 等后端，已使用内置演示数据。"
        : "已连接 EducationQASystem 真实后端。";
    } catch (_) {
      statusDotEl.className = "status-dot bad";
      modeLabelEl.textContent = "无法连接服务";
    }
  }

  async function sendMessage(text) {
    if (!text.trim()) return;
    addUserMessage(text);
    rememberSession(sessionId, text.slice(0, 22));
    const wrap = addAssistantPlaceholder();
    setSession(sessionId);

    const body = JSON.stringify({ query: text, session_id: sessionId });

    try {
      const res = await fetch("/api/qa/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!res.ok || !res.body) throw new Error("stream failed");
      await consumeStream(res.body, wrap);
    } catch (_) {
      // Fallback to non-streaming endpoint.
      try {
        const res2 = await fetch("/api/qa/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        });
        const data = await res2.json();
        setSession(data.session_id);
        rememberSession(data.session_id, text.slice(0, 22));
        finishAssistant(wrap, data.answer, data.source);
      } catch (err) {
        showError(wrap, "请求失败，请确认服务已启动。");
      }
    }
  }

  async function consumeStream(body, wrap) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentSource = "rag";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const dataLine = raw.split("\n").find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        let payload;
        try {
          payload = JSON.parse(dataLine.slice(5).trim());
        } catch (_) {
          continue;
        }
        handleEvent(payload, wrap, (s) => (currentSource = s));
      }
    }
  }

  function handleEvent(payload, wrap, setSource) {
    switch (payload.type) {
      case "meta":
        setSession(payload.session_id);
        setSource(payload.source);
        setBadge(wrap, payload.source);
        rememberSession(payload.session_id);
        break;
      case "token":
        appendToken(wrap, payload.content);
        break;
      case "done":
        break;
      case "error":
        showError(wrap, payload.message || "生成失败");
        break;
    }
  }

  // ---------- events ----------
  function submit() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    autoGrow();
    sendEl.disabled = true;
    sendMessage(text).finally(() => (sendEl.disabled = false));
  }

  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  }

  sendEl.addEventListener("click", submit);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });
  inputEl.addEventListener("input", autoGrow);

  $("#new-chat").addEventListener("click", () => {
    setSession(newSessionId());
    clearMessages();
    messagesEl.innerHTML =
      '<div class="empty-state"><div class="empty-icon">💡</div>' +
      "<p>你好，我是 EduRAG 答疑助手。</p>" +
      '<p class="muted">提问会从 FAQ 知识库与 RAG 检索中为你作答。</p></div>';
  });

  $("#clear-chat").addEventListener("click", async () => {
    try {
      await fetch(`/api/qa/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
    } catch (_) {}
    clearMessages();
  });

  // ---------- init ----------
  setSession(sessionId);
  renderHistory();
  checkHealth();
  loadFaq();
  inputEl.focus();
})();
