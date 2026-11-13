(function () {
  var STORAGE_KEY = "mnwhl_faq_session_id";
  var GREETING = "Привет! Я FAQ-ассистент Минвайл. Спросите о компании, продуктах и подходе";
  var ERROR_TEXT = "Не удалось получить ответ. Проверьте, что сервер запущен, и попробуйте ещё раз.";

  var launcher = document.getElementById("chat-launcher");
  var widget = document.getElementById("chat-widget");
  var closeBtn = document.getElementById("chat-close");
  var messagesEl = document.getElementById("chat-messages");
  var form = document.getElementById("chat-form");
  var inputEl = document.getElementById("chat-input");
  var sendBtn = document.getElementById("chat-send");
  var isSending = false;
  var greeted = false;

  function readSessionId() {
    try {
      return sessionStorage.getItem(STORAGE_KEY);
    } catch (err) {
      return null;
    }
  }

  function storeSessionId(id) {
    try {
      if (id) sessionStorage.setItem(STORAGE_KEY, id);
    } catch (err) {
      /* sessionStorage может быть недоступен */
    }
  }

  function appendMessage(text, from) {
    var div = document.createElement("div");
    div.className = "chat-message " + from;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function appendTyping() {
    var div = document.createElement("div");
    div.className = "chat-message bot";
    div.id = "typing-indicator";
    var row = document.createElement("span");
    row.className = "typing";
    for (var i = 0; i < 3; i += 1) {
      row.appendChild(document.createElement("i"));
    }
    div.appendChild(row);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTyping() {
    var el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  function appendSuggestions(context) {
    if (!context || !context.length) return;
    var wrap = document.createElement("div");
    wrap.className = "chat-suggestions";
    var label = document.createElement("div");
    label.className = "chat-suggestions-label";
    label.textContent = "Похожие вопросы";
    wrap.appendChild(label);
    context.forEach(function (item) {
      if (!item || !item.question) return;
      var button = document.createElement("button");
      button.type = "button";
      button.className = "chat-suggestion";
      button.textContent = item.question;
      button.addEventListener("click", function () {
        inputEl.value = item.question;
        sendMessage();
      });
      wrap.appendChild(button);
    });
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function openChat() {
    widget.classList.add("is-open");
    widget.setAttribute("aria-hidden", "false");
    launcher.classList.add("is-hidden");
    launcher.setAttribute("aria-expanded", "true");
    if (!greeted) {
      appendMessage(GREETING, "bot");
      greeted = true;
    }
    window.setTimeout(function () { inputEl.focus(); }, 30);
  }

  function closeChat() {
    widget.classList.remove("is-open");
    widget.setAttribute("aria-hidden", "true");
    launcher.classList.remove("is-hidden");
    launcher.setAttribute("aria-expanded", "false");
    launcher.focus();
  }

  async function sendMessage() {
    if (isSending) return;
    var text = inputEl.value.trim();
    if (!text) return;
    appendMessage(text, "user");
    inputEl.value = "";
    isSending = true;
    sendBtn.disabled = true;
    appendTyping();
    var payload = { message: text, top_k: 3 };
    var sessionId = readSessionId();
    if (sessionId) payload.session_id = sessionId;
    try {
      var response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error("request failed");
      var data = await response.json();
      removeTyping();
      storeSessionId(data.session_id);
      appendMessage(data.answer || ERROR_TEXT, "bot");
      if (data.found) appendSuggestions(data.context);
    } catch (err) {
      removeTyping();
      appendMessage(ERROR_TEXT, "bot");
    } finally {
      isSending = false;
      sendBtn.disabled = false;
    }
  }

  launcher.addEventListener("click", openChat);
  closeBtn.addEventListener("click", closeChat);
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    sendMessage();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && widget.classList.contains("is-open")) closeChat();
  });
})();
