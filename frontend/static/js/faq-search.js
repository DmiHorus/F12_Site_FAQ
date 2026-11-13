(function () {
  var form = document.getElementById("faq-search-form");
  var input = document.getElementById("faq-q");
  var box = document.getElementById("faq-results");
  if (!form || !input || !box) return;

  function clearBox() {
    while (box.firstChild) box.removeChild(box.firstChild);
  }

  function showMessage(text) {
    clearBox();
    var p = document.createElement("p");
    p.textContent = text;
    box.appendChild(p);
  }

  function render(data) {
    clearBox();
    var results = data.results || [];
    if (!results.length) {
      showMessage("Ничего не найдено.");
      return;
    }
    results.forEach(function (item) {
      var article = document.createElement("article");
      article.className = "search-hit";
      var title = document.createElement("h3");
      title.textContent = item.question || "";
      var meta = document.createElement("p");
      meta.className = "score";
      meta.textContent = (item.category || "") + " · сходство " + item.score + " · " + (item.id || "");
      var answer = document.createElement("p");
      answer.textContent = item.answer || "";
      article.appendChild(title);
      article.appendChild(meta);
      article.appendChild(answer);
      box.appendChild(article);
    });
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    var query = input.value.trim();
    if (!query) {
      showMessage("Введите запрос.");
      return;
    }
    showMessage("Ищем…");
    try {
      var response = await fetch("/api/faq/search?q=" + encodeURIComponent(query) + "&top_k=5");
      if (!response.ok) throw new Error("request failed");
      render(await response.json());
    } catch (err) {
      showMessage("Не удалось выполнить поиск. Проверьте, что сервер запущен.");
    }
  });
})();
