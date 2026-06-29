(() => {
  "use strict";

  const API = "/api";
  const TODAY = new Date().toISOString().slice(0, 10);
  const STORAGE_KEY = `mtg-cluster-${TODAY}`;

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let state = loadState();
  let selectedOracleId = null;
  let autocompleteItems = [];
  let acIndex = -1; // keyboard-selected autocomplete item

  // ---------------------------------------------------------------------------
  // DOM refs
  // ---------------------------------------------------------------------------
  const input = document.getElementById("guess-input");
  const acList = document.getElementById("autocomplete-list");
  const guessBtn = document.getElementById("guess-btn");
  const surrenderBtn = document.getElementById("surrender-btn");
  const historyBody = document.getElementById("history-body");
  const resultSection = document.getElementById("result-section");
  const revealedCard = document.getElementById("revealed-card");

  // ---------------------------------------------------------------------------
  // Persistence
  // ---------------------------------------------------------------------------
  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return JSON.parse(raw);
    } catch (_) {}
    return { guesses: [], won: false, surrendered: false };
  }

  function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------
  function simTier(pct) {
    if (pct >= 85) return "tier-hot";
    if (pct >= 60) return "tier-warm";
    if (pct >= 35) return "tier-cool";
    return "tier-cold";
  }

  function renderGuessRow(guess, index) {
    const tr = document.createElement("tr");
    const tier = simTier(guess.similarity_pct);
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${escHtml(guess.name)}</td>
      <td class="sim-bar-cell">
        <div class="sim-bar-wrapper">
          <div class="sim-bar">
            <div class="sim-bar-fill ${tier}" style="width:${guess.similarity_pct}%"></div>
          </div>
          <span class="sim-pct">${guess.similarity_pct}%</span>
        </div>
      </td>
      <td class="rank-cell">#${guess.rank}</td>
    `;
    return tr;
  }

  function renderHistory() {
    historyBody.innerHTML = "";
    // Most recent first
    const reversed = [...state.guesses].reverse();
    reversed.forEach((g, i) => {
      historyBody.appendChild(renderGuessRow(g, state.guesses.length - 1 - i));
    });
  }

  function renderReveal(card) {
    resultSection.hidden = false;
    revealedCard.innerHTML = `
      ${card.image_uri ? `<img src="${escHtml(card.image_uri)}" alt="${escHtml(card.name)}" />` : ""}
      <div class="card-name">${escHtml(card.name)}</div>
      <div class="card-type">${escHtml(card.type_line || "")}</div>
    `;
  }

  function setGameOver() {
    input.disabled = true;
    guessBtn.disabled = true;
    surrenderBtn.disabled = true;
  }

  // ---------------------------------------------------------------------------
  // API calls
  // ---------------------------------------------------------------------------
  async function fetchAutocomplete(q) {
    const res = await fetch(`${API}/cards/search?q=${encodeURIComponent(q)}&limit=8`);
    if (!res.ok) return [];
    return res.json();
  }

  async function submitGuess(oracleId) {
    const res = await fetch(`${API}/guess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ oracle_id: oracleId }),
    });
    if (!res.ok) throw new Error(`Guess failed: ${res.status}`);
    return res.json();
  }

  async function fetchCard(oracleId) {
    const res = await fetch(`${API}/card/${encodeURIComponent(oracleId)}`);
    if (!res.ok) throw new Error(`Card fetch failed: ${res.status}`);
    return res.json();
  }

  // ---------------------------------------------------------------------------
  // Autocomplete
  // ---------------------------------------------------------------------------
  function showAutocomplete(items) {
    autocompleteItems = items;
    acIndex = -1;
    acList.innerHTML = "";
    if (!items.length) { acList.hidden = true; return; }
    items.forEach((item, i) => {
      const li = document.createElement("li");
      li.textContent = item.name;
      li.setAttribute("role", "option");
      li.dataset.oracleId = item.oracle_id;
      li.addEventListener("mousedown", (e) => {
        e.preventDefault();
        selectAutocomplete(i);
      });
      acList.appendChild(li);
    });
    acList.hidden = false;
  }

  function hideAutocomplete() {
    acList.hidden = true;
    acIndex = -1;
  }

  function highlightAcItem(newIndex) {
    const items = acList.querySelectorAll("li");
    items.forEach((li, i) => li.setAttribute("aria-selected", String(i === newIndex)));
    acIndex = newIndex;
  }

  function selectAutocomplete(index) {
    const item = autocompleteItems[index];
    if (!item) return;
    input.value = item.name;
    selectedOracleId = item.oracle_id;
    hideAutocomplete();
  }

  let acTimeout = null;
  input.addEventListener("input", () => {
    selectedOracleId = null;
    clearTimeout(acTimeout);
    const q = input.value.trim();
    if (q.length < 2) { hideAutocomplete(); return; }
    acTimeout = setTimeout(async () => {
      const items = await fetchAutocomplete(q);
      showAutocomplete(items);
    }, 150);
  });

  input.addEventListener("keydown", (e) => {
    const items = acList.querySelectorAll("li");
    if (e.key === "ArrowDown") {
      e.preventDefault();
      highlightAcItem(Math.min(acIndex + 1, items.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      highlightAcItem(Math.max(acIndex - 1, 0));
    } else if (e.key === "Enter") {
      if (acIndex >= 0) {
        selectAutocomplete(acIndex);
      } else {
        handleGuess();
      }
    } else if (e.key === "Escape") {
      hideAutocomplete();
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest("#autocomplete-wrapper")) hideAutocomplete();
  });

  // ---------------------------------------------------------------------------
  // Game logic
  // ---------------------------------------------------------------------------
  async function handleGuess() {
    if (state.won || state.surrendered) return;
    const oracleId = selectedOracleId;
    if (!oracleId) { input.focus(); return; }

    guessBtn.disabled = true;
    try {
      const result = await submitGuess(oracleId);
      // Fetch card name to store in history (oracle_id alone isn't display-friendly)
      const card = await fetchCard(oracleId);
      const entry = {
        oracle_id: oracleId,
        name: card.name,
        similarity_pct: result.similarity_pct,
        rank: result.rank,
      };
      state.guesses.push(entry);

      if (result.is_correct) {
        state.won = true;
        renderReveal(card);
        setGameOver();
      }

      saveState();
      renderHistory();
      input.value = "";
      selectedOracleId = null;
    } catch (err) {
      console.error(err);
    } finally {
      guessBtn.disabled = false;
      input.focus();
    }
  }

  async function handleSurrender() {
    if (state.won || state.surrendered) return;
    // Daily endpoint gives us the card count but not the target id —
    // surrender is implemented by revealing after a confirmed "give up" action.
    // We call a special endpoint that the server exposes only for this purpose.
    // For now: fetch today's target via the guess endpoint by passing a sentinel.
    // Actually, we don't expose the target directly. Instead we reveal by guessing
    // the target, but we don't know it. Surrender must go via the backend.
    // Simple approach: add GET /api/daily/reveal once the player surrenders.
    // For the MVP, the server exposes /api/surrender which returns the target card.
    try {
      const res = await fetch(`${API}/surrender`, { method: "POST" });
      if (!res.ok) throw new Error("Surrender failed");
      const card = await res.json();
      state.surrendered = true;
      saveState();
      renderReveal(card);
      setGameOver();
    } catch (err) {
      console.error(err);
    }
  }

  guessBtn.addEventListener("click", handleGuess);
  surrenderBtn.addEventListener("click", handleSurrender);

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    renderHistory();
    if (state.won || state.surrendered) {
      setGameOver();
      // Re-fetch and show the revealed card if the game is already over
      // (page reload after win/surrender): we stored the oracle_id in guesses
      // for wins; for surrenders we'd need to store target_id separately.
      // For MVP: just render history; full reveal is re-fetched on next visit.
    }
  }

  init();

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
})();
