(() => {
  "use strict";

  const API = "/api";
  const TODAY = new Date().toISOString().slice(0, 10);

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let practiceTargetId = null; // non-null when in practice mode
  function getStorageKey() {
    return practiceTargetId ? `mtg-cluster-practice-${practiceTargetId}` : `mtg-cluster-${TODAY}`;
  }
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
  const similarCards = document.getElementById("similar-cards");
  const similarBody = document.getElementById("similar-body");
  const bestGuessSection = document.getElementById("best-guess-section");
  const bestGuessName = document.getElementById("best-guess-name");
  const bestGuessBar = document.getElementById("best-guess-bar");
  const bestGuessPct = document.getElementById("best-guess-pct");
  const bestGuessRank = document.getElementById("best-guess-rank");
  const cardTooltip = document.getElementById("card-tooltip");
  const cardTooltipImg = document.getElementById("card-tooltip-img");
  const practiceBtn = document.getElementById("practice-btn");
  const dailyBtn = document.getElementById("daily-btn");
  const newPracticeBtn = document.getElementById("new-practice-btn");
  const resultHeading = document.getElementById("result-heading");
  const modeSubtitle = document.getElementById("mode-subtitle");

  // ---------------------------------------------------------------------------
  // Card image tooltip
  // ---------------------------------------------------------------------------
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-image-uri]");
    if (!el) return;
    cardTooltipImg.src = el.dataset.imageUri;
    cardTooltip.hidden = false;
    positionTooltip(e);
  });

  document.addEventListener("mousemove", (e) => {
    if (!cardTooltip.hidden) positionTooltip(e);
  });

  document.addEventListener("mouseout", (e) => {
    if (e.target.closest("[data-image-uri]") && !e.relatedTarget?.closest("[data-image-uri]")) {
      cardTooltip.hidden = true;
    }
  });

  function positionTooltip(e) {
    const TW = 200, OFFSET = 16;
    let x = e.clientX + OFFSET;
    if (x + TW > window.innerWidth - 8) x = e.clientX - TW - OFFSET;
    cardTooltip.style.left = `${x}px`;
    cardTooltip.style.top = `${Math.max(8, e.clientY - 60)}px`;
  }

  // ---------------------------------------------------------------------------
  // Persistence
  // ---------------------------------------------------------------------------
  function loadState() {
    try {
      const raw = localStorage.getItem(getStorageKey());
      if (raw) return JSON.parse(raw);
    } catch (_) {}
    return { guesses: [], won: false, surrendered: false };
  }

  function saveState() {
    localStorage.setItem(getStorageKey(), JSON.stringify(state));
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

  function renderManaCost(manaCost) {
    if (!manaCost) return "";
    return [...manaCost.matchAll(/\{([^}]+)\}/g)]
      .map(([, sym]) => `<img class="mana-sym" src="https://svgs.scryfall.io/card-symbols/${encodeURIComponent(sym)}.svg" alt="{${escHtml(sym)}}" title="{${escHtml(sym)}}">`)
      .join("");
  }

  function renderGuessRow(guess, index) {
    const tr = document.createElement("tr");
    const tier = simTier(guess.similarity_pct);
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>
        <div class="card-name-cell" ${guess.image_uri ? `data-image-uri="${escHtml(guess.image_uri)}"` : ""}>${escHtml(guess.name)}</div>
        <div class="card-type-cell">${escHtml(guess.type_line || "")}</div>
      </td>
      <td class="mana-cost-cell">${renderManaCost(guess.mana_cost)}</td>
      <td class="sim-bar-cell">
        <div class="sim-bar-wrapper">
          <div class="sim-bar">
            <div class="sim-bar-fill ${tier}" style="width:${guess.similarity_pct}%"></div>
          </div>
          <span class="sim-pct">${guess.similarity_pct}%</span>
        </div>
        ${guess.top_features?.length ? `<div class="feature-hints">${guess.top_features.map(f => `${escHtml(f.feature)} (${f.similarity_pct}%)`).join(" · ")}</div>` : ""}
      </td>
      <td class="rank-cell">#${guess.rank}</td>
    `;
    return tr;
  }

  function getBestGuess() {
    if (!state.guesses.length) return null;
    return state.guesses.reduce((best, g) =>
      g.similarity_pct > best.similarity_pct ? g : best
    );
  }

  function renderBestGuess() {
    const best = getBestGuess();
    if (!best) { bestGuessSection.hidden = true; return; }
    const tier = simTier(best.similarity_pct);
    bestGuessName.textContent = best.name;
    if (best.image_uri) bestGuessName.dataset.imageUri = best.image_uri;
    else delete bestGuessName.dataset.imageUri;
    bestGuessBar.className = `sim-bar-fill ${tier}`;
    bestGuessBar.style.width = `${best.similarity_pct}%`;
    bestGuessPct.textContent = `${best.similarity_pct}%`;
    bestGuessRank.textContent = `#${best.rank}`;
    bestGuessSection.hidden = false;
  }

  function renderHistory() {
    historyBody.innerHTML = "";
    // Most recent first
    const reversed = [...state.guesses].reverse();
    reversed.forEach((g, i) => {
      historyBody.appendChild(renderGuessRow(g, state.guesses.length - 1 - i));
    });
    renderBestGuess();
  }

  function renderReveal(card) {
    resultSection.hidden = false;
    resultHeading.textContent = practiceTargetId ? "The card was…" : "Today's card was…";
    revealedCard.innerHTML = `
      ${card.image_uri ? `<img src="${escHtml(card.image_uri)}" alt="${escHtml(card.name)}" />` : ""}
      <div class="card-name">${escHtml(card.name)}</div>
      <div class="card-type">${escHtml(card.type_line || "")}</div>
    `;
  }

  async function fetchAndRenderSimilar(targetId = null) {
    try {
      const url = targetId
        ? `${API}/similar?limit=10&target_id=${encodeURIComponent(targetId)}`
        : `${API}/similar?limit=10`;
      const res = await fetch(url);
      if (!res.ok) return;
      const cards = await res.json();
      similarBody.innerHTML = "";
      cards.forEach((c) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>
            <span class="card-name-cell" ${c.image_uri ? `data-image-uri="${escHtml(c.image_uri)}"` : ""}>${escHtml(c.name)}</span>
            <div class="card-type-cell">${escHtml(c.type_line || "")}</div>
          </td>
          <td class="mana-cost-cell">${renderManaCost(c.mana_cost)}</td>
          <td class="sim-bar-cell">
            <div class="sim-bar-wrapper">
              <div class="sim-bar">
                <div class="sim-bar-fill ${simTier(c.similarity_pct)}" style="width:${c.similarity_pct}%"></div>
              </div>
              <span class="sim-pct">${c.similarity_pct}%</span>
            </div>
          </td>
        `;
        similarBody.appendChild(tr);
      });
      similarCards.hidden = false;
    } catch (err) {
      console.error("Failed to fetch similar cards:", err);
    }
  }

  function setInputEnabled(enabled) {
    input.disabled = !enabled;
    guessBtn.disabled = !enabled;
    surrenderBtn.disabled = !enabled;
  }

  function setGameOver() {
    setInputEnabled(false);
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
    const body = { oracle_id: oracleId };
    if (practiceTargetId) body.target_id = practiceTargetId;
    const res = await fetch(`${API}/guess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
        type_line: card.type_line || "",
        mana_cost: card.mana_cost || "",
        image_uri: card.image_uri || "",
        similarity_pct: result.similarity_pct,
        rank: result.rank,
        top_features: result.top_features || [],
      };
      state.guesses.push(entry);

      if (result.is_correct) {
        state.won = true;
        bestGuessSection.hidden = true;
        renderReveal(card);
        fetchAndRenderSimilar(practiceTargetId);
        setGameOver();
        if (practiceTargetId) newPracticeBtn.hidden = false;
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
    try {
      const body = practiceTargetId ? { target_id: practiceTargetId } : {};
      const res = await fetch(`${API}/surrender`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error("Surrender failed");
      const card = await res.json();
      state.surrendered = true;
      saveState();
      renderReveal(card);
      fetchAndRenderSimilar(practiceTargetId);
      setGameOver();
      if (practiceTargetId) newPracticeBtn.hidden = false;
    } catch (err) {
      console.error(err);
    }
  }

  // ---------------------------------------------------------------------------
  // Practice mode
  // ---------------------------------------------------------------------------
  async function startPractice() {
    try {
      const res = await fetch(`${API}/practice/new`);
      if (!res.ok) throw new Error("Failed to start practice");
      const { oracle_id } = await res.json();
      practiceTargetId = oracle_id;
      state = { guesses: [], won: false, surrendered: false };
      resultSection.hidden = true;
      similarCards.hidden = true;
      newPracticeBtn.hidden = true;
      bestGuessSection.hidden = true;
      setInputEnabled(true);
      renderHistory();
      practiceBtn.hidden = true;
      dailyBtn.hidden = false;
      modeSubtitle.textContent = "Practice mode — guessing a random card.";
    } catch (err) {
      console.error(err);
    }
  }

  function returnToDaily() {
    practiceTargetId = null;
    state = loadState();
    resultSection.hidden = true;
    similarCards.hidden = true;
    newPracticeBtn.hidden = true;
    practiceBtn.hidden = false;
    dailyBtn.hidden = true;
    modeSubtitle.textContent = "Guess today's secret Magic card by similarity.";
    renderHistory();
    if (state.won || state.surrendered) {
      setGameOver();
    } else {
      setInputEnabled(true);
    }
  }

  guessBtn.addEventListener("click", handleGuess);
  surrenderBtn.addEventListener("click", handleSurrender);
  practiceBtn.addEventListener("click", startPractice);
  dailyBtn.addEventListener("click", returnToDaily);
  newPracticeBtn.addEventListener("click", startPractice);

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
