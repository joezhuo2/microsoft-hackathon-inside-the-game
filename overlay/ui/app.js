const tauri = window.__TAURI__;
const API = tauri ? "http://127.0.0.1:8765" : "";
const $ = (id) => document.getElementById(id);
const offlineMessage = tauri ? "Replay server offline. Run: python -m football_analytics.replay_server" : "Replay server unreachable.";
const controls = ["game", "speed", "language", "club", "player", "enrich"];
const MAX_FEED = 5;
const MAX_HISTORY = 10;

const ICONS = {
  ball: '<circle cx="12" cy="12" r="9"/><path d="M12 7.5l3.6 2.6-1.4 4.2H9.8l-1.4-4.2zM12 7.5V3.2M15.6 10.1l4-1.3M14.2 14.3l2.5 3.5M9.8 14.3l-2.5 3.5M8.4 10.1l-4-1.3"/>',
  pie: '<circle cx="12" cy="12" r="9"/><path d="M12 3v9h9"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  pass: '<path d="M3 8h14M13 4l4 4-4 4M21 16H7M11 12l-4 4 4 4"/>',
  check: '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16.5 9.5"/>',
  shield: '<path d="M12 3l8 3v6c0 4.6-3.4 8-8 9-4.6-1-8-4.4-8-9V6z"/>',
  bolt: '<path d="M13 2L4.5 13.5H11L10 22l8.5-11.5H12z"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
  wave: '<path d="M2 12h4l3-7 4 14 3-7h6"/>',
  sliders: '<path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>',
  spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
};
const MOMENT_ICON = { goal: "ball", shot: "target", key_pass: "pass", sustained_build_up: "pass", top_speed: "bolt", press_escape: "bolt", high_turnover: "shield", momentum_shift: "wave", rhythm_change: "wave" };
const STATS = [
  { key: "possession", label: "Possession", icon: "pie", unit: "%", casual: true },
  { key: "shots", label: "Shots", icon: "target", casual: true },
  { key: "passes", label: "Passes", icon: "pass", casual: true },
  { key: "pass_accuracy", label: "Pass accuracy", icon: "check", unit: "%" },
  { key: "ball_wins", label: "Ball wins", icon: "shield" },
  { key: "sprints", label: "Sprints", icon: "bolt" },
];

const STALE_TIMESTEPS = 300;
const STUDIO = new URLSearchParams(location.search).get("view") === "studio";
const STUDIO_MAX = 60;
const MAX_CHIPS = 2;
const PREF_KEY = "match-prefs";
const DEPTH_PRIORITIES = { headlines: ["high"], balanced: ["high", "medium"], deep: ["high", "medium", "low"] };
const ACTION_QUESTIONS = { explain_more: "Explain more", show_tactics: "Show tactics", simpler: "Make it simpler", numbers: "Give me the numbers", why: "Why did this matter?", recap: "Give me the match recap." };
const REPLY_ACTIONS = Object.fromEntries(Object.entries(ACTION_QUESTIONS).map(([action, text]) => [text, action]));
const FEEDBACK = [["too_basic", "Too basic"], ["too_technical", "Too technical"], ["more_tactical", "More tactical"], ["more_emotional", "More emotional"], ["follow_player", "Follow this player"]];
const I18N_KEYS = ["Kick off", "Half time", "Second half", "Full time", "Match recap", "Goal threat", "Momentum shift", "Player spotlight", "Tactical reset", "Money move", "Why did this matter?", "Show tactics", "Give me the numbers", "Make it simpler", "Explain more", "Focus on", "Too basic", "Too technical", "More tactical", "More emotional", "Follow this player", "Observed", "Interpretation", "Projection"];
const I18N = Object.fromEntries(Object.entries({
  es: ["Saque inicial", "Descanso", "Segunda parte", "Final", "Resumen del partido", "Ocasión de gol", "Cambio de impulso", "Jugador destacado", "Reajuste táctico", "Jugada de valor", "¿Por qué importó?", "Ver táctica", "Dame los números", "Más sencillo", "Explica más", "Foco en", "Demasiado básico", "Demasiado técnico", "Más táctico", "Más emoción", "Seguir a este jugador", "Observado", "Interpretación", "Proyección"],
  fr: ["Coup d'envoi", "Mi-temps", "Seconde période", "Fin du match", "Résumé du match", "Occasion de but", "Bascule d'élan", "Joueur à la une", "Ajustement tactique", "Coup de valeur", "Pourquoi c'est important ?", "Voir la tactique", "Donne-moi les chiffres", "Plus simple", "Explique davantage", "Focus sur", "Trop basique", "Trop technique", "Plus tactique", "Plus d'émotion", "Suivre ce joueur", "Observé", "Interprétation", "Projection"],
  de: ["Anstoß", "Halbzeit", "Zweite Halbzeit", "Abpfiff", "Spielbericht", "Torgefahr", "Momentum-Wechsel", "Spieler im Fokus", "Taktischer Reset", "Wert-Moment", "Warum war das wichtig?", "Taktik zeigen", "Zeig mir die Zahlen", "Einfacher", "Mehr erklären", "Fokus auf", "Zu einfach", "Zu technisch", "Taktischer", "Emotionaler", "Diesem Spieler folgen", "Beobachtet", "Einordnung", "Prognose"],
  pt: ["Pontapé inicial", "Intervalo", "Segundo tempo", "Fim de jogo", "Resumo da partida", "Chance de gol", "Virada de momento", "Jogador em destaque", "Ajuste tático", "Lance de valor", "Por que isso importou?", "Mostrar tática", "Quero os números", "Mais simples", "Explique mais", "Foco em", "Básico demais", "Técnico demais", "Mais tático", "Mais emoção", "Seguir este jogador", "Observado", "Interpretação", "Projeção"],
  it: ["Calcio d'inizio", "Intervallo", "Secondo tempo", "Fine partita", "Riepilogo", "Occasione da gol", "Cambio d'inerzia", "Giocatore in evidenza", "Ritocco tattico", "Mossa di valore", "Perché è stato importante?", "Mostra la tattica", "Dammi i numeri", "Più semplice", "Spiega di più", "Focus su", "Troppo semplice", "Troppo tecnico", "Più tattico", "Più emozione", "Segui questo giocatore", "Osservato", "Interpretazione", "Proiezione"],
}).map(([lang, words]) => [lang, Object.fromEntries(I18N_KEYS.map((key, i) => [key, words[i]]))]));
const tr = (text) => {
  const words = I18N[$("language").value] || {};
  return words[text] || text.replace(/^(Half time|Full time|Focus on) /, (m, key) => `${words[key] || key} `);
};
let savedPrefs = {};
try { savedPrefs = JSON.parse(localStorage.getItem(PREF_KEY)) || {}; } catch { savedPrefs = {}; }
const state = { studio: { seq: 0, items: [], onAir: true, minConfidence: 0, autoAir: { high: true, medium: true, low: false } }, prefs: { depth: "balanced", tone: "neutral", tactical: false, ...savedPrefs }, askToken: 0, last: null, roster: null, source: null, frames: [], lastT: 0, paused: false, alertTimer: null, alertToken: 0, alertQueue: [], alertBusy: false, recent: [], entries: new Map(), moments: new Map(), stats: {}, score: [0, 0] };

document.body.classList.add(tauri ? "tauri" : "web");

function svg(name) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || THEME_ICONS[name] || ""}</svg>`;
}

function paintIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((node) => { node.innerHTML = svg(node.dataset.icon); });
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
}

function reveal(node, text) {
  node.replaceChildren(...String(text || "").split(/(\s+)/).filter(Boolean).map((part, i) => {
    const span = el("span", { class: "w" }, part);
    span.style.setProperty("--i", String(Math.floor(i / 2)));
    return span;
  }));
}

function label(kind) {
  return kind.replaceAll("_", " ");
}

function profile() {
  return { interests: Object.entries(interests()).map(([k, v]) => `${k}:${v}`).join(","), language: $("language").value, club: $("club").value, player: $("player").value, depth: state.prefs.depth, tone: state.prefs.tone, tactical: state.prefs.tactical ? "1" : "0" };
}

const INTERESTS = ["speed", "passing", "pressure", "possession", "money"];
const BANDS = [[75, "Obsessed"], [50, "Into it"], [20, "Light"], [0, "Skip"]];
const PERSONA_DEPTH = { analyst: "deep", newcomer: "headlines" };

function interests() {
  return Object.fromEntries(INTERESTS.map((k) => [k, state.prefs.interests?.[k] ?? 50]));
}

function persona() {
  const i = interests(), mean = INTERESTS.reduce((sum, k) => sum + i[k], 0) / INTERESTS.length;
  if (i.money >= 70 && INTERESTS.every((k) => k === "money" || i[k] < i.money)) return "money";
  return mean >= 70 ? "analyst" : mean <= 25 ? "newcomer" : "casual";
}

function analyst() {
  return !["casual", "newcomer"].includes(persona());
}

function buildDials(box) {
  box.append(...INTERESTS.map((k) => el("label", { class: "dial" }, el("span", {}, k), el("input", { type: "range", min: "0", max: "100", step: "5", "data-interest": k }), el("output", {}))));
}

function paintDials() {
  const i = interests();
  document.querySelectorAll("[data-interest]").forEach((input) => {
    const v = i[input.dataset.interest], band = BANDS.find(([min]) => v >= min)[1];
    input.value = String(v);
    input.setAttribute("aria-valuetext", `${v}, ${band}`);
    input.nextElementSibling.textContent = `${v} · ${band}`;
  });
}

function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function tween(node, to, unit = "") {
  const from = Number(node.dataset.value || 0);
  node.dataset.value = to;
  if (from === to) return;
  node.classList.remove("bump");
  void node.offsetWidth;
  node.classList.add("bump");
  const start = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - start) / 600);
    node.textContent = `${Math.round(from + (to - from) * (1 - (1 - k) ** 3))}${unit}`;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function buildTiles() {
  $("tiles").replaceChildren(...STATS.map((s) => el("div", { class: "stat-wrap", "data-key": s.key },
    el("div", {}, el("div", { class: "stat frame" },
      el("span", { class: "ibox", "data-icon": s.icon }),
      el("div", {}, el("div", { class: "label" }, s.label),
        el("div", { class: "values" }, el("span", { "data-team": "0" }, `0${s.unit || ""}`), el("i", {}, "|"), el("span", { "data-team": "1" }, `0${s.unit || ""}`))))))));
  paintIcons($("tiles"));
  applyPersona();
}

function applyPersona() {
  const club = $("club").value;
  for (const wrap of $("tiles").children) {
    const stat = STATS.find((s) => s.key === wrap.dataset.key);
    wrap.classList.toggle("in", analyst() || stat.casual);
    wrap.querySelectorAll("[data-team]").forEach((span) => span.classList.toggle("dim", club !== "" && span.dataset.team !== club));
  }
}

function updateStats(stats) {
  for (const s of STATS) {
    const values = stats?.[s.key];
    if (!values) continue;
    const wrap = $("tiles").querySelector(`[data-key="${s.key}"]`);
    wrap.querySelectorAll("[data-team]").forEach((span) => tween(span, values[Number(span.dataset.team)], s.unit || ""));
  }
}

function clubName(teamId) {
  return teamId == null ? "Match" : state.roster?.clubs?.[String(teamId)]?.name || `Team ${teamId}`;
}

function visiblePriorities() {
  return DEPTH_PRIORITIES[state.prefs.depth] || DEPTH_PRIORITIES.balanced;
}

function savePrefs() {
  try { localStorage.setItem(PREF_KEY, JSON.stringify(state.prefs)); } catch { /* storage unavailable */ }
}

function preferences() {
  const club = $("club").value, player = $("player").value;
  return { persona: persona(), ...state.prefs, language: $("language").value, favorite_club_id: club === "" ? null : Number(club), favorite_player_id: player === "" ? null : Number(player) };
}

function savePicks() {
  const { language, favorite_club_id, favorite_player_id } = preferences();
  Object.assign(state.prefs, { language, favorite_club_id, favorite_player_id });
  savePrefs();
}

function restorePicks() {
  for (const [id, saved] of [["language", state.prefs.language], ["club", state.prefs.favorite_club_id], ["player", state.prefs.favorite_player_id]]) {
    if (saved != null && [...$(id).options].some((o) => o.value === String(saved))) $(id).value = String(saved);
  }
}

const playerId = () => $("player").value === "" ? null : Number($("player").value);

function post(path, body) {
  return fetch(`${API}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: abort.signal }).then(async (response) => {
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.status);
    return data;
  });
}

function compactMode() {
  const now = performance.now();
  state.recent = state.recent.filter((at) => now - at < 4000);
  return window.matchMedia("(max-width: 900px)").matches || state.recent.length >= 3;
}

function routeCue(data) {
  const cue = data.cue;
  if (!cue) return showAlert(data);
  if (STUDIO) return studioCue(data);
  if (!visiblePriorities().includes(cue.priority)) return;
  airCue(data);
}

function studioCue(data) {
  const cue = data.cue;
  const s = state.studio;
  if (s.items.some((item) => item.cue.event_id === cue.event_id && item.cue.timestep === cue.timestep)) return;
  const item = { id: ++s.seq, cue, data, status: "queued" };
  s.items.unshift(item);
  s.items.length = Math.min(s.items.length, STUDIO_MAX);
  if (s.onAir && s.autoAir[cue.priority] && cue.confidence >= s.minConfidence / 100) airItem(item);
  else renderQueue();
}

function airItem(item) {
  item.status = "aired";
  airCue(item.data);
  renderQueue();
}

function airCue(data) {
  const cue = data.cue;
  sting(cue);
  logCue(cue);
  const compact = compactMode();
  state.recent.push(performance.now());
  if (cue.priority === "high" && !compact) {
    state.alertQueue.push(data);
    if (!state.alertBusy) nextAlert();
  } else {
    showChip(cue);
  }
}

function nextAlert() {
  while (state.alertQueue.length && state.lastT - state.alertQueue[0].cue.timestep > STALE_TIMESTEPS) state.alertQueue.shift();
  const data = state.alertQueue.shift();
  state.alertBusy = Boolean(data);
  const live = epoch;
  if (data) showAlert(data, () => setTimeout(() => live === epoch && nextAlert(), 380));
}

function logCue(cue) {
  const look = theme(cue.theme);
  const item = el("li", { dir: "auto" }, el("b", {}, `${cue.match_clock} · ${look.label}`), " ", cue.primary_insight);
  item.style.setProperty("--accent", look.accent);
  $("historyList").prepend(item);
  [...$("historyList").children].slice(MAX_HISTORY).forEach((n) => n.remove());
}

function metricNodes(metrics) {
  return metrics.map((s) => el("div", {}, el("b", {}, `${s.value}${s.unit ? " " + s.unit : ""}`), el("span", {}, s.label)));
}

function showAlert(data, onDone) {
  const o = data.overlay;
  const cue = data.cue;
  const look = cue ? theme(cue.theme) : null;
  const token = ++state.alertToken;
  const box = $("alert");
  const wasShown = box.classList.contains("show");
  box.classList.remove("show");
  clearTimeout(state.alertTimer);
  setTimeout(() => {
    if (token !== state.alertToken) return;
    const kind = cue?.event_type || data.moment_type;
    box.classList.toggle("goal", kind === "goal");
    box.style.setProperty("--accent", look?.accent || "#fff");
    box.dataset.motion = look?.motion || "";
    box.querySelector(".badge span").dataset.icon = kind === "goal" ? "ball" : look?.icon || MOMENT_ICON[kind] || "spark";
    paintIcons(box.querySelector(".badge"));
    $("alertKicker").textContent = cue ? `${cue.match_clock} · ${clubName(cue.team_id)}` : o.body;
    $("alertTag").textContent = look?.label || label(kind);
    reveal($("alertHeadline"), cue?.primary_insight || o.headline);
    reveal($("alertBody"), cue?.why_it_matters || "");
    $("alertStats").replaceChildren(...metricNodes(analyst() ? cue?.supporting_metrics || o?.stats || [] : []));
    const lowConfidence = Boolean(cue && cue.confidence < 0.5);
    box.classList.toggle("low-confidence", lowConfidence);
    $("alertTrail").replaceChildren(...(cue && (analyst() || lowConfidence) ? [
      el("span", { class: "tag ghost" }, `${cue.evidence} · ${Math.round(cue.confidence * 100)}%`),
      lowConfidence ? el("span", { class: "tag warn" }, "low confidence") : null,
      analyst() && cue.source_event_ids.length ? el("span", { class: "ids" }, `events ${cue.source_event_ids.slice(-3).join(", ")}`) : null,
    ].filter(Boolean) : []));
    void box.offsetWidth;
    box.classList.add("show");
    state.alertTimer = setTimeout(() => {
      box.classList.remove("show");
      onDone?.();
    }, cue?.duration_ms || o.display_seconds * 1000);
  }, wasShown ? 320 : 0);
}

function showChip(cue) {
  const look = theme(cue.theme);
  const chip = el("div", { class: "cue-chip frame", role: "status" },
    el("span", { class: "chip-icon", "data-icon": look.icon }),
    el("span", { class: "chip-text", dir: "auto" }, el("b", {}, look.label), " ", cue.primary_insight));
  chip.style.setProperty("--accent", look.accent);
  paintIcons(chip);
  $("lane").prepend(chip);
  requestAnimationFrame(() => requestAnimationFrame(() => chip.classList.add("in")));
  const live = [...$("lane").children].filter((n) => !n.classList.contains("leaving"));
  live.slice(MAX_CHIPS).forEach(dismissChip);
  setTimeout(() => dismissChip(chip), cue.duration_ms);
}

function dismissChip(chip) {
  if (chip.classList.contains("leaving")) return;
  chip.classList.add("leaving");
  chip.classList.remove("in");
  setTimeout(() => chip.remove(), 450);
}

function addEntry(momentId, build) {
  const inner = el("div", { class: "entry loading" });
  inner.style.setProperty("--accent", theme(state.moments.get(momentId)?.cue?.theme).accent);
  const item = el("div", { class: "feed-item" }, el("div", {}, inner));
  build(inner);
  $("feedList").prepend(item);
  requestAnimationFrame(() => requestAnimationFrame(() => item.classList.add("in")));
  state.entries.set(momentId, inner);
  const items = [...$("feedList").children].filter((n) => !n.classList.contains("leaving"));
  for (const old of items.slice(MAX_FEED)) {
    old.classList.add("leaving");
    old.classList.remove("in");
    old.addEventListener("transitionend", () => old.remove(), { once: true });
  }
}

function showPending(momentId) {
  const moment = state.moments.get(momentId);
  addEntry(momentId, (inner) => inner.append(
    el("div", { class: "row" }, el("span", { class: "tag" }, `${label(moment?.moment_type || "moment")} · ${momentId}`), el("span", { class: "tag ghost dots" }, "agents analysing")),
    el("div", { class: "swap" }, el("h3", { dir: "auto" }, moment?.overlay.headline || momentId), el("p", {}, "Explainer, then narrator and personalizer in parallel."))));
}

function handoffLine(handoffs) {
  if (!handoffs.length) return "";
  const total = handoffs.reduce((sum, h) => sum + (h.seconds || 0), 0);
  if (!analyst()) return `${new Set(handoffs.map((h) => h.agent)).size} agents · ${Math.round(total)}s`;
  return handoffs.map((h) => `${h.agent.replace("match-", "")}${h.attempt > 1 ? " (retry)" : ""} ${h.seconds}s${h.error ? " failed" : ""}`).join(" → ");
}

function showEnriched(data) {
  if (!state.entries.has(data.moment_id)) showPending(data.moment_id);
  const inner = state.entries.get(data.moment_id);
  const moment = state.moments.get(data.moment_id);
  const swap = inner.querySelector(".swap");
  swap.style.opacity = "0";
  setTimeout(() => {
    inner.classList.remove("loading");
    const verified = data.validation?.valid;
    inner.querySelector(".row").replaceChildren(...[
      el("span", { class: "tag" }, `${label(moment?.moment_type || "moment")} · ${data.moment_id}`),
      el("span", { class: "tag ghost" }, (data.language || "en").toUpperCase()),
      el("span", { class: verified ? "tag ok" : "tag warn" }, verified ? "verified" : "unverified"),
      data.cached ? el("span", { class: "tag ghost" }, "cached") : null,
      data.fallbacks?.length ? el("span", { class: "tag warn", title: data.fallbacks.join("; ") }, "recovered") : null].filter(Boolean));
    const h3 = el("h3", { dir: "auto" });
    const p = el("p", { dir: "auto" });
    swap.replaceChildren(h3, p, el("div", { class: "meta" }, handoffLine(data.handoffs || [])));
    reveal(h3, data.overlay?.headline || "");
    reveal(p, data.text || data.overlay?.body || "");
    swap.style.opacity = "";
  }, 300);
}

const STINGS = { goal_threat: "Goal threat", momentum_swing: "Momentum shift", player_spotlight: "Player spotlight", control_index: "Tactical reset", chaos_index: "Tactical reset", value_swing: "Money move" };
const STING_GAP_MS = 12000;
let splashTimer = null;
let lastSting = -STING_GAP_MS;

function splash(text, ms, { accent = null, small = false } = {}) {
  clearTimeout(splashTimer);
  $("splashText").textContent = tr(text);
  $("splash").classList.toggle("sting", small);
  $("splash").style.setProperty("--accent", accent || "#fff");
  $("splash").classList.add("show");
  splashTimer = setTimeout(() => $("splash").classList.remove("show"), ms);
  return new Promise((resolve) => setTimeout(resolve, ms + 400));
}

function sting(cue) {
  const text = STINGS[cue.theme];
  const now = performance.now();
  if (!text || state.splashBusy || now - lastSting < STING_GAP_MS) return;
  lastSting = now;
  splash(text, 1300, { accent: theme(cue.theme).accent, small: true });
}

async function sequence(steps) {
  const live = epoch;
  state.splashBusy = true;
  for (const [text, ms] of steps) {
    if (live !== epoch) return false;
    await splash(text, ms);
  }
  if (live !== epoch) return false;
  state.splashBusy = false;
  return true;
}

function renderQueue() {
  const s = state.studio;
  $("sCount").textContent = `${s.items.length}`;
  $("sQueue").replaceChildren(...s.items.map((item) => {
    const cue = item.cue;
    const look = theme(cue.theme);
    const dropped = item.status === "dropped";
    return el("div", { class: `q-row ${item.status}`, "data-id": item.id },
      el("span", { class: "q-clock" }, cue.match_clock),
      el("span", { class: `q-pri ${cue.priority}`, style: `color: ${look.accent}` }, cue.priority),
      el("span", { class: "q-insight", dir: "auto", title: cue.primary_insight }, `${look.label}: ${cue.primary_insight}`),
      el("span", { class: "q-actions" },
        el("button", { type: "button", "data-air": "", ...(dropped || item.status === "aired" ? { disabled: "" } : {}) }, item.status === "aired" ? "On air" : "Air"),
        " ",
        el("button", { type: "button", "data-drop": "", ...(item.status === "aired" ? { disabled: "" } : {}) }, dropped ? "Dropped" : "Drop")),
      el("span", { class: "q-meta" }, `conf ${Math.round(cue.confidence * 100)}% · ${cue.evidence} · ${cue.duration_ms / 1000}s · ts ${cue.timestep}${item.data.cue_validation?.valid === false ? " · failed check" : ""}`));
  }));
}

function clockMinutes(clock) {
  return Number.parseInt(clock, 10);
}

function applyFrame(f) {
  state.frames = [state.frames.at(-1) || f, { ...f, at: performance.now() }];
  state.lastT = f.t;
  f.score.forEach((goals, team) => {
    const node = $(`score${team}`);
    if (state.score[team] !== goals) {
      node.classList.remove("pulse");
      void node.offsetWidth;
      if (state.score[team] < goals) node.classList.add("pulse");
    }
    node.textContent = goals;
  });
  state.score = f.score;
  $("clock").textContent = f.clock;
  if (!state.halfDone && clockMinutes(f.clock) >= 45) {
    state.halfDone = true;
    sequence([[`Half time ${f.score[0]}–${f.score[1]}`, 3000], ["Second half", 2200]]);
  }
  if (f.rhythm) {
    $("rhythm").textContent = f.rhythm.rhythm;
    $("rhythm").className = `tag ${f.rhythm.rhythm}`;
    $("momentum").style.left = `${50 - 50 * Math.max(-1, Math.min(1, f.rhythm.momentum_team0))}%`;
  }
  updateStats(f.stats);
}

let epoch = 0, abort = new AbortController();

function connect(fromStart = false) {
  state.source?.close();
  epoch++;
  abort.abort();
  abort = new AbortController();
  ++state.alertToken;
  clearTimeout(state.alertTimer);
  state.alertBusy = false;
  $("alert").classList.remove("show");
  clearTimeout(splashTimer);
  $("splash").classList.remove("show");
  state.splashBusy = false;
  if (state.paused) {
    state.resumeFromStart ||= fromStart;
    return status("Paused · changes apply on resume");
  }
  fromStart ||= Boolean(state.resumeFromStart);
  state.resumeFromStart = false;
  if (fromStart) {
    state.lastT = 0;
    state.score = [0, 0];
    state.frames = [];
    $("historyList").replaceChildren();
  }
  const query = new URLSearchParams({ game: $("game").value, speed: $("speed").value, start: String(state.lastT), enrich: $("enrich").checked ? "1" : "0", ...profile() });
  state.entries.clear();
  state.alertQueue = [];
  $("feedList").replaceChildren();
  $("lane").replaceChildren();
  $("feedLang").textContent = $("language").value.toUpperCase();
  const source = new EventSource(`${API}/api/stream?${query}`);
  state.source = source;
  status("Connecting…");
  source.addEventListener("start", (e) => {
    const s = JSON.parse(e.data);
    status(`Game ${s.game_id} live · ${s.enrich ? "AI analysis on" : "instant overlays only"}`);
    if (fromStart) {
      state.halfDone = false;
      state.studio.items = [];
      renderQueue();
      splash("Kick off", 2600);
    }
  });
  source.addEventListener("frame", (e) => applyFrame(JSON.parse(e.data)));
  source.addEventListener("moment", (e) => {
    const data = JSON.parse(e.data);
    state.moments.set(data.overlay.moment_id, data);
    routeCue(data);
  });
  source.addEventListener("cue", (e) => routeCue(JSON.parse(e.data)));
  source.addEventListener("enriching", (e) => showPending(JSON.parse(e.data).moment_id));
  source.addEventListener("enriched", (e) => showEnriched(JSON.parse(e.data)));
  source.addEventListener("enrich_failed", (e) => {
    const data = JSON.parse(e.data);
    const inner = state.entries.get(data.moment_id);
    inner?.querySelector(".row").replaceChildren(el("span", { class: "tag" }, data.moment_id), el("span", { class: "tag warn" }, "agents unavailable"));
    inner?.classList.remove("loading");
  });
  source.addEventListener("end", (e) => {
    const f = JSON.parse(e.data);
    applyFrame(f);
    sequence([[`Full time ${f.score[0]}–${f.score[1]}`, 3600], ["Match recap", 2000]]).then((ok) => ok && askMatch("recap"));
    status("Full time. Finishing analysis…");
  });
  source.addEventListener("done", () => {
    source.close();
    status("Full time. All analysis delivered.");
  });
  source.onerror = () => {
    source.close();
    if (state.source !== source || state.paused) return;
    status(state.frames.length ? "Connection lost, reconnecting…" : offlineMessage, true);
    setTimeout(() => state.source === source && !state.paused && connect(), 2000);
  };
}

const radar = $("radar");
const ctx = radar.getContext("2d");

function sizeRadar() {
  const ratio = window.devicePixelRatio || 1;
  radar.width = Math.round(radar.clientWidth * ratio);
  radar.height = Math.round(radar.clientHeight * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawPitch(w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(255,255,255,0.025)";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 1;
  const sx = w / 120, sy = h / 80;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
  ctx.beginPath(); ctx.arc(w / 2, h / 2, 9.15 * sx, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeRect(0.5, (40 - 20.15) * sy, 16.5 * sx, 40.3 * sy);
  ctx.strokeRect(w - 0.5 - 16.5 * sx, (40 - 20.15) * sy, 16.5 * sx, 40.3 * sy);
}

function lerp(a, b, k) {
  const prev = new Map((a.players || []).map((p) => [p[0], p]));
  return (b.players || []).map(([id, x, y]) => {
    const q = prev.get(id);
    return q ? [id, q[1] + (x - q[1]) * k, q[2] + (y - q[2]) * k] : [id, x, y];
  });
}

function draw() {
  const w = radar.clientWidth, h = radar.clientHeight;
  drawPitch(w, h);
  const [a, b] = state.frames;
  if (b) {
    const k = Math.min(1, (performance.now() - b.at) / 100);
    const sx = w / 120, sy = h / 80;
    const favourite = Number($("player").value || NaN);
    for (const [id, x, y] of lerp(a, b, k)) {
      ctx.beginPath();
      ctx.arc(x * sx, y * sy, 4.5, 0, Math.PI * 2);
      if (id < 100) {
        ctx.fillStyle = "#fff";
        ctx.fill();
      } else {
        ctx.fillStyle = "rgba(255,255,255,0.12)";
        ctx.fill();
        ctx.strokeStyle = "#bdbdbd";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      if (id === favourite) {
        ctx.beginPath();
        ctx.arc(x * sx, y * sy, 9, 0, Math.PI * 2);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.shadowColor = "#fff";
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = "#fff";
        ctx.font = "600 10px 'Segoe UI', sans-serif";
        ctx.fillText((state.roster?.players[String(id)]?.short || String(id)).toUpperCase(), x * sx + 11, y * sy - 7);
      }
    }
    if (a.ball && b.ball) {
      const bx = a.ball[0] + (b.ball[0] - a.ball[0]) * k, by = a.ball[1] + (b.ball[1] - a.ball[1]) * k;
      ctx.beginPath();
      ctx.arc(bx * sx, by * sy, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.shadowColor = "#fff";
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }
  requestAnimationFrame(draw);
}

async function ask(question, action) {
  $("answerWrap").classList.add("open");
  $("answer").replaceChildren(el("span", { class: "tag ghost dots" }, "agents working"));
  const token = state.askToken;
  try {
    const response = await fetch(`${API}/api/ask`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ game: $("game").value, timestep: state.lastT, question, action, ...profile(), depth: state.prefs.depth, tone: state.prefs.tone, tactical: state.prefs.tactical }), signal: abort.signal });
    const data = await response.json();
    if (token !== state.askToken) return;
    if (!response.ok) throw new Error(data.error || response.status);
    const text = el("div", { dir: "auto" });
    $("answer").replaceChildren(...[
      el("span", { class: data.verified ? "tag ok" : "tag warn" }, data.verified ? "verified by gate" : "not verified by gate"),
      data.fallbacks?.length ? el("span", { class: "tag warn", title: data.fallbacks.join("; ") }, "partial") : null,
      text,
      data.evidence?.length ? el("span", { class: "tag ghost" }, `Evidence: ${data.evidence.join(", ")}`) : null].filter(Boolean));
    reveal(text, data.answer);
  } catch (error) {
    if (token === state.askToken) $("answer").textContent = error.name === "AbortError" ? "" : `Could not reach the agents (${error.message}).`;
  }
}

const DOCK = [["panelWrap", "dockBtn"], ["askWrap", "askBtn"], ["valueWrap", "valueBtn"]];

function setDock(wrap, open) {
  for (const [w, b] of DOCK) {
    const on = w === wrap && open;
    $(w).classList.toggle("open", on);
    $(b).setAttribute("aria-expanded", String(on));
  }
  if (tauri) setTimeout(reportHitRegions, 500);
}

const setPanel = (open) => setDock("panelWrap", open);
const setAsk = (open) => setDock("askWrap", open);
const setValue = (open) => {
  setDock("valueWrap", open);
  if (open) loadValue(valueTarget());
};
const isOpen = (wrap) => $(wrap).classList.contains("open");

const VERDICTS = [["bargain", "Bargain"], ["fair value", "Fair"], ["overpaid so far", "Overpaid"]];
const VALUE_KEY = "match-value-picks";
const VALUE_REFRESH_MS = 5000;
const value = { id: null, data: null, board: [], busy: false, at: 0, t: -1, boardOpen: false, openRows: new Set() };

function gbp(n) {
  const sign = n < 0 ? "-" : "";
  const a = Math.abs(n);
  if (a >= 1e6) return `${sign}£${(a / 1e6).toFixed(2)}m`;
  if (a >= 1e3) return `${sign}£${Math.round(a / 1e3)}k`;
  return `${sign}£${Math.round(a)}`;
}

function readPicks() {
  try { return JSON.parse(localStorage.getItem(VALUE_KEY)) || {}; } catch { return {}; }
}

function savePick(id, pick) {
  const picks = readPicks();
  picks[id] = pick;
  try { localStorage.setItem(VALUE_KEY, JSON.stringify(picks)); } catch { /* storage unavailable */ }
}

function valueTarget() {
  return value.id ?? ($("player").value || null);
}

function buildValueSelect() {
  const groups = Object.values(state.roster.clubs).map((club) => el("optgroup", { label: club.name }));
  const players = Object.entries(state.roster.players).sort((a, b) => Number(a[0]) - Number(b[0]));
  for (const [id, p] of players) groups[Number(id) >= 100 ? 1 : 0]?.append(new Option(p.name, id));
  $("valuePlayer").replaceChildren(...groups);
}

async function loadValue(playerId) {
  if (value.busy) return;
  value.busy = true;
  value.at = performance.now();
  value.t = state.lastT;
  const fetchValue = (id) => post("/api/value", { game: $("game").value, timestep: state.lastT, player_id: id == null ? null : Number(id), language: $("language").value });
  try {
    let id = playerId;
    let data = await fetchValue(id);
    if (id == null && data.board.length) {
      id = data.board[0].player_id;
      data = await fetchValue(id);
    }
    value.board = data.board;
    value.data = data.player;
    value.id = id == null ? null : String(id);
    if (value.id != null) $("valuePlayer").value = value.id;
    renderValue();
  } catch (error) {
    if (error.name !== "AbortError") $("valueCard").textContent = `Could not load player value (${error.message}).`;
  } finally {
    value.busy = false;
  }
}

function verdictClass(verdict) {
  return verdict === "bargain" ? "good" : verdict === "overpaid so far" ? "bad" : "fair";
}

function verdictName(verdict) {
  return VERDICTS.find(([key]) => key === verdict)?.[1] || "Fair";
}

function compareBar(name, amount, max, cls) {
  const fill = el("i", { class: cls });
  fill.style.width = `${Math.min(100, (amount / max) * 100)}%`;
  return el("div", { class: "cmp" }, el("span", {}, name), el("div", { class: "bar" }, fill), el("b", {}, amount.toFixed(1)));
}

function breakdownRow(row, max) {
  const open = value.openRows.has(row.action);
  const fill = el("i", { class: row.value_gbp < 0 ? "neg" : "" });
  fill.style.width = `${Math.max(2, (Math.abs(row.value_gbp) / max) * 100)}%`;
  const head = el("button", { type: "button", class: "v-row", "data-action-key": row.action, "aria-expanded": String(open) },
    el("span", { class: "v-name" }, `${row.label} ×${row.count}`, row.synthetic ? el("em", { class: "tag ghost synthetic" }, "synthetic") : null),
    el("div", { class: "bar" }, fill),
    el("b", {}, gbp(row.value_gbp)));
  const moments = el("div", { class: "v-moments" }, (row.moments.length ? row.moments : ["no moments yet"]).map((m) => el("span", { class: "tag ghost" }, m)));
  moments.hidden = !open;
  return el("div", {}, head, moments);
}

function pickText(pick, p) {
  return `You said ${verdictName(pick)}, the model says ${verdictName(p.verdict)}. ${p.why_value}`;
}

function verdictPanel(p) {
  const pick = readPicks()[value.id];
  return el("div", {},
    el("div", { class: "actions" }, VERDICTS.map(([key, text]) => el("button", { type: "button", "data-pick": key, "aria-pressed": String(pick === key) }, text))),
    el("p", { class: "v-result", dir: "auto" }, pick ? pickText(pick, p) : ""));
}

function renderValue() {
  const p = value.data;
  if (!p) {
    $("valueCard").textContent = "No player value yet.";
    return;
  }
  const max = Math.max(1, ...p.breakdown.map((r) => Math.abs(r.value_gbp)));
  const statMax = Math.max(1, ...p.role_stats.flatMap((r) => [r.player_per_match, r.role_per_match]));
  const meter = el("i", { class: verdictClass(p.verdict) });
  meter.style.width = `${Math.max(0, Math.min(3, p.value_ratio)) / 3 * 100}%`;
  $("valueCard").replaceChildren(
    el("h3", {}, p.name, el("span", { class: "tag ghost" }, p.match_clock)),
    el("p", { class: "v-sub" }, `${p.club} · ${p.role} · ${gbp(p.weekly_wage_gbp)} a week · ${p.contract_note}`),
    el("div", { class: "v-money" },
      el("div", {}, el("b", {}, gbp(p.match_value_gbp)), el("span", {}, "match value")),
      el("div", {}, el("b", {}, gbp(p.wage_so_far_gbp)), el("span", {}, "wage so far"))),
    el("div", { class: "meter-row" },
      el("div", { class: "bar meter", role: "img", "aria-label": `Value for money ${p.value_ratio.toFixed(2)}x` }, meter),
      el("b", {}, `${p.value_ratio.toFixed(2)}x`),
      el("span", { class: `badge-v ${verdictClass(p.verdict)}` }, p.verdict)),
    el("h4", {}, "Why they're paid that"),
    el("p", { dir: "auto" }, p.why_paid),
    ...p.role_stats.map((r) => el("div", { class: "cmp-group" }, el("span", { class: "cmp-title" }, r.label), compareBar("Player", r.player_per_match, statMax, ""), compareBar("Role avg", r.role_per_match, statMax, "avg"))),
    el("h4", {}, "Where the value came from"),
    ...p.breakdown.map((r) => breakdownRow(r, max)),
    el("p", { class: "v-why", dir: "auto" }, p.why_value),
    el("h4", {}, "Your call"),
    verdictPanel(p));
  renderBoard();
  if (tauri) reportHitRegions();
}

function renderBoard() {
  $("valueBoard").hidden = !value.boardOpen;
  $("valueBoardBtn").setAttribute("aria-expanded", String(value.boardOpen));
  const rows = value.board.length > 8 ? [...value.board.slice(0, 5), ...value.board.slice(-3)] : value.board;
  $("valueBoard").replaceChildren(...rows.map((b) => el("button", { type: "button", class: `b-row${String(b.player_id) === value.id ? " on" : ""}`, "data-player": b.player_id },
    el("span", { class: "b-name" }, `${b.name} · ${b.club}`),
    el("span", { class: `badge-v ${verdictClass(b.verdict)}` }, `${b.value_ratio.toFixed(1)}x`),
    el("b", {}, gbp(b.match_value_gbp)))));
}

function maybeRefreshValue() {
  if (!isOpen("valueWrap") || value.busy || state.paused) return;
  if (performance.now() - value.at < VALUE_REFRESH_MS || state.lastT === value.t) return;
  loadValue(value.id);
}

function rev(tag, text) {
  const node = el(tag, { dir: "auto" });
  reveal(node, text);
  return node;
}

function section(title, ...body) {
  return [el("h4", {}, tr(title)), ...body];
}

function renderCommentary(r, note) {
  $("commentary").replaceChildren(...[
    el("h3", {}, r.headline, el("span", { class: "tag ghost" }, r.match_clock)),
    r.observed.length ? section("Observed", el("ul", {}, r.observed.map((t) => rev("li", t)))) : null,
    section("Interpretation", rev("p", r.interpretation)),
    r.projection ? section("Projection", rev("p", r.projection)) : null,
    el("div", { class: "trail" }, r.context_trail.map((t) => el("span", { class: "tag ghost", title: t.kind }, t.label))),
    el("div", { class: "replies" }, r.quick_replies.map((q) => el("button", { type: "button", "data-reply": q }, tr(q)))),
    el("div", { class: "replies" }, FEEDBACK.map(([kind, text]) => el("button", { type: "button", "data-feedback": kind }, tr(text)))),
    note ? el("p", { class: "note" }, note) : null,
  ].flat().filter(Boolean));
}

function agentQuestion(action, question, playerId) {
  if (question) return question;
  if (action !== "focus_player") return ACTION_QUESTIONS[action];
  const id = playerId ?? ($("player").value || null);
  const person = id == null ? null : state.roster?.players?.[id];
  return person ? `Player focus on ${person.name} (player_id ${id}).` : "Focus on the player at the centre of the latest moment.";
}

async function askMatch(action, question = null, playerId = null, withAgent = true, note = null) {
  setAsk(true);
  const token = ++state.askToken;
  $("commentary").replaceChildren(el("span", { class: "tag ghost dots" }, "analysing"));
  const agent = withAgent && $("enrich").checked && !$("enrich").disabled;
  if (agent) $("answerWrap").classList.remove("open");
  try {
    const r = await post("/api/commentary", { game: $("game").value, timestep: state.lastT, action, question, player_id: playerId, preferences: preferences() });
    if (token !== state.askToken) return;
    state.last = r;
    renderCommentary(r, note);
  } catch (error) {
    if (token === state.askToken) $("commentary").textContent = error.name === "AbortError" ? "" : `Could not load commentary (${error.message}).`;
    return;
  }
  if (agent) ask(agentQuestion(action, question, playerId), action);
}

async function sendFeedback(kind) {
  const r = state.last;
  if (!r) return;
  const player = r.player_id ?? ($("player").value || null);
  try {
    const data = await post("/api/feedback", { preferences: preferences(), event: { kind, game_id: Number($("game").value), timestep: state.lastT, response_id: r.response_id, player_id: player == null ? null : Number(player) } });
    const { depth, tone, tactical, favorite_player_id: favorite } = data.preferences;
    const restyled = depth !== state.prefs.depth || tone !== state.prefs.tone || tactical !== state.prefs.tactical;
    state.prefs = { ...state.prefs, depth, tone, tactical };
    savePrefs();
    if (favorite != null && String(favorite) !== $("player").value) {
      $("player").value = String(favorite);
      $("player").dispatchEvent(new Event("change"));
    } else if (restyled) connect();
    askMatch(r.action, r.question, r.player_id, false, data.effects.join(" "));
  } catch (error) {
    status(`Feedback failed (${error.message}).`, true);
  }
}

let lastRegions = "";
function reportHitRegions() {
  const rects = [...document.querySelectorAll("[data-hit]")]
    .map((node) => node.getBoundingClientRect())
    .filter((r) => r.width > 0 && r.height > 0)
    .map((r) => [r.left, r.top, r.width, r.height]);
  const key = JSON.stringify(rects);
  if (key !== lastRegions) {
    lastRegions = key;
    tauri.core.invoke("set_hit_regions", { rects });
  }
}

async function init() {
  paintIcons();
  buildTiles();
  try {
    const meta = await (await fetch(`${API}/api/meta`)).json();
    state.roster = meta.roster;
    for (const g of meta.games) $("game").append(new Option(`Game ${g.game_id} · ${g.final_score}`, g.game_id));
    for (const [id, club] of Object.entries(meta.roster.clubs)) {
      $("club").append(new Option(club.name, id));
      $(`team${id}`).querySelector("b").textContent = club.short;
      $(`team${id}`).style.setProperty("--club", club.primary_color);
      $(`legend${id}`).textContent = club.short;
    }
    const players = Object.entries(meta.roster.players).sort((a, b) => Number(a[0]) - Number(b[0]));
    buildValueSelect();
    for (const [id, p] of players) $("player").append(new Option(`${p.name} (${meta.roster.clubs[Number(id) >= 100 ? "1" : "0"].short})`, id));
    restorePicks();
    $("enrich").checked = meta.enrich;
    $("enrich").disabled = !meta.enrich;
  } catch (err) {
    console.error("init failed", err);
    status(`${offlineMessage} (${err && err.message ? err.message : err})`, true);
    setPanel(true);
    return;
  }
  for (const id of controls) $(id).addEventListener("change", () => connect(id === "game"));
  buildDials($("dials"));
  buildDials($("onboardDials"));
  paintDials();
  document.addEventListener("input", (e) => {
    const key = e.target.dataset?.interest;
    if (!key) return;
    state.prefs.interests = { ...interests(), [key]: Number(e.target.value) };
    paintDials();
  });
  document.addEventListener("change", (e) => {
    if (!e.target.dataset?.interest) return;
    state.prefs.depth = PERSONA_DEPTH[persona()] || "balanced";
    savePrefs();
    applyPersona();
    if (!$("onboard").hidden) return;
    connect();
  });
  $("club").addEventListener("change", applyPersona);
  const showFavourite = () => {
    const p = state.roster.players[$("player").value];
    $("legendFav").hidden = !p;
    $("favName").textContent = p?.short || "";
  };
  $("player").addEventListener("change", showFavourite);
  for (const id of ["language", "club", "player"]) $(id).addEventListener("change", savePicks);
  showFavourite();
  applyPersona();
  $("dockBtn").onclick = () => setPanel(!isOpen("panelWrap"));
  $("restart").onclick = () => connect(true);
  $("pause").onclick = () => {
    state.paused = !state.paused;
    $("pause").textContent = state.paused ? "Resume" : "Pause";
    if (state.paused) { state.source?.close(); status("Paused"); } else connect();
  };
  if (STUDIO) {
    document.body.classList.add("studio");
    $("studio").hidden = false;
    $("sOnAir").onchange = (e) => { state.studio.onAir = e.target.checked; };
    document.querySelectorAll("[data-auto]").forEach((box) => box.onchange = (e) => { state.studio.autoAir[box.dataset.auto] = e.target.checked; });
    $("sConf").oninput = (e) => {
      state.studio.minConfidence = Number(e.target.value);
      $("sConfOut").textContent = `${e.target.value}%`;
    };
    $("sQueue").onclick = (e) => {
      const id = Number(e.target.closest("[data-id]")?.dataset.id);
      const item = state.studio.items.find((x) => x.id === id);
      if (!item) return;
      if (e.target.closest("[data-air]") && item.status !== "dropped") airItem(item);
      if (e.target.closest("[data-drop]") && item.status !== "aired") {
        item.status = "dropped";
        renderQueue();
      }
    };
  }
  $("askBtn").onclick = () => setAsk(!isOpen("askWrap"));
  $("valueBtn").onclick = () => setValue(!isOpen("valueWrap"));
  $("valuePlayer").onchange = () => loadValue($("valuePlayer").value);
  $("valueBoardBtn").onclick = () => {
    value.boardOpen = !value.boardOpen;
    renderBoard();
    if (tauri) reportHitRegions();
  };
  $("valueBoard").onclick = (e) => {
    const id = e.target.closest("[data-player]")?.dataset.player;
    if (id) loadValue(id);
  };
  $("valueCard").onclick = (e) => {
    const row = e.target.closest("[data-action-key]");
    const pick = e.target.closest("[data-pick]")?.dataset.pick;
    if (row) {
      const key = row.dataset.actionKey;
      if (!value.openRows.delete(key)) value.openRows.add(key);
      renderValue();
    }
    if (pick && value.data) {
      savePick(value.id, pick);
      renderValue();
    }
  };
  setInterval(maybeRefreshValue, 1000);
  $("focusBtn").onclick = () => askMatch("focus_player", null, playerId());
  $("recapBtn").onclick = () => askMatch("recap");
  $("askChips").onclick = (e) => {
    if (e.target.closest("[data-value]")) return setValue(true);
    const action = e.target.closest("[data-action]")?.dataset.action;
    if (action) askMatch(action, null, action === "focus_player" ? playerId() : null);
  };
  $("commentary").onclick = (e) => {
    const reply = e.target.closest("[data-reply]")?.dataset.reply;
    const kind = e.target.closest("[data-feedback]")?.dataset.feedback;
    if (kind) return sendFeedback(kind);
    if (!reply) return;
    const named = Object.keys(state.roster.players).find((id) => reply === `Focus on ${state.roster.players[id].short}`);
    if (reply.startsWith("Focus on ")) return askMatch("focus_player", null, named == null ? state.last?.player_id ?? null : Number(named));
    askMatch(REPLY_ACTIONS[reply] || "question", REPLY_ACTIONS[reply] ? null : reply);
  };
  $("ask").onsubmit = (e) => {
    e.preventDefault();
    const q = $("question").value.trim();
    if (q) askMatch("question", q);
  };
  let onboarded = false;
  try { onboarded = Boolean(localStorage.getItem("match-onboarded")); } catch { onboarded = false; }
  const backdrop = [...document.body.children].filter((n) => n !== $("onboard") && n.tagName !== "SCRIPT");
  const setOnboard = (open) => {
    $("onboard").hidden = !open;
    backdrop.forEach((n) => { n.inert = open; });
  };
  const closeOnboard = () => {
    setOnboard(false);
    try { localStorage.setItem("match-onboarded", "1"); } catch { /* storage unavailable */ }
  };
  $("onboardStart").onclick = () => {
    closeOnboard();
    $("dockBtn").focus();
    connect(true);
  };
  $("onboard").onkeydown = (e) => { if (e.key === "Escape") $("onboardStart").click(); };
  setOnboard(!onboarded);
  if (onboarded) connect(true);
  else $("onboardDials").querySelector("input").focus();
}

if (tauri) {
  tauri.event.listen("click-through", (e) => document.body.classList.toggle("locked", e.payload));
  $("close").onclick = () => tauri.window.getCurrentWindow().close();
  setInterval(reportHitRegions, 150);
}
if (!tauri) setPanel(true);
window.addEventListener("resize", sizeRadar);
sizeRadar();
requestAnimationFrame(draw);
init();
