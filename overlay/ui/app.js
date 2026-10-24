const tauri = window.__TAURI__;
const API = tauri ? "http://127.0.0.1:8765" : "";
const $ = (id) => document.getElementById(id);
const offlineMessage = tauri ? "Replay server offline. Run: python -m football_analytics.replay_server" : "Replay server unreachable.";
const controls = ["game", "speed", "persona", "language", "club", "player", "focus", "enrich"];
const MAX_FEED = 5;

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
const MAX_CHIPS = 2;
const PREF_KEY = "match-prefs";
const DEPTH_PRIORITIES = { headlines: ["high"], balanced: ["high", "medium"], deep: ["high", "medium", "low"] };
const ACTION_QUESTIONS = { explain_more: "Explain more", show_tactics: "Show tactics", simpler: "Make it simpler", numbers: "Give me the numbers", why: "Why did this matter?" };
const REPLY_ACTIONS = Object.fromEntries(Object.entries(ACTION_QUESTIONS).map(([action, text]) => [text, action]));
const FEEDBACK = [["too_basic", "Too basic"], ["too_technical", "Too technical"], ["more_tactical", "More tactical"], ["more_emotional", "More emotional"], ["follow_player", "Follow this player"]];
let savedPrefs = {};
try { savedPrefs = JSON.parse(localStorage.getItem(PREF_KEY)) || {}; } catch { savedPrefs = {}; }
const state = { prefs: { depth: "balanced", tone: "neutral", tactical: false, ...savedPrefs }, askToken: 0, last: null, roster: null, source: null, frames: [], lastT: 0, paused: false, alertTimer: null, alertToken: 0, alertQueue: [], alertBusy: false, recent: [], entries: new Map(), moments: new Map(), stats: {}, score: [0, 0] };

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
  return { persona: $("persona").value, language: $("language").value, club: $("club").value, player: $("player").value, focus: $("focus").value };
}

const PERSONAS = { casual: "Casual fan", superfan: "Superfan", analyst: "Tactical analyst", fantasy: "Fantasy player", producer: "Studio producer", newcomer: "New to football" };
const PERSONA_DEPTH = { analyst: "deep", producer: "deep", newcomer: "headlines" };

function analyst() {
  return !["casual", "newcomer"].includes($("persona").value);
}

function setPersona(persona) {
  $("persona").value = persona;
  $("persona").dispatchEvent(new Event("change"));
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
  return { persona: $("persona").value, language: $("language").value, favorite_club_id: club === "" ? null : Number(club), favorite_player_id: player === "" ? null : Number(player), metric_focus: $("focus").value || null, ...state.prefs };
}

function post(path, body) {
  return fetch(`${API}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then(async (response) => {
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
  if (!visiblePriorities().includes(cue.priority)) return;
  sting(cue);
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
  if (data) showAlert(data, () => setTimeout(nextAlert, 380));
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

const STINGS = { goal_threat: "Goal threat", momentum_swing: "Momentum shift", player_spotlight: "Player spotlight", control_index: "Tactical reset", chaos_index: "Tactical reset" };
const STING_GAP_MS = 12000;
let splashTimer = null;
let lastSting = -STING_GAP_MS;

function splash(text, ms, { accent = null, small = false } = {}) {
  clearTimeout(splashTimer);
  $("splashText").textContent = text;
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
  state.splashBusy = true;
  for (const [text, ms] of steps) await splash(text, ms);
  state.splashBusy = false;
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

function connect(fromStart = false) {
  state.source?.close();
  if (fromStart) {
    state.lastT = 0;
    state.score = [0, 0];
    state.frames = [];
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
    sequence([[`Full time ${f.score[0]}–${f.score[1]}`, 3600], ["Match recap", 2000]]).then(() => askMatch("question", "Give me the match recap so far.", null, false));
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

async function ask(question) {
  $("answerWrap").classList.add("open");
  $("answer").replaceChildren(el("span", { class: "tag ghost dots" }, "agents working"));
  try {
    const response = await fetch(`${API}/api/ask`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ game: $("game").value, timestep: state.lastT, question, ...profile(), depth: state.prefs.depth, tone: state.prefs.tone, tactical: state.prefs.tactical }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.status);
    const text = el("div", { dir: "auto" });
    $("answer").replaceChildren(...[
      el("span", { class: data.verified ? "tag ok" : "tag warn" }, data.verified ? "verified by gate" : "not verified by gate"),
      data.fallbacks?.length ? el("span", { class: "tag warn", title: data.fallbacks.join("; ") }, "partial") : null,
      text].filter(Boolean));
    reveal(text, data.answer);
  } catch (error) {
    $("answer").textContent = `Could not reach the agents (${error.message}).`;
  }
}

function setPanel(open) {
  $("panelWrap").classList.toggle("open", open);
  $("dockBtn").setAttribute("aria-expanded", String(open));
  if (open) setAsk(false);
}

function setAsk(open) {
  $("askWrap").classList.toggle("open", open);
  $("askBtn").setAttribute("aria-expanded", String(open));
  if (open && $("panelWrap").classList.contains("open")) setPanel(false);
}

function rev(tag, text) {
  const node = el(tag, { dir: "auto" });
  reveal(node, text);
  return node;
}

function section(title, ...body) {
  return [el("h4", {}, title), ...body];
}

function renderCommentary(r, note) {
  $("commentary").replaceChildren(...[
    el("h3", {}, r.headline, el("span", { class: "tag ghost" }, r.match_clock)),
    r.observed.length ? section("Observed", el("ul", {}, r.observed.map((t) => rev("li", t)))) : null,
    section("Interpretation", rev("p", r.interpretation)),
    r.projection ? section("Projection", rev("p", r.projection)) : null,
    el("div", { class: "trail" }, r.context_trail.map((t) => el("span", { class: "tag ghost", title: t.kind }, t.label))),
    el("div", { class: "replies" }, r.quick_replies.map((q) => el("button", { type: "button", "data-reply": q }, q))),
    el("div", { class: "replies" }, FEEDBACK.map(([kind, text]) => el("button", { type: "button", "data-feedback": kind }, text))),
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
    if (token === state.askToken) $("commentary").textContent = `Could not load commentary (${error.message}).`;
    return;
  }
  if (agent) ask(agentQuestion(action, question, playerId));
}

async function sendFeedback(kind) {
  const r = state.last;
  if (!r) return;
  const player = r.player_id ?? ($("player").value || null);
  try {
    const data = await post("/api/feedback", { preferences: preferences(), event: { kind, game_id: Number($("game").value), timestep: state.lastT, response_id: r.response_id, player_id: player == null ? null : Number(player) } });
    const { depth, tone, tactical, favorite_player_id: favorite } = data.preferences;
    state.prefs = { depth, tone, tactical };
    savePrefs();
    if (favorite != null && String(favorite) !== $("player").value) {
      $("player").value = String(favorite);
      $("player").dispatchEvent(new Event("change"));
    }
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
    for (const [id, p] of players) $("player").append(new Option(`${p.name} (${meta.roster.clubs[Number(id) >= 100 ? "1" : "0"].short})`, id));
    $("enrich").checked = meta.enrich;
    $("enrich").disabled = !meta.enrich;
  } catch {
    status(offlineMessage, true);
    setPanel(true);
    return;
  }
  for (const id of controls) $(id).addEventListener("change", () => connect(id === "game"));
  $("persona").addEventListener("change", () => {
    applyPersona();
    state.prefs.depth = PERSONA_DEPTH[$("persona").value] || "balanced";
    savePrefs();
  });
  $("club").addEventListener("change", applyPersona);
  $("player").addEventListener("change", () => {
    const p = state.roster.players[$("player").value];
    $("legendFav").hidden = !p;
    $("favName").textContent = p?.short || "";
  });
  $("dockBtn").onclick = () => setPanel(!$("panelWrap").classList.contains("open"));
  $("restart").onclick = () => connect(true);
  $("pause").onclick = () => {
    state.paused = !state.paused;
    $("pause").textContent = state.paused ? "Resume" : "Pause";
    if (state.paused) { state.source?.close(); status("Paused"); } else connect();
  };
  $("askBtn").onclick = () => setAsk(!$("askWrap").classList.contains("open"));
  $("focusBtn").onclick = () => askMatch("focus_player", null, Number($("player").value) || null);
  $("recapBtn").onclick = () => askMatch("question", "Give me the match recap so far.");
  $("askChips").onclick = (e) => {
    const action = e.target.closest("[data-action]")?.dataset.action;
    if (action) askMatch(action, null, action === "focus_player" ? Number($("player").value) || null : null);
  };
  $("commentary").onclick = (e) => {
    const reply = e.target.closest("[data-reply]")?.dataset.reply;
    const kind = e.target.closest("[data-feedback]")?.dataset.feedback;
    if (kind) return sendFeedback(kind);
    if (!reply) return;
    if (reply.startsWith("Focus on ")) return askMatch("focus_player", null, state.last?.player_id ?? null);
    askMatch(REPLY_ACTIONS[reply] || "question", REPLY_ACTIONS[reply] ? null : reply);
  };
  $("ask").onsubmit = (e) => {
    e.preventDefault();
    const q = $("question").value.trim();
    if (q) askMatch("question", q);
  };
  let onboarded = false;
  try { onboarded = Boolean(localStorage.getItem("match-onboarded")); } catch { onboarded = false; }
  const closeOnboard = () => {
    $("onboard").hidden = true;
    try { localStorage.setItem("match-onboarded", "1"); } catch { /* storage unavailable */ }
  };
  $("onboardChoices").replaceChildren(...Object.entries(PERSONAS).map(([id, name]) => el("button", { type: "button", "data-persona": id }, name)));
  $("onboardChoices").onclick = (e) => {
    const persona = e.target.closest("[data-persona]")?.dataset.persona;
    if (!persona) return;
    setPersona(persona);
    closeOnboard();
  };
  $("onboardSkip").onclick = closeOnboard;
  $("onboard").hidden = onboarded;
  connect(true);
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
