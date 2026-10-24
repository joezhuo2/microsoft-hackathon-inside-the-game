const tauri = window.__TAURI__;
const API = tauri ? "http://127.0.0.1:8765" : "";
const $ = (id) => document.getElementById(id);
const controls = ["game", "speed", "persona", "language", "club", "player", "focus", "enrich"];

const state = { roster: null, source: null, frames: [], lastT: 0, paused: false, liveTimer: null, pending: new Map() };

document.body.classList.add(tauri ? "tauri" : "web");

function profile() {
  return { persona: $("persona").value, language: $("language").value, club: $("club").value, player: $("player").value, focus: $("focus").value };
}

function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function teamColor(teamId) {
  const club = state.roster?.clubs?.[String(teamId)];
  return club?.primary_color || "var(--accent)";
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "style") node.style.cssText = value;
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
}

function statsBlock(stats) {
  if (!stats?.length) return null;
  return el("div", { class: "stats" }, stats.map((s) => el("div", {}, el("b", {}, `${s.value}${s.unit ? " " + s.unit : ""}`), el("span", {}, s.label))));
}

function showLive(data) {
  const o = data.overlay;
  const analyst = $("persona").value === "analyst";
  const node = el("div", { class: `card overlay ${o.template}`, style: data.team_id != null ? `--team:${teamColor(data.team_id)}` : "" },
    el("div", { class: "kicker" }, `${data.moment_type.replaceAll("_", " ")} · ${o.moment_id}`),
    el("h2", { dir: "auto" }, o.headline),
    el("p", { dir: "auto" }, o.body),
    analyst ? statsBlock(o.stats) : null);
  $("live").replaceChildren(node);
  clearTimeout(state.liveTimer);
  state.liveTimer = setTimeout(() => {
    node.classList.add("out");
    setTimeout(() => node.remove(), 400);
  }, o.display_seconds * 1000);
}

function trimFeed() {
  const feed = $("feed");
  while (feed.children.length > 4) feed.lastElementChild.remove();
}

function showPending(momentId) {
  const node = el("div", { class: "card pending" }, el("span", { class: "spinner" }), `Agents analysing ${momentId}…`);
  state.pending.set(momentId, node);
  $("feed").prepend(node);
  trimFeed();
}

function clearPending(momentId) {
  state.pending.get(momentId)?.remove();
  state.pending.delete(momentId);
}

function handoffChips(handoffs) {
  return el("div", { class: "handoffs" }, handoffs.map((h) => el("span", { title: h.error || "" }, `${h.agent.replace("match-", "")}${h.attempt > 1 ? " retry" : ""} ${h.seconds}s${h.error ? " ✕" : ""}`)));
}

function showEnriched(data) {
  clearPending(data.moment_id);
  const o = data.overlay;
  const verified = data.validation?.valid;
  const team = state.lastMoments?.get(data.moment_id)?.team_id;
  const node = el("div", { class: "card insight", style: team != null ? `--team:${teamColor(team)}` : "" },
    el("div", { class: "meta" },
      el("span", { class: "badge" }, "AI"),
      el("span", { class: `badge ${verified ? "ok" : "warn"}` }, verified ? "verified" : "unverified"),
      el("span", { class: "badge" }, data.language),
      data.cached ? el("span", { class: "badge" }, "cached") : null,
      el("span", {}, data.moment_id)),
    o ? el("h3", { dir: "auto" }, o.headline) : null,
    el("p", { dir: "auto" }, data.text || o?.body || ""),
    $("persona").value === "analyst" && o ? statsBlock(o.stats) : null,
    data.fallbacks?.length ? el("div", { class: "meta" }, el("span", { class: "badge warn" }, `recovered: ${data.fallbacks.join("; ")}`)) : null,
    handoffChips(data.handoffs || []));
  $("feed").prepend(node);
  trimFeed();
}

function applyFrame(f) {
  state.frames = [state.frames.at(-1) || f, { ...f, at: performance.now() }];
  state.lastT = f.t;
  $("score").textContent = `${f.score[0]} – ${f.score[1]}`;
  $("clock").textContent = f.clock;
  if (f.rhythm) {
    $("rhythm").textContent = f.rhythm.rhythm;
    $("rhythm").className = `pill ${f.rhythm.rhythm}`;
    $("momentum").style.left = `${50 + 50 * Math.max(-1, Math.min(1, f.rhythm.momentum_team0))}%`;
    $("poss").textContent = `${state.roster?.clubs?.["0"]?.short || "Home"} poss ${Math.round(f.rhythm.possession_share_team0 * 100)}%`;
  }
}

function connect(fromStart = false) {
  state.source?.close();
  if (fromStart) state.lastT = 0;
  const p = profile();
  const query = new URLSearchParams({ game: $("game").value, speed: $("speed").value, start: String(state.lastT), enrich: $("enrich").checked ? "1" : "0", ...p });
  state.lastMoments = new Map();
  state.pending.forEach((node) => node.remove());
  state.pending.clear();
  const source = new EventSource(`${API}/api/stream?${query}`);
  state.source = source;
  status("Connecting…");
  source.addEventListener("start", (e) => {
    const s = JSON.parse(e.data);
    status(`Live replay · game ${s.game_id} · ${s.enrich ? "AI analysis on" : "instant overlays only"}`);
  });
  source.addEventListener("frame", (e) => applyFrame(JSON.parse(e.data)));
  source.addEventListener("moment", (e) => {
    const data = JSON.parse(e.data);
    state.lastMoments.set(data.overlay.moment_id, data);
    showLive(data);
  });
  source.addEventListener("enriching", (e) => showPending(JSON.parse(e.data).moment_id));
  source.addEventListener("enriched", (e) => showEnriched(JSON.parse(e.data)));
  source.addEventListener("enrich_failed", (e) => {
    const data = JSON.parse(e.data);
    clearPending(data.moment_id);
    status(`Agent analysis failed for ${data.moment_id} (${data.error}); instant overlay kept.`, true);
  });
  source.addEventListener("end", (e) => {
    applyFrame(JSON.parse(e.data));
    status("Full time. Waiting for remaining analysis…");
  });
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED || !state.frames.length) {
      status(tauri ? "Replay server offline. Run: python -m football_analytics.replay_server" : "Replay server unreachable.", true);
      source.close();
    }
  };
}

const radar = $("radar");
const ctx = radar.getContext("2d");

function drawPitch(w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(14, 58, 32, 0.88)";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.lineWidth = 1;
  const sx = w / 120, sy = h / 80;
  ctx.strokeRect(4, 4, w - 8, h - 8);
  ctx.beginPath(); ctx.moveTo(w / 2, 4); ctx.lineTo(w / 2, h - 4); ctx.stroke();
  ctx.beginPath(); ctx.arc(w / 2, h / 2, 9.15 * sx, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeRect(4, (40 - 20.15) * sy, 16.5 * sx, 40.3 * sy);
  ctx.strokeRect(w - 4 - 16.5 * sx, (40 - 20.15) * sy, 16.5 * sx, 40.3 * sy);
}

function lerpPoints(a, b, k) {
  const prev = new Map((a.players || []).map((p) => [p[0], p]));
  return (b.players || []).map(([id, x, y]) => {
    const q = prev.get(id);
    return q ? [id, q[1] + (x - q[1]) * k, q[2] + (y - q[2]) * k] : [id, x, y];
  });
}

function draw() {
  const w = radar.width, h = radar.height;
  drawPitch(w, h);
  const [a, b] = state.frames;
  if (b) {
    const k = Math.min(1, (performance.now() - b.at) / 100);
    const sx = w / 120, sy = h / 80;
    const favourite = Number($("player").value || NaN);
    for (const [id, x, y] of lerpPoints(a, b, k)) {
      const team = id >= 100 ? 1 : 0;
      ctx.beginPath();
      ctx.arc(x * sx, y * sy, 5, 0, Math.PI * 2);
      ctx.fillStyle = teamColor(team);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 1;
      ctx.stroke();
      if (id === favourite) {
        ctx.beginPath();
        ctx.arc(x * sx, y * sy, 9, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffd23f";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.font = "11px Bahnschrift, Segoe UI, sans-serif";
        ctx.fillText(state.roster.players[String(id)]?.short || id, x * sx + 10, y * sy - 8);
      }
    }
    if (b.ball && a.ball) {
      const bx = a.ball[0] + (b.ball[0] - a.ball[0]) * k, by = a.ball[1] + (b.ball[1] - a.ball[1]) * k;
      ctx.beginPath();
      ctx.arc(bx * sx, by * sy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.stroke();
    }
  }
  requestAnimationFrame(draw);
}

async function ask(question) {
  $("answer").hidden = false;
  $("answer").textContent = "Agents working… (this can take a minute)";
  try {
    const response = await fetch(`${API}/api/ask`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ game: $("game").value, timestep: state.lastT, question, ...profile() }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.status);
    $("answer").replaceChildren(
      el("div", { class: "meta" }, el("span", { class: `badge ${data.verified ? "ok" : "warn"}` }, data.verified ? "verified by gate" : "not verified by gate"),
        data.fallbacks?.length ? el("span", { class: "badge warn" }, "partial: " + data.fallbacks.join("; ")) : null),
      el("div", { dir: "auto" }, data.answer));
  } catch (error) {
    $("answer").textContent = `Could not reach the agents (${error.message}).`;
  }
}

async function init() {
  try {
    const meta = await (await fetch(`${API}/api/meta`)).json();
    state.roster = meta.roster;
    for (const g of meta.games) $("game").append(new Option(`Game ${g.game_id} · ${g.final_score}`, g.game_id));
    for (const [id, club] of Object.entries(meta.roster.clubs)) {
      $("club").append(new Option(club.name, id));
      $(`club${id}`).querySelector("b").textContent = club.short;
      $(`club${id}`).querySelector("i").style.background = club.primary_color;
    }
    const players = Object.entries(meta.roster.players).sort((a, b) => Number(a[0]) - Number(b[0]));
    for (const [id, p] of players) $("player").append(new Option(`${p.name} (${meta.roster.clubs[Number(id) >= 100 ? "1" : "0"].short})`, id));
    $("enrich").checked = meta.enrich;
    $("enrich").disabled = !meta.enrich;
  } catch {
    status(tauri ? "Replay server offline. Run: python -m football_analytics.replay_server" : "Replay server unreachable.", true);
    return;
  }
  for (const id of controls) $(id).addEventListener("change", () => connect(id === "game"));
  $("restart").onclick = () => connect(true);
  $("pause").onclick = () => {
    state.paused = !state.paused;
    $("pause").textContent = state.paused ? "Resume" : "Pause";
    if (state.paused) { state.source?.close(); status("Paused"); } else connect();
  };
  $("focusBtn").onclick = () => {
    const id = $("player").value;
    if (!id) return status("Pick a player first.", true);
    ask(`Player focus on ${state.roster.players[id].name} (player_id ${id}).`);
  };
  $("recapBtn").onclick = () => ask("Give me the match recap so far.");
  $("ask").onsubmit = (e) => {
    e.preventDefault();
    const q = $("question").value.trim();
    if (q) ask(q);
  };
  connect(true);
}

if (tauri) {
  tauri.event.listen("click-through", (e) => document.body.classList.toggle("locked", e.payload));
  $("close").onclick = () => tauri.window.getCurrentWindow().close();
}
requestAnimationFrame(draw);
init();
