const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = async (url, opt) => { const r = await fetch(url, opt); const j = await r.json(); if (!r.ok) throw new Error(j.error || "Something went wrong"); return j; };
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------- navigation ---------- */
function show(v) {
  $$(".view").forEach(e => e.classList.toggle("on", e.id === "v-" + v));
  $$(".nav").forEach(e => e.classList.toggle("on", e.dataset.view === v));
  if (v === "insights") loadInsights();
  if (v === "talk") setTimeout(() => $("#msg").focus(), 50);
  history.replaceState(null, "", "#" + v);
}
document.addEventListener("click", e => { const b = e.target.closest("[data-view]"); if (b) show(b.dataset.view); });

/* ---------- today: check-in ---------- */
const MOOD = ["Very low", "Low", "Okay", "Good", "Great"], LONE = ["Not at all", "A little", "Some", "Quite", "Very"];
function scale(id, name, labels) {
  $("#" + id).insertAdjacentHTML("beforeend", `<div class="opts">${labels.map((l, i) =>
    `<label><input type="radio" name="${name}" value="${i + 1}"><span><b>${i + 1}</b>${l}</span></label>`).join("")}</div>`);
}
scale("mood", "mood", MOOD); scale("lonely", "lonely", LONE);
const h = new Date().getHours();
$("#greet").textContent = (h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening") + ". How is today?";

function drawChart(rows) {
  const el = $("#chart");
  if (rows.length < 2) { el.innerHTML = `<p class="empty">Save a check-in on two different days to see your line here.</p>`; return; }
  const W = 560, H = 190, P = { l: 26, r: 10, t: 10, b: 24 };
  const days = [...Array(14)].map((_, i) => { const d = new Date(); d.setDate(d.getDate() - 13 + i); return d.toISOString().slice(0, 10); });
  const x = i => P.l + i * (W - P.l - P.r) / 13, y = v => P.t + (5 - v) * (H - P.t - P.b) / 4;
  const line = (k, col) => {
    const pts = days.map((d, i) => { const r = rows.find(r => r.day === d); return r ? [x(i), y(r[k])] : null; });
    let path = "", pen = false;
    pts.forEach(p => { if (!p) { pen = false; return; } path += (pen ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); pen = true; });
    return `<path d="${path}" fill="none" stroke="${col}" stroke-width="2.5" stroke-linejoin="round"/>` + pts.filter(Boolean).map(p => `<circle cx="${p[0]}" cy="${p[1]}" r="3.5" fill="${col}"/>`).join("");
  };
  const grid = [1, 2, 3, 4, 5].map(v => `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" stroke="#e3eaec"/><text x="4" y="${y(v) + 4}" font-size="11" fill="#5b6b76">${v}</text>`).join("");
  const ticks = [0, 6, 13].map(i => `<text x="${x(i)}" y="${H - 6}" font-size="11" fill="#5b6b76" text-anchor="${i === 0 ? "start" : i === 13 ? "end" : "middle"}">${days[i].slice(5)}</text>`).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mood and loneliness, last 14 days">${grid}${ticks}${line("loneliness", "#8a6fb5")}${line("mood", "#1f7a7a")}</svg>`;
}
function showTrend(t) {
  const n = $("#trendNote");
  n.className = "note";
  if (t.nudge === "down") { n.className = "note trend-down"; n.textContent = "Compared with your own earlier two weeks, this week looks heavier. That’s worth noticing. Talking with Saathi or someone you trust may help, and a counsellor is always an option."; }
  else if (t.nudge === "up") { n.className = "note trend-up"; n.textContent = "This week looks a little brighter than your earlier two weeks. Nice."; }
  else if (t.recent_n >= 3 && t.earlier_n >= 3) n.textContent = "Your week looks similar to your earlier two weeks.";
  else n.textContent = "Saathi compares this week with your own earlier two weeks. It needs about 3 check-ins in each.";
}
async function loadCheckins() {
  const d = await api("/api/checkins"); drawChart(d.entries); showTrend(d.trend); $("#consent").checked = d.consent;
  const today = d.entries.find(e => e.day === new Date().toISOString().slice(0, 10));
  if (today) { $(`input[name=mood][value="${today.mood}"]`).checked = true; $(`input[name=lonely][value="${today.loneliness}"]`).checked = true; }
}
$("#saveCheck").onclick = async () => {
  const m = $("input[name=mood]:checked"), l = $("input[name=lonely]:checked"), msg = $("#saveMsg");
  if (!m || !l) { msg.style.color = "#bf4b45"; msg.textContent = "Pick one option in each row."; return; }
  try { const d = await post("/api/checkins", { mood: +m.value, loneliness: +l.value }); msg.style.color = ""; msg.textContent = "Saved"; drawChart(d.entries); showTrend(d.trend); }
  catch (e) { msg.style.color = "#bf4b45"; msg.textContent = e.message; }
};

/* ---------- talk ---------- */
const log = $("#log"), hist = [];
let speakOn = false, voiceLang = "en-IN";
$("#speak").onchange = e => speakOn = e.target.checked;
function say(text) { if (!speakOn || !window.speechSynthesis) return; speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(text); u.lang = voiceLang; speechSynthesis.speak(u); }
function bubble(role, text, cls = "") { const d = document.createElement("div"); d.className = `msg ${role} ${cls}`; d.textContent = text; log.append(d); log.scrollTop = log.scrollHeight; return d; }
function crisisCard() {
  log.insertAdjacentHTML("beforeend", `<div class="crisis" role="alert"><h3>Please talk to a real person now</h3><div class="btns">
    <a href="tel:14416">Call Tele-MANAS 14416</a><a href="tel:18008914416">1-800-891-4416</a><a href="tel:112">Emergency 112</a></div></div>`);
  log.scrollTop = log.scrollHeight;
}
async function send(text) {
  text = text.trim(); if (!text) return;
  $("#chips").hidden = true; bubble("user", text); $("#msg").value = ""; autosize();
  const t = bubble("bot", "Saathi is writing…", "typing"); $("#send").disabled = true;
  try {
    const r = await post("/api/chat", { message: text, history: hist.slice(-16) });
    t.remove(); bubble("bot", r.reply); if (r.level === "crisis") crisisCard();
    hist.push({ role: "user", text }, { role: "bot", text: r.reply });
    voiceLang = r.lang === "hi" ? "hi-IN" : "en-IN"; say(r.reply);
  } catch (e) { t.remove(); bubble("bot", "I couldn’t reach the server. If you need help right now, call 14416 or 112."); }
  $("#send").disabled = false; $("#msg").focus();
}
bubble("bot", "Hello, I’m Saathi, an AI companion. I’m here to listen, not to diagnose or give medical advice. If you ever feel unsafe, I’ll point you to real people. What would you like to talk about?");
$("#send").onclick = () => send($("#msg").value);
$("#msg").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(e.target.value); } });
const autosize = () => { const t = $("#msg"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 120) + "px"; };
$("#msg").addEventListener("input", autosize);
$("#chips").addEventListener("click", e => { if (e.target.tagName === "BUTTON") send(e.target.textContent); });
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SR) {
  const mic = $("#mic"); mic.hidden = false; let rec;
  mic.onclick = () => {
    if (rec) { rec.stop(); return; }
    rec = new SR(); rec.lang = voiceLang; rec.interimResults = false;
    rec.onresult = e => { $("#msg").value = e.results[0][0].transcript; autosize(); };
    rec.onend = () => { mic.classList.remove("rec"); mic.textContent = "Mic"; rec = null; };
    mic.classList.add("rec"); mic.textContent = "Listening"; rec.start();
  };
}

/* ---------- insights ---------- */
const LVL = { low: "Low", moderate: "Moderate", high: "High" };
async function loadInsights() {
  const box = $("#insights"), d = await api("/api/risk");
  if (!d.ready) {
    box.innerHTML = `<div class="card empty-state narrow"><h2>The risk model has not been trained yet</h2>
      <p>This page shows results for held-out test participants, replayed as simulated users. To create them, run in the project folder:</p>
      <p><code>python -m src.train_models</code></p><p class="note">It reads the prepared train, validation and test files, trains the models, calibrates the probabilities, and writes <b>outputs/models/</b>. Then reload this page.</p></div>`;
    return;
  }
  const t = d.metrics?.test || {};
  const M = d.metrics || {};
  const m = `<div class="card metrics"><h2>How well did the model do on unseen participants?</h2><table>
    <tr><th>Chosen model</th><td>${esc(M.chosen_model)}</td></tr>
    <tr><th>Inputs</th><td>${esc(M.variant_description || "behaviour and context data")}</td></tr>
    <tr><th>ROC-AUC (test)</th><td>${t.roc_auc} <span class="note">vs ${t.baseline_start_score_only_auc} using only the starting questionnaire score</span></td></tr>
    <tr><th>Probability error (Brier)</th><td>${t.brier} <span class="note">lower is better</span></td></tr></table>
    <p class="note">${t.beats_baseline ? "The model beats the starting-score-only baseline." : "The model does not beat the starting-score-only baseline, so behaviour data adds little beyond the person’s own earlier score."}
    Explanations: ${esc(d.method || "not available")}. Factors show what pushed a score up or down; they are not causes.</p></div>`;
  const cards = d.users.map(u => {
    const mx = Math.max(...u.factors.map(f => f.size), 0.001), s = d.support[u.level];
    const mp = Math.round((d.thresholds?.moderate ?? 0.3) * 100), hp = Math.round((d.thresholds?.high ?? 0.6) * 100);
    return `<article class="card"><div class="u-head"><b>${esc(u.label)}</b><span class="pill ${u.level}">${LVL[u.level]} risk</span></div>
      <div>Estimated chance of a PHQ-9 score of 10 or more: <b>${Math.round(u.probability * 100)}%</b></div>
      <div class="gauge" style="--m:${mp}%;--h:${hp}%"><i style="left:${Math.min(99, u.probability * 100)}%"></i></div>
      <div class="gl"><span>0%</span><span>100%</span></div>
      <h3 style="font-size:.9rem;margin:12px 0 4px">What mattered most</h3>
      ${u.factors.map(f => `<div class="f"><span>${esc(f.name)} <span class="note">${f.direction} the estimate</span></span><s class="${f.direction === "raises" ? "up" : "dn"}" style="width:${Math.max(8, f.size / mx * 100)}%"></s></div>`).join("") || `<p class="note">No factor breakdown for this model type.</p>`}
      <div class="sup"><b>${esc(s.title)}</b><ul>${s.actions.map(a => `<li>${esc(a)}</li>`).join("")}</ul></div>
      <p class="note" style="margin-top:10px">Questionnaire later: ${u.questionnaire_end_for_evaluation_only} (evaluation only, not shown to the model)</p></article>`;
  }).join("");
  box.innerHTML = m + `<div class="cards">${cards}</div>`;
}

/* ---------- privacy ---------- */
$("#consent").onchange = e => post("/api/consent", { on: e.target.checked });
$("#wipe").onclick = async () => {
  if (!confirm("Delete all stored check-ins and settings? This cannot be undone.")) return;
  await api("/api/data", { method: "DELETE" }); $("#wipeMsg").textContent = "Deleted"; loadCheckins();
};

loadCheckins();
const first = (location.hash || "").slice(1);
show(document.getElementById("v-" + first) ? first : "today");
