"use strict";
const $ = s => document.querySelector(s), esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const J = (p, o) => fetch(p, o).then(r => r.json());
const TABS = [["home", "Overview"], ["today", "Today"], ["talk", "Saathi"], ["insights", "Risk insights"], ["help", "Help now"], ["privacy", "Privacy"]];
const nav = $("#nav");
function go(k) {
  TABS.forEach(([id]) => $("#v-" + id).classList.toggle("on", id === k));
  [...nav.children].forEach(b => b.setAttribute("aria-selected", b.dataset.k === k));
  history.replaceState(null, "", "#" + k);
}
TABS.forEach(([k, t]) => { const b = document.createElement("button"); b.textContent = t; b.dataset.k = k; b.onclick = () => go(k); nav.appendChild(b); });
go(TABS.some(t => t[0] === location.hash.slice(1)) ? location.hash.slice(1) : "home");

/* Overview */
$("#flow").innerHTML = [["1 DATA", "Wearable steps and sleep, cleaned and split by person"], ["2 MODEL", "Risk of a PHQ-9 score of 10 or more"], ["3 EXPLAIN", "SHAP shows which factors drove the estimate"], ["4 SUPPORT", "Low, moderate or high leads to different next steps"], ["5 SAATHI", "A safe, listen-first companion"]].map(a => `<li><b>${a[0]}</b>${a[1]}</li>`).join("");
const pct = x => Math.round(x * 100) + "%";
J("/api/risk").then(d => {
  if (!d.ready) { $("#kpis").innerHTML = '<div class="card">Model results are not available yet. Run <code>python -m src.train_advanced</code> once.</div>'; return; }
  const t = d.metrics.test, bo = d.metrics.cv_results.filter(r => r.variant === "behaviour_only").sort((a, b) => b.cv_auc_mean - a.cv_auc_mean)[0];
  const k = [[t.roc_auc.toFixed(3), "ROC-AUC on unseen test participants"], [t.baseline_start_score_only_auc.toFixed(3), "Baseline: starting PHQ-9 score only"], [bo ? bo.cv_auc_mean.toFixed(3) : "n/a", "Behavior only (cross-validation)"], [t.n_rows.toLocaleString(), "Test rows, never used for tuning"]];
  $("#kpis").innerHTML = k.map(x => `<div class="card kpi"><b>${x[0]}</b><span>${x[1]}</span></div>`).join("");
  $("#honest").innerHTML = `<b>How to read this:</b> the model ${t.beats_baseline ? "edges past" : "does not beat"} the start-score baseline, and behavior alone is clearly weaker. Estimates are risk indicators for people to review, not diagnoses.`;
  renderInsights(d);
});

/* Today */
let mood = 0, lon = 0;
function scale(id, set) { const el = $(id); for (let i = 1; i <= 5; i++) { const b = document.createElement("button"); b.type = "button"; b.textContent = i; b.setAttribute("aria-pressed", "false"); b.onclick = () => { [...el.children].forEach(x => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); set(i); }; el.appendChild(b); } }
scale("#mood", v => mood = v); scale("#lonely", v => lon = v);
const h = new Date().getHours(); $("#greet").textContent = (h < 12 ? "Good morning." : h < 18 ? "Good afternoon." : "Good evening.") + " A quick daily check-in. Your trend is compared with your own earlier days.";
function chart(d) {
  const e = d.entries; $("#consent").checked = !!d.consent;
  if (!e.length) { $("#chart").innerHTML = '<p class="mute">No check-ins yet.</p>'; $("#nudge").textContent = ""; return; }
  const w = 320, ht = 100, x = i => e.length > 1 ? 10 + i * (w - 20) / (e.length - 1) : w / 2, y = v => ht - 10 - (v - 1) * (ht - 20) / 4, ln = k => e.map((p, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p[k]).toFixed(1)).join(" ");
  $("#chart").innerHTML = `<svg viewBox="0 0 ${w} ${ht}" width="100%" role="img" aria-label="Mood and loneliness over the last three weeks"><path d="${ln("mood")}" fill="none" stroke="#1F7A6D" stroke-width="3"/><path d="${ln("loneliness")}" fill="none" stroke="#F2B84B" stroke-width="3"/></svg><div class="mute">Teal: mood. Amber: loneliness.</div>`;
  const t = d.trend; $("#nudge").textContent = t.nudge === "down" ? "This week looks lower than your own usual. Talking to Saathi or someone you trust may help." : t.nudge === "up" ? "This week looks better than your own usual. Nice." : t.mood_change === undefined ? "Add a few more days to see change from your own baseline." : "Your week looks close to your own usual.";
}
J("/api/checkins").then(chart);
$("#save").onclick = () => { const s = $("#saved"); if (!mood || !lon) { s.textContent = "Pick both ratings first."; return; } J("/api/checkins", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mood, loneliness: lon }) }).then(d => { if (d.error) { s.textContent = d.error; return; } s.textContent = "Saved."; chart(d); }); };

/* Saathi */
const log = $("#log"), hist = []; let lang = "en-IN";
function bubble(c, t) { const d = document.createElement("div"); d.className = "m " + c; d.textContent = t; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }
function say(t) { if (!$("#speak").checked || !window.speechSynthesis) return; speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(t); u.lang = lang; speechSynthesis.speak(u); }
function crisisCard(hl) { const d = document.createElement("div"); d.className = "m crisis"; d.innerHTML = `<b>You are not alone.</b> Free, 24x7 help: <a href="tel:${esc(hl.numbers[0])}">${esc(hl.name)} ${esc(hl.numbers[0])}</a>. Emergency: <a href="tel:${esc(hl.emergency)}">${esc(hl.emergency)}</a>.`; log.appendChild(d); log.scrollTop = log.scrollHeight; }
bubble("bot", "Good to see you. How has your day been? Take your time.");
["I feel lonely today", "I could not sleep well", "Mujhe thoda udaas lag raha hai", "I just want to talk"].forEach(t => { const b = document.createElement("button"); b.type = "button"; b.textContent = t; b.onclick = () => { $("#msg").value = t; send(); }; $("#chips").appendChild(b); });
function send() {
  const i = $("#msg"), text = i.value.trim(); if (!text) return; bubble("me", text); i.value = ""; const wait = bubble("bot", "…");
  J("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text, history: hist.slice(-10) }) }).then(r => {
    wait.remove(); if (r.error) { bubble("bot", r.error); return; }
    bubble(r.level === "crisis" ? "crisis" : "bot", r.reply); if (r.level === "crisis" && r.helplines) crisisCard(r.helplines);
    hist.push({ role: "user", text }, { role: "bot", text: r.reply }); lang = r.lang === "hi" ? "hi-IN" : "en-IN"; say(r.reply);
  }).catch(() => { wait.remove(); bubble("bot", "I could not reach the server. Please try again."); });
}
$("#send").onclick = send; $("#msg").addEventListener("keydown", e => { if (e.key === "Enter") send(); });
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SR) { $("#mic").onclick = () => { const r = new SR(); r.lang = lang; r.onresult = e => { $("#msg").value = e.results[0][0].transcript; send(); }; r.start(); }; } else $("#mic").hidden = true;

/* Risk insights */
const LV = { low: "Low", moderate: "Moderate", high: "High" };
function renderInsights(d) {
  const th = d.thresholds || {}, mp = (th.moderate || .27) * 100, hp = (th.high || .42) * 100;
  let html = d.users.map(u => { const mx = Math.max(...u.factors.map(f => f.size), .001), s = d.support[u.level];
    return `<article class="card"><div class="uh"><b>${esc(u.label)}</b><span class="pill ${u.level}">${LV[u.level]} risk</span></div>
<div>Estimated chance of a PHQ-9 score of 10 or more: <b>${Math.round(u.probability * 100)}%</b></div><div class="gauge" style="--m:${mp}%;--h:${hp}%"><i style="left:${Math.min(99, u.probability * 100)}%"></i></div>
${u.factors.map(f => `<div class="f"><span>${esc(f.name)}<br><span class="mute">${f.direction} the estimate</span></span><div><s class="${f.direction === "raises" ? "" : "dn"}" style="width:${Math.max(8, f.size / mx * 100)}%"></s></div></div>`).join("")}
<p><b>${esc(s.title)}</b></p><ul class="act">${s.actions.map(a => `<li>${esc(a)}</li>`).join("")}</ul></article>`; }).join("");
  const fg = d.metrics && d.metrics.fairness_auc_by_group;
  if (fg && typeof fg === "object") { const rows = Object.entries(fg).map(([k, v]) => { const n = typeof v === "number" ? v : (v && (v.roc_auc ?? v.auc)); return typeof n === "number" ? `<tr><td>${esc(k)}</td><td>${n.toFixed(3)}</td></tr>` : ""; }).join(""); if (rows) html += `<article class="card"><h3>Fairness: ROC-AUC by group</h3><table><tbody>${rows}</tbody></table><p class="mute">Compare groups, and treat small groups with care.</p></article>`; }
  $("#insights").innerHTML = html;
}

/* Help + Privacy */
J("/api/helplines").then(h => { $("#helpbox").innerHTML = `<div class="card"><h3>Immediate help</h3><p><b>${esc(h.name)}: <a href="tel:${esc(h.numbers[0])}">${esc(h.numbers[0])}</a></b><br>${esc(h.note)}</p><p><b>Emergency: <a href="tel:${esc(h.emergency)}">${esc(h.emergency)}</a></b></p></div>
<div class="card"><h3>Next steps</h3><ol><li>Tell one trusted person how you feel.</li><li>Book a visit with a counsellor or doctor.</li><li>Take your weekly mood summary with you.</li></ol></div>`; });
$("#consent").onchange = e => J("/api/consent", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ on: e.target.checked }) });
$("#wipe").onclick = () => { if (!confirm("Delete all check-ins and settings?")) return; fetch("/api/data", { method: "DELETE" }).then(() => { $("#wiped").textContent = "Deleted."; J("/api/checkins").then(chart); }); };
