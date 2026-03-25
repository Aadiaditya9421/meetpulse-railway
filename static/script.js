/* MeetPulse v3 — script.js */
"use strict";

const API_BASE = window.location.origin;

const SAMPLES = [
  "Great progress on the sprint today! The team delivered all user stories ahead of schedule and the demo went exceptionally well. Client loved the new features and stakeholder confidence is very high.",
  "We are significantly behind schedule on this project. Multiple critical blockers remain unresolved. The build pipeline keeps failing and the client is starting to escalate. Team morale is very low.",
  "The team reviewed the backlog during today's planning session and estimated story points for the upcoming sprint. Architecture trade-offs for the new microservices approach were also discussed.",
  "Excellent collaboration across all teams this week. The code review process is working well and quality has improved noticeably. Everyone is aligned on the roadmap and motivated to deliver.",
  "Production is down again due to another critical bug introduced in yesterday's deployment. Customer impact is severe. We need an emergency hotfix immediately and a proper post-mortem."
];

let historyLog = [];
let sampleIdx  = 0;

const textarea      = document.getElementById("transcript");
const charCount     = document.getElementById("charCount");
const analyzeBtn    = document.getElementById("analyzeBtn");
const explainBtn    = document.getElementById("explainBtn");
const clearBtn      = document.getElementById("clearBtn");
const sampleBtn     = document.getElementById("sampleBtn");
const clearHistBtn  = document.getElementById("clearHistoryBtn");
const statusBadge   = document.getElementById("statusBadge");
const idleState     = document.getElementById("idleState");
const loader        = document.getElementById("loader");
const results       = document.getElementById("results");
const errorBox      = document.getElementById("errorBox");
const errorMsg      = document.getElementById("errorMsg");
const historyBody   = document.getElementById("historyBody");
const explainPanel  = document.getElementById("explainPanel");
const explainList   = document.getElementById("explainList");
const requestId     = document.getElementById("requestId");
const refreshMetBtn = document.getElementById("refreshMetricsBtn");

async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === "ok") {
      statusBadge.innerHTML = `<i class="bi bi-circle-fill me-1" style="font-size:.5rem"></i>Live`;
      statusBadge.className = "badge bg-success-subtle text-success border border-success-subtle";
      document.getElementById("modelType").textContent = data.active_model || "MLP";
    } else {
      statusBadge.innerHTML = `<i class="bi bi-circle-fill me-1" style="font-size:.5rem"></i>Model Not Ready`;
      statusBadge.className = "badge bg-warning-subtle text-warning border border-warning-subtle";
    }
  } catch {
    statusBadge.innerHTML = `<i class="bi bi-circle-fill me-1" style="font-size:.5rem"></i>Offline`;
    statusBadge.className = "badge bg-danger-subtle text-danger border border-danger-subtle";
  }
}

function showState(state) {
  idleState.classList.add("d-none");
  loader.classList.add("d-none");
  results.classList.add("d-none");
  errorBox.classList.add("d-none");
  if (state === "idle")    idleState.classList.remove("d-none");
  if (state === "loading") loader.classList.remove("d-none");
  if (state === "results") results.classList.remove("d-none");
  if (state === "error")   errorBox.classList.remove("d-none");
}

function sentimentIcon(s) {
  return s === "Positive" ? "😊" : s === "Negative" ? "😟" : "😐";
}
function sentimentClass(s) {
  return s === "Positive" ? "positive" : s === "Negative" ? "negative" : "neutral";
}
function badgeClass(s) {
  return s === "Positive" ? "badge-pos" : s === "Negative" ? "badge-neg" : "badge-neu";
}
function confBarClass(s) {
  return s === "Positive" ? "bg-success" : s === "Negative" ? "bg-danger" : "bg-warning";
}
function confLevelBadgeClass(level) {
  return level === "high" ? "bg-success" : level === "moderate" ? "bg-warning text-dark" : "bg-danger";
}

function renderResults(data) {
  const cls = sentimentClass(data.prediction);

  const badge = document.getElementById("sentimentBadge");
  badge.className = `sentiment-badge ${cls} mx-auto mb-2`;
  badge.textContent = sentimentIcon(data.prediction);

  const lbl = document.getElementById("sentimentLabel");
  lbl.textContent = data.prediction;
  lbl.style.color = cls === "positive" ? "#16a34a" : cls === "negative" ? "#dc2626" : "#d97706";

  const conf = data.confidence;
  const level = data.confidence_level || (conf >= 70 ? "high" : conf >= 55 ? "moderate" : "low");
  document.getElementById("confidenceVal").textContent = `${conf}%`;

  const confLvlBadge = document.getElementById("confLevelBadge");
  confLvlBadge.textContent = level.charAt(0).toUpperCase() + level.slice(1);
  confLvlBadge.className = `badge ${confLevelBadgeClass(level)}`;

  const bar = document.getElementById("confidenceBar");
  bar.style.width = `${conf}%`;
  bar.className   = `progress-bar ${confBarClass(data.prediction)}`;

  const lowWarn = document.getElementById("lowConfWarn");
  if (level === "low" || level === "moderate") {
    lowWarn.classList.remove("d-none");
  } else {
    lowWarn.classList.add("d-none");
  }

  const breakdown = document.getElementById("scoreBreakdown");
  breakdown.innerHTML = "";
  const colorMap = { Positive: "#16a34a", Negative: "#dc2626", Neutral: "#d97706" };
  Object.entries(data.scores)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, pct]) => {
      breakdown.innerHTML += `
        <div class="d-flex align-items-center score-row mb-1">
          <span class="score-label">${label}</span>
          <div class="score-bar-wrap flex-grow-1 mx-2">
            <div class="score-bar-fill" style="width:${pct}%;background:${colorMap[label]}"></div>
          </div>
          <span class="score-pct fw-semibold">${pct}%</span>
        </div>`;
    });

  document.getElementById("wordCount").textContent  = data.word_count;
  document.getElementById("latency").textContent    = data.latency_ms;
  document.getElementById("modelUsed").textContent  = (data.model_used || "MLP").replace("Classifier","");
  document.getElementById("modelF1").textContent    = data.model_f1 || "0.8805";
  requestId.textContent = data.request_id ? `rid: ${data.request_id}` : "";

  explainPanel.classList.add("d-none");
  showState("results");
}

async function analyze() {
  const text = textarea.value.trim();
  if (!text) { showState("error"); errorMsg.textContent = "Please enter some meeting transcript text before analyzing."; return; }
  if (text.length > 5000) { showState("error"); errorMsg.textContent = "Text exceeds 5000 character limit."; return; }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span>Analyzing…`;
  showState("loading");

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || `Server error: ${res.status}`); }
    const data = await res.json();
    renderResults(data);
    addHistory(text, data);
  } catch (err) {
    showState("error"); errorMsg.textContent = err.message || "Network error — is the API running?";
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `<i class="bi bi-lightning-fill me-1"></i>Analyze`;
  }
}

async function explainPrediction() {
  const text = textarea.value.trim();
  if (!text) { alert("Enter text first, then click Explain."); return; }

  explainBtn.disabled = true;
  explainBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span>`;

  try {
    const res = await fetch(`${API_BASE}/predict/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail); }
    const data = await res.json();

    if (results.classList.contains("d-none")) {
      // If no analysis yet, show a minimal result
      document.getElementById("sentimentBadge").textContent = sentimentIcon(data.prediction);
      document.getElementById("sentimentLabel").textContent = data.prediction;
      showState("results");
    }

    explainPanel.classList.remove("d-none");
    explainList.innerHTML = data.top_features.map(f => `
      <div class="d-flex align-items-center gap-2 mb-1" style="font-size:.87rem">
        <span class="badge ${f.sentiment_direction === 'positive' ? 'bg-success' : 'bg-danger'}">
          ${f.sentiment_direction === 'positive' ? '▲' : '▼'}
        </span>
        <code>${f.word}</code>
        <span class="text-muted">score: ${f.tfidf_score}</span>
      </div>`).join("");
  } catch (err) {
    alert("Explain error: " + err.message);
  } finally {
    explainBtn.disabled = false;
    explainBtn.innerHTML = `<i class="bi bi-search me-1"></i>Explain`;
  }
}

function addHistory(text, data) {
  const level = data.confidence_level || (data.confidence >= 70 ? "high" : data.confidence >= 55 ? "moderate" : "low");
  historyLog.unshift({
    idx:        historyLog.length + 1,
    excerpt:    text.slice(0, 60) + (text.length > 60 ? "…" : ""),
    sentiment:  data.prediction,
    confidence: data.confidence,
    level:      level,
    words:      data.word_count,
    time:       new Date().toLocaleTimeString()
  });
  renderHistory();
}

function renderHistory() {
  if (historyLog.length === 0) {
    historyBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted py-4">No analyses yet.</td></tr>`;
    return;
  }
  historyBody.innerHTML = historyLog.slice(0, 20).map(e => `
    <tr>
      <td class="text-muted">${e.idx}</td>
      <td style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${e.excerpt}</td>
      <td><span class="badge ${badgeClass(e.sentiment)}">${e.sentiment}</span></td>
      <td>${e.confidence}%</td>
      <td><span class="badge ${confLevelBadgeClass(e.level)}">${e.level}</span></td>
      <td>${e.words}</td>
      <td class="text-muted">${e.time}</td>
    </tr>`).join("");
}

async function loadMetrics() {
  try {
    const res  = await fetch(`${API_BASE}/metrics`);
    const data = await res.json();
    document.getElementById("mTotal").textContent  = data.total_requests;
    document.getElementById("mAvgMs").textContent  = data.avg_latency_ms;
    document.getElementById("mPos").textContent    = data.sentiment_distribution?.Positive ?? 0;
    document.getElementById("mNeg").textContent    = data.sentiment_distribution?.Negative ?? 0;
    document.getElementById("mNeu").textContent    = data.sentiment_distribution?.Neutral  ?? 0;
    document.getElementById("mLowConf").textContent= data.low_confidence_count ?? 0;
  } catch {
    // silent fail
  }
}

textarea.addEventListener("input", () => {
  charCount.textContent = textarea.value.length;
  charCount.style.color = textarea.value.length > 4800 ? "red" : "";
});

analyzeBtn.addEventListener("click", analyze);
explainBtn.addEventListener("click", explainPrediction);
clearBtn.addEventListener("click", () => { textarea.value = ""; charCount.textContent = "0"; showState("idle"); explainPanel.classList.add("d-none"); });
sampleBtn.addEventListener("click", () => { textarea.value = SAMPLES[sampleIdx % SAMPLES.length]; charCount.textContent = textarea.value.length; sampleIdx++; showState("idle"); });
clearHistBtn.addEventListener("click", () => { historyLog = []; renderHistory(); });
refreshMetBtn.addEventListener("click", loadMetrics);
textarea.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze(); });

checkHealth();
loadMetrics();
setInterval(checkHealth, 30000);
setInterval(loadMetrics, 60000);
