let last = null;
let chatLog = [];
let features = [];
let isSliderMode = true; // start with slider mode

// DOM Elements
const diagEl = document.getElementById("diag");
const runRandomBtn = document.getElementById("runRandom");
const runManualBtn = document.getElementById("runManual");
const askBtn = document.getElementById("ask");
const modal = document.getElementById("manualModal");
const openModalBtn = document.getElementById("openManualBtn");
const closeModalBtn = document.getElementById("closeModalBtn");
const cancelModalBtn = document.getElementById("cancelModalBtn");
const toggleSliderBtn = document.getElementById("toggleSlider");
const toggleNumberBtn = document.getElementById("toggleNumber");
const sensorInputsContainer = document.getElementById("sensorInputs");

function setStatus(s) {
  const el = document.getElementById("status");
  el.textContent = s;
  el.className = "pill " + s;
}

function renderXai(rows) {
  const body = document.getElementById("xaiBody");
  body.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.component}</td><td>${Number(r.value).toFixed(2)}</td><td>${Number(r.health_impact).toFixed(2)}</td>`;
    body.appendChild(tr);
  }
}

function renderChat() {
  const el = document.getElementById("chat");
  el.innerHTML = chatLog.map((m) => {
    const isUser = m.role === "user";
    const nameColor = isUser ? 'var(--accent)' : '#fff';
    const textFormat = m.text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br/>');
    return `<div style="margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 6px; border-left: 3px solid ${isUser ? 'var(--accent)' : '#888'};">
      <div style="font-size: 11px; font-weight: bold; color: ${nameColor}; margin-bottom: 4px; text-transform: uppercase;">${m.role}</div>
      <div style="font-size: 14px; line-height: 1.4;">${textFormat}</div>
    </div>`;
  }).join("");
}

async function loadFeatures() {
  try {
    const res = await fetch("/features");
    if (res.ok) {
      features = await res.json();
      renderSensorInputs();
    }
  } catch (e) {
    console.error("Failed to load features", e);
  }
}

function renderSensorInputs() {
  sensorInputsContainer.innerHTML = "";
  features.forEach(f => {
    const div = document.createElement("div");
    div.className = "ctrl";
    const labelText = f.replace(/_/g, " ");
    
    // We create both inputs, and toggle their visibility
    div.innerHTML = `
      <div style="display:flex; justify-content:space-between;">
        <label for="slider_${f}">${labelText}</label>
        <span id="val_${f}" class="small" style="color:var(--accent)">0.50</span>
      </div>
      <input id="slider_${f}" class="sensor-slider" type="range" min="0" max="1" step="0.01" value="0.5" style="width:100%"/>
      <input id="num_${f}" class="sensor-num hidden" type="number" min="0" max="1" step="0.01" value="0.5" style="width:100%"/>
    `;
    sensorInputsContainer.appendChild(div);
    
    const slider = document.getElementById(`slider_${f}`);
    const num = document.getElementById(`num_${f}`);
    const text = document.getElementById(`val_${f}`);

    // Sync mechanisms
    slider.addEventListener("input", (e) => {
      text.textContent = Number(e.target.value).toFixed(2);
      num.value = e.target.value;
    });
    num.addEventListener("input", (e) => {
      let val = Number(e.target.value);
      if(val < 0) val = 0;
      if(val > 1) val = 1;
      text.textContent = val.toFixed(2);
      slider.value = val;
    });

    updateInputVisibility(f);
  });
}

function updateInputVisibility(f) {
  const slider = document.getElementById(`slider_${f}`);
  const num = document.getElementById(`num_${f}`);
  if (!slider || !num) return;
  if(isSliderMode) {
    slider.style.display = "block";
    num.style.display = "none";
  } else {
    slider.style.display = "none";
    num.style.display = "block";
  }
}

function applyToggleMode() {
  toggleSliderBtn.classList.toggle("active", isSliderMode);
  toggleNumberBtn.classList.toggle("active", !isSliderMode);
  features.forEach(f => updateInputVisibility(f));
}

toggleSliderBtn.addEventListener("click", () => {
  isSliderMode = true;
  applyToggleMode();
});
toggleNumberBtn.addEventListener("click", () => {
  isSliderMode = false;
  applyToggleMode();
});

// Modal Actions
openModalBtn.addEventListener("click", () => {
  if (features.length === 0) loadFeatures();
  modal.classList.remove("hidden");
});

function closeModal() {
  modal.classList.add("hidden");
}

closeModalBtn.addEventListener("click", closeModal);
cancelModalBtn.addEventListener("click", closeModal);

// Predictions
async function handlePredictionResult(res, scenarioText) {
  if (!res.ok) {
    diagEl.textContent = "Error: " + (await res.text());
    return;
  }

  last = await res.json();
  document.getElementById("pred").textContent = Number(last.prediction_km).toFixed(0);
  setStatus(last.status);
  renderXai(last.xai);
  document.getElementById("scenario").textContent = scenarioText || (last.scenario ? `Scenario: unit=${last.scenario.unit}, cycle=${last.scenario.cycle}` : "");

  const dres = await fetch("/diagnose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prediction_km: last.prediction_km, sensors: last.sensors }),
  });
  if (dres.ok) {
    const data = await dres.json();
    const formatted = data.markdown
      .split('\n')
      .filter(line => line.trim() !== '')
      .map(line => `<div style="padding: 10px; background: rgba(255,255,255,0.05); margin-bottom: 8px; border-radius: 6px; line-height: 1.5;">${line.replace(/\*\*(.*?)\*\*/g, '<strong style="color: var(--accent);">$1</strong>')}</div>`)
      .join('');
    diagEl.innerHTML = formatted;
  } else {
    diagEl.textContent = "Diagnosis unavailable (set GEMINI_API_KEY).";
  }
}

async function runRandomPrediction() {
  runRandomBtn.disabled = true;
  runRandomBtn.textContent = "Running Random...";
  diagEl.innerHTML = '<span class="loading">Loading Diagnosis...</span>';

  try {
    const res = await fetch("/predict/live");
    await handlePredictionResult(res);
  } finally {
    runRandomBtn.disabled = false;
    runRandomBtn.textContent = "Run Random Test";
  }
}

async function runManualPrediction() {
  runManualBtn.disabled = true;
  runManualBtn.textContent = "Running...";
  diagEl.innerHTML = '<span class="loading">Loading Diagnosis...</span>';

  const sensors = {};
  features.forEach(f => {
    // Both slider and num have synced values, just read slider since it's the source of truth if they are synced
    const el = document.getElementById(`slider_${f}`);
    if(el) sensors[f] = Number(el.value);
  });

  try {
    closeModal();
    const res = await fetch("/predict/manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sensors, seed: 42 }),
    });
    await handlePredictionResult(res, "Scenario: Manual Testing");
  } finally {
    runManualBtn.disabled = false;
    runManualBtn.textContent = "Run Manual Diagnostics";
  }
}

runRandomBtn.addEventListener("click", runRandomPrediction);
runManualBtn.addEventListener("click", runManualPrediction);

async function ask() {
  if (!last) return;
  const qEl = document.getElementById("q");
  const q = qEl.value.trim();
  if (!q) return;
  qEl.value = "";

  chatLog.push({ role: "user", text: q });
  renderChat();

  askBtn.disabled = true;
  askBtn.textContent = "Asking...";

  try {
    const worst = [...last.xai].sort((a, b) => a.health_impact - b.health_impact)[0]?.component || "Unknown";
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, prediction_km: last.prediction_km, worst_sensor: worst }),
    });
    const payload = res.ok ? await res.json() : { reply: "Chat unavailable (set GEMINI_API_KEY)." };
    chatLog.push({ role: "assistant", text: payload.reply });
    renderChat();
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
}

askBtn.addEventListener("click", ask);
