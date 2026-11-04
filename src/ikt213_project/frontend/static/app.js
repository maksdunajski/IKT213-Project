const POLL_MS = 150;

const letterEl = document.getElementById("letter");
const meterEl = document.getElementById("meter-fill");
const statusEl = document.getElementById("status");
const modelEl = document.getElementById("model");
const modelErrorEl = document.getElementById("model-error");

function statusText(state) {
  if (state.error) return state.error;
  if (!state.model) return "No model loaded";
  if (!state.hand_detected) return "Show a hand to the camera";
  return `${Math.round(state.confidence * 100)}% confidence`;
}

function render(state) {
  letterEl.textContent = state.letter ?? "–";
  meterEl.style.width = `${(state.confidence ?? 0) * 100}%`;
  statusEl.textContent = statusText(state);
}

async function poll() {
  try {
    const response = await fetch("/api/state");
    render(await response.json());
  } catch {
    render({ error: "Lost connection to the server" });
  }
  setTimeout(poll, POLL_MS);
}

modelEl.addEventListener("change", async () => {
  modelEl.disabled = true;
  modelErrorEl.hidden = true;
  try {
    const response = await fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: modelEl.value }),
    });
    const body = await response.json();
    if (!response.ok) {
      modelErrorEl.textContent = body.error;
      modelErrorEl.hidden = false;
    }
    // After a failed switch this puts the menu back on the model still in use
    if (body.model) modelEl.value = body.model;
  } catch {
    modelErrorEl.textContent = "Lost connection to the server";
    modelErrorEl.hidden = false;
  }
  modelEl.disabled = false;
});

poll();
