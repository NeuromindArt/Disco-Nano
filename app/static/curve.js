/* Keyframes, and the curve editor that draws and drags one setting's keyframes over a clip. */
"use strict";

function splitTop(text) {
  const parts = [];
  let depth = 0, current = "";
  for (const char of text) {
    depth += (char === "(") - (char === ")");
    if (char === "," && depth === 0) { parts.push(current); current = ""; } else current += char;
  }
  return [...parts, current];
}

function parseKeys(text) {
  const keys = [];
  for (const part of splitTop(String(text || ""))) {
    const at = part.indexOf(":");
    if (at < 0 || !/^\s*\d+\s*$/.test(part.slice(0, at))) continue;
    let expr = part.slice(at + 1).trim();
    if (expr.startsWith("(") && expr.endsWith(")")) expr = expr.slice(1, -1).trim();
    keys.push({ f: Number(part.slice(0, at)), expr });
  }
  return keys.sort((a, b) => a.f - b.f);
}

const keysText = (keys) => keys.map((k) => `${k.f}:(${k.expr})`).join(", ");

// The curve editor works on one setting's keyframes. The server works out the values, formulas included.
let curve = null;

function curveShape() { return state.advanced.interp_spline || byName.interp_spline.default; }

async function openCurve(name) {
  readControls();
  curve = { name, keys: parseKeys($(name).value), values: [], drag: null };
  if (!curve.keys.length) curve.keys = [{ f: 0, expr: fmtValue(Number($(`${name}_slider`).value)) }];
  withWords($("curveTitle"), `{${name}}`);
  $("curveShape").value = curveShape();
  $("curveDialog").showModal();
  await refreshCurve(true);
}

let curveBusy = false, curveAgain = false;
async function refreshCurve(rescale) {
  if (curveBusy) { curveAgain = true; return; }
  curveBusy = true;
  const asked = curve;
  const text = keysText(curve.keys);
  if (document.activeElement !== $("curveText")) $("curveText").value = text;
  const frames = Math.max(Number(state.max_frames) || 1, curve.keys.at(-1).f + 1, 2);
  try {
    const r = await api(`/api/curve?text=${encodeURIComponent(text)}&frames=${frames}&shape=${curveShape()}`);
    if (curve !== asked) return;  // the editor closed, or opened on another setting, while the server worked
    $("curveError").hidden = !r.error;
    $("curveError").textContent = r.error || "";
    if (!r.error) {
      curve.values = r.values;
      if (rescale || !curve.range) curve.range = curveRange();
      drawCurve();
    }
  } finally {
    curveBusy = false;
    if (curveAgain) { curveAgain = false; if (curve) refreshCurve(); }
  }
}

function curveRange() {
  const spec = MOTION[curve.name];
  const lo = Math.min(...curve.values), hi = Math.max(...curve.values);
  const span = Math.max(hi - lo, spec.step * 20);
  const mid = (hi + lo) / 2;
  return { lo: mid - span * 0.65, hi: mid + span * 0.65 };
}

function curveFrames() { return curve.values.length; }

function curveBox() {
  const c = $("curveCanvas");
  return { c, w: c.width, h: c.height, left: 56, right: 16, top: 14, bottom: 30 };
}

function toCanvas(f, v) {
  const b = curveBox(), { lo, hi } = curve.range;
  return [b.left + (f / Math.max(curveFrames() - 1, 1)) * (b.w - b.left - b.right),
    b.top + (1 - (v - lo) / (hi - lo)) * (b.h - b.top - b.bottom)];
}

function fromCanvas(x, y) {
  const b = curveBox(), { lo, hi } = curve.range;
  const f = Math.round(((x - b.left) / (b.w - b.left - b.right)) * Math.max(curveFrames() - 1, 1));
  const v = lo + (1 - (y - b.top) / (b.h - b.top - b.bottom)) * (hi - lo);
  return [Math.min(Math.max(f, 0), curveFrames() - 1), v];
}

function drawCurve() {
  const { c, w, h, left, right, top, bottom } = curveBox();
  const g = c.getContext("2d");
  const css = getComputedStyle(document.body);
  const color = (name) => css.getPropertyValue(name).trim();
  g.clearRect(0, 0, w, h);
  g.font = "11px sans-serif";
  g.fillStyle = color("--ink-label");
  g.strokeStyle = color("--line-strong");
  g.lineWidth = 1;
  const { lo, hi } = curve.range;
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    const [, y] = toCanvas(0, v);
    g.beginPath(); g.moveTo(left, y); g.lineTo(w - right, y); g.stroke();
    g.fillText(fmtValue(Number(v.toPrecision(3))), 6, y + 4);
  }
  const frames = curveFrames();
  const stepF = Math.max(1, Math.ceil(frames / 10 / 5) * 5);
  for (let f = 0; f < frames; f += stepF) {
    const [x] = toCanvas(f, lo);
    g.fillText(String(f), x - 6, h - 10);
  }
  const still = curve.name === "zoom" ? 1 : 0;  // the value that means no move
  if (still > lo && still < hi) {
    const [, y] = toCanvas(0, still);
    g.strokeStyle = color("--ink-disabled");
    g.setLineDash([4, 4]); g.beginPath(); g.moveTo(left, y); g.lineTo(w - right, y); g.stroke(); g.setLineDash([]);
  }
  // Ping-pong goes out on the first half's moves and comes back on the same ones, so the rest aren't used.
  if (lookValue("loop_mode") === "Ping-pong" && ["2D", "3D"].includes(state.animation_mode)) {
    const half = Math.floor((Number(state.max_frames) || frames) / 2);
    if (half < frames - 1) {
      const [x0] = toCanvas(half + 0.5, lo);
      g.fillStyle = color("--app");
      g.globalAlpha = 0.65;
      g.fillRect(x0, top, w - right - x0, h - top - bottom);
      g.globalAlpha = 1;
      g.fillStyle = color("--ink-label");
      g.fillText("Ping-pong plays the first half back", Math.min(x0 + 6, w - right - 200), top + 12);
    }
  }
  g.strokeStyle = color("--accent");
  g.lineWidth = 2;
  g.beginPath();
  curve.values.forEach((v, f) => { const [x, y] = toCanvas(f, v); if (f) g.lineTo(x, y); else g.moveTo(x, y); });
  g.stroke();
  for (const k of curve.keys) {
    const [x, y] = toCanvas(k.f, curve.values[Math.min(k.f, frames - 1)]);
    g.beginPath(); g.arc(x, y, 5, 0, Math.PI * 2);
    g.fillStyle = NUMBER.test(k.expr) ? color("--accent") : color("--panel");
    g.fill(); g.strokeStyle = color("--accent"); g.lineWidth = 1.5; g.stroke();
  }
}

function canvasPoint(e) {
  const c = $("curveCanvas"), box = c.getBoundingClientRect();
  return [(e.clientX - box.left) * (c.width / box.width), (e.clientY - box.top) * (c.height / box.height)];
}

function keyNear(x, y) {
  return curve.keys.findIndex((k) => {
    const [kx, ky] = toCanvas(k.f, curve.values[Math.min(k.f, curveFrames() - 1)]);
    return Math.hypot(kx - x, ky - y) < 9;
  });
}

function snap(v) { const step = MOTION[curve.name].step / 5; return Math.round(v / step) * step; }

function curvePointerDown(e) {
  const [x, y] = canvasPoint(e);
  const at = keyNear(x, y);
  if (at < 0 || !NUMBER.test(curve.keys[at].expr)) return;
  curve.drag = at;
  $("curveCanvas").setPointerCapture(e.pointerId);
}

function curvePointerMove(e) {
  if (curve?.drag == null) return;
  const [x, y] = canvasPoint(e);
  let [f, v] = fromCanvas(x, y);
  const keys = curve.keys, i = curve.drag;
  const lowF = i ? keys[i - 1].f + 1 : 0, highF = i < keys.length - 1 ? keys[i + 1].f - 1 : curveFrames() - 1;
  keys[i] = { f: i === 0 ? 0 : Math.min(Math.max(f, lowF), highF), expr: fmtValue(snap(v)) };
  refreshCurve();
}

function curvePointerUp() {
  if (curve?.drag == null) return;
  curve.drag = null;
  curve.range = curveRange();
  refreshCurve();
}

function curveDoubleClick(e) {
  const [x, y] = canvasPoint(e);
  const at = keyNear(x, y);
  if (at >= 0) {
    if (curve.keys.length > 1 && curve.keys[at].f !== 0) curve.keys.splice(at, 1);
  } else {
    const [f, v] = fromCanvas(x, y);
    if (!curve.keys.some((k) => k.f === f)) curve.keys.push({ f, expr: fmtValue(snap(v)) });
    curve.keys.sort((a, b) => a.f - b.f);
  }
  refreshCurve();
}

function closeCurve() {
  if (!curve) return;
  $(curve.name).value = keysText(curve.keys);
  $(curve.name).dispatchEvent(new Event("change", { bubbles: true }));
  curve = null;
  $("curveDialog").close();
}

function setCurveShape() {
  const shape = $("curveShape").value;
  if (shape === byName.interp_spline.default) delete state.advanced.interp_spline;
  else state.advanced.interp_spline = shape;
  if ($("adv-interp_spline")) $("adv-interp_spline").value = shape;
  $("changedCount").textContent = changedLabel();
  save();
  refreshCurve(true);
}

function setUpCurve() {
  $("curveDone").addEventListener("click", closeCurve);
  $("curveDialog").addEventListener("cancel", (e) => { e.preventDefault(); closeCurve(); });
  $("curveShape").addEventListener("change", setCurveShape);
  $("curveText").addEventListener("change", () => { curve.keys = parseKeys($("curveText").value); if (curve.keys.length) refreshCurve(true); });
  $("curveCanvas").addEventListener("pointerdown", curvePointerDown);
  $("curveCanvas").addEventListener("pointermove", curvePointerMove);
  $("curveCanvas").addEventListener("pointerup", curvePointerUp);
  $("curveCanvas").addEventListener("pointercancel", curvePointerUp);
  $("curveCanvas").addEventListener("dblclick", curveDoubleClick);
}
