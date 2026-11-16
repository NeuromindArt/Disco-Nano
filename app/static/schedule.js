/* The schedule editor: a schedule's values over the render, drawn as levels to drag. */
"use strict";

/* Schedules hold a value for each part of the render, written [12]*400+[4]*600 over 1000 parts. The editor
   draws them as levels to drag. top is the highest value shown; log draws a wide range evenly. */
const SCHEDULES = {
  cut_overview: { top: 32, step: 1 }, cut_innercut: { top: 32, step: 1 },
  cut_ic_pow: { top: 100, step: 0.5, low: 0.5, log: true }, cut_icgray_p: { top: 1, step: 0.05 },
  time_travel_curve: { top: 4, step: 1 }, guidance_curve: { top: 3, step: 0.05 }, detail_curve: { top: 1, step: 0.05 },
  painters_curve: { top: 128, step: 4 }, graft_curve: { top: 1, step: 0.05 }, borrow_curve: { top: 1, step: 0.05 },
  ghost_curve: { top: 1, step: 0.05 }, edge_curve: { top: 1, step: 0.05 }, depth_curve: { top: 1, step: 0.05 },
  pose_curve: { top: 1, step: 0.05 }, person_curve: { top: 1, step: 0.05 }, face_curve: { top: 1, step: 0.05 }, face_shape_curve: { top: 1, step: 0.05 }, route_curve: { top: 1, step: 0.05 }, layout_curve: { top: 1, step: 0.05 },
  robust_curve: { top: 1, step: 0.05 }, one_subject_curve: { top: 1, step: 0.05 }, correct_curve: { top: 1, step: 1 },
  small_hold_curve: { top: 1, step: 0.05 },
};
const SCHEDULE_PART = /^\[(-?\d+(?:\.\d+)?)\]\*(\d+)$/;
let sched = null;

function parseSchedule(text) {
  const parts = String(text).replace(/\s/g, "").split("+").map((p) => p.match(SCHEDULE_PART));
  if (!parts.length || parts.some((m) => !m)) return null;
  const levels = parts.map((m) => ({ v: Number(m[1]), n: Number(m[2]) })).filter((l) => l.n > 0);
  const total = levels.reduce((s, l) => s + l.n, 0);
  if (!levels.length) return null;
  if (total < 1000) levels.at(-1).n += 1000 - total;  // the renderer holds the last value to the end
  let left = 1000;
  return levels.filter((l) => left > 0).map((l) => { const n = Math.min(l.n, left); left -= n; return { v: l.v, n }; }).filter((l) => l.n > 0);
}

function scheduleText(levels) {
  const merged = [];
  for (const l of levels) {
    if (merged.length && merged.at(-1).v === l.v) merged.at(-1).n += l.n;
    else merged.push({ ...l });
  }
  return merged.map((l) => `[${fmtValue(l.v)}]*${l.n}`).join("+");
}

function openSchedule(name) {
  const f = byName[name];
  const current = name in state.advanced ? state.advanced[name] : defaultOf(name);
  sched = { name, spec: { low: 0, ...SCHEDULES[name] }, levels: parseSchedule(current) || parseSchedule(defaultOf(name)), drag: null };
  withWords($("scheduleTitle"), `{${name}}`);
  setHelp($("scheduleTitle"), f.help);
  $("scheduleError").hidden = true;
  $("scheduleDialog").showModal();
  showSchedule();
}

function showSchedule() {
  if (document.activeElement !== $("scheduleText")) $("scheduleText").value = scheduleText(sched.levels);
  drawSchedule();
}

function commitSchedule() {
  const f = byName[sched.name];
  const text = scheduleText(sched.levels);
  if (text === scheduleText(parseSchedule(defaultOf(sched.name)))) delete state.advanced[sched.name];
  else state.advanced[sched.name] = text;
  const input = $(`adv-${sched.name}`);
  if (input) {
    input.value = text;
    input.closest(".adv-field")?.classList.toggle("changed", sched.name in state.advanced);
  }
  $("changedCount").textContent = changedLabel();
  showLook();
  showResets();
  showSchedulePreviews();
  showMoreCounts();
  save();
  updateFit();
}

function schedBox() {
  const c = $("scheduleCanvas");
  return { c, w: c.width, h: c.height, left: 56, right: 16, top: 22, bottom: 30 };
}

function schedY(v) {
  const b = schedBox(), { low, top, log } = sched.spec;
  const hi = Math.max(top, ...sched.levels.map((l) => l.v));
  const t = log ? Math.log(Math.max(v, low) / low) / Math.log(hi / low) : (v - low) / (hi - low);
  return b.top + (1 - t) * (b.h - b.top - b.bottom);
}

function schedValue(y) {
  const b = schedBox(), { low, top, log, step } = sched.spec;
  const hi = Math.max(top, ...sched.levels.map((l) => l.v));
  const t = Math.min(Math.max(1 - (y - b.top) / (b.h - b.top - b.bottom), 0), 1);
  const v = log ? low * (hi / low) ** t : low + t * (hi - low);
  return Math.max(low, Math.round(v / step) * step);
}

function schedX(at) { const b = schedBox(); return b.left + (at / 1000) * (b.w - b.left - b.right); }
function schedAt(x) { const b = schedBox(); return Math.round((((x - b.left) / (b.w - b.left - b.right)) * 1000) / 10) * 10; }

function drawSchedule() {
  const { c, w, h, left, right, top, bottom } = schedBox();
  const g = c.getContext("2d");
  const css = getComputedStyle(document.body);
  const color = (n) => css.getPropertyValue(n).trim();
  g.clearRect(0, 0, w, h);
  g.font = "11px sans-serif";
  g.lineWidth = 1;
  const { low, log } = sched.spec;
  const hi = Math.max(sched.spec.top, ...sched.levels.map((l) => l.v));
  for (let i = 0; i <= 4; i++) {
    const v = log ? low * (hi / low) ** (i / 4) : low + ((hi - low) * i) / 4;
    const y = schedY(v);
    g.strokeStyle = color("--line-strong");
    g.beginPath(); g.moveTo(left, y); g.lineTo(w - right, y); g.stroke();
    g.fillStyle = color("--ink-label");
    g.fillText(fmtValue(Number(v.toPrecision(3))), 6, y + 4);
  }
  for (const [at, label] of [[0, "Start"], [500, "Halfway"], [1000, "End"]]) {
    g.fillText(label, Math.min(Math.max(schedX(at) - 16, left), w - right - 30), h - 10);
  }
  let at = 0;
  sched.levels.forEach((l, i) => {
    const x0 = schedX(at), x1 = schedX(at + l.n), y = schedY(l.v);
    g.fillStyle = color("--accent-soft");
    g.fillRect(x0, y, x1 - x0, h - bottom - y);
    g.strokeStyle = color("--accent");
    g.lineWidth = sched.drag?.level === i ? 4 : 2.5;
    g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
    g.fillStyle = color("--ink");
    g.fillText(fmtValue(l.v), (x0 + x1) / 2 - 8, y - 6);
    at += l.n;
    if (i < sched.levels.length - 1) {
      g.strokeStyle = sched.drag?.edge === i ? color("--accent") : color("--ink-label");
      g.lineWidth = 1.5;
      g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(x1, top); g.lineTo(x1, h - bottom); g.stroke();
      g.setLineDash([]);
    }
  });
}

/* What is under the pointer: the line after level i (edge), or level i itself. */
function schedHit(e) {
  const c = $("scheduleCanvas"), box = c.getBoundingClientRect();
  const x = (e.clientX - box.left) * (c.width / box.width), y = (e.clientY - box.top) * (c.height / box.height);
  let at = 0;
  for (let i = 0; i < sched.levels.length; i++) {
    at += sched.levels[i].n;
    if (i < sched.levels.length - 1 && Math.abs(schedX(at) - x) < 6) return { edge: i, x, y };
  }
  at = 0;
  for (let i = 0; i < sched.levels.length; i++) {
    if (x >= schedX(at) - 2 && x <= schedX(at + sched.levels[i].n) + 2) return { level: i, x, y };
    at += sched.levels[i].n;
  }
  return { x, y };
}

function schedDown(e) {
  const hit = schedHit(e);
  if (hit.edge == null && hit.level == null) return;
  sched.drag = hit;
  $("scheduleCanvas").setPointerCapture(e.pointerId);
  drawSchedule();
}

function schedMove(e) {
  const c = $("scheduleCanvas");
  if (!sched?.drag) {
    if (sched) { const hit = schedHit(e); c.style.cursor = hit.edge != null ? "ew-resize" : hit.level != null ? "ns-resize" : "default"; }
    return;
  }
  const { x, y } = schedHit(e);
  const levels = sched.levels;
  if (sched.drag.edge != null) {
    const i = sched.drag.edge;
    const start = levels.slice(0, i).reduce((s, l) => s + l.n, 0);
    const end = start + levels[i].n + levels[i + 1].n;
    const at = Math.min(Math.max(schedAt(x), start + 10), end - 10);
    levels[i].n = at - start;
    levels[i + 1].n = end - at;
  } else {
    levels[sched.drag.level].v = schedValue(y);
  }
  showSchedule();
}

function schedUp() {
  if (!sched?.drag) return;
  sched.drag = null;
  commitSchedule();
  showSchedule();
}

function schedDoubleClick(e) {
  const hit = schedHit(e);
  const levels = sched.levels;
  if (hit.edge != null) {
    levels[hit.edge].n += levels[hit.edge + 1].n;  // the left side's value carries on
    levels.splice(hit.edge + 1, 1);
  } else if (hit.level != null) {
    const start = levels.slice(0, hit.level).reduce((s, l) => s + l.n, 0);
    const at = schedAt(hit.x) - start;
    const l = levels[hit.level];
    if (at < 10 || at > l.n - 10) return;
    levels.splice(hit.level, 1, { v: l.v, n: at }, { v: l.v, n: l.n - at });
  }
  commitSchedule();
  showSchedule();
}

function schedTyped() {
  const levels = parseSchedule($("scheduleText").value);
  $("scheduleError").hidden = !!levels;
  $("scheduleError").textContent = levels ? "" : "Write it as [value]*parts joined by +, such as [12]*400+[4]*600.";
  if (!levels) return;
  sched.levels = levels;
  commitSchedule();
  drawSchedule();
}

function setUpSchedule() {
  $("scheduleCanvas").addEventListener("pointerdown", schedDown);
  $("scheduleCanvas").addEventListener("pointermove", schedMove);
  $("scheduleCanvas").addEventListener("pointerup", schedUp);
  $("scheduleCanvas").addEventListener("pointercancel", schedUp);
  $("scheduleCanvas").addEventListener("dblclick", schedDoubleClick);
  $("scheduleText").addEventListener("change", schedTyped);
  $("scheduleDone").addEventListener("click", () => { $("scheduleDialog").close(); sched = null; });
  $("scheduleDialog").addEventListener("close", () => { sched = null; });
  $("scheduleReset").addEventListener("click", () => { sched.levels = parseSchedule(defaultOf(sched.name)); commitSchedule(); showSchedule(); });
}
