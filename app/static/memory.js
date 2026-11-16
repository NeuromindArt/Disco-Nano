/* The memory plan from the server, the VRAM, RAM and disk bars, and the settings its rules rule out. */
"use strict";

/* The server knows the card, the RAM and which models are downloaded, so it plans the memory and says which
   settings rule others out. The same rules refuse a render, so what's grayed out here is what can't run. */
let plan = null;
let planSent = "";
let planTimer = 0;

function currentSettings() {
  const settings = { ...state.advanced };
  for (const name of QUICK) settings[name] = state[name];
  settings.width_height = [state.width, state.height];
  if (state.init_image) settings.init_image = state.init_image;
  const regions = regionsFor();
  if (regions) settings.regions = regions;
  return settings;
}

function refreshPlan() {
  if (!fields.length) return;
  const body = JSON.stringify({ settings: currentSettings() });
  const gpu = lastStatus?.gpu ? `${lastStatus.gpu.used_gb}` : "";
  const key = body + gpu + (lastStatus?.task?.active ? "busy" : "");
  if (key === planSent) return;
  planSent = key;
  clearTimeout(planTimer);
  planTimer = setTimeout(async () => {
    try { plan = await post("/api/plan", JSON.parse(body)); } catch { return; }
    showPlan();
    showBlocked();
    showTime();
  }, 100);
}

const gb = (n) => `${n.toFixed(1)}GB`;

/* Draws parts end to end, as shares of the whole. A part past `limit` turns red where it crosses. Each part's
   kind is its color: other is the rest of the PC in gray, vram, ram and disk are Disco's own, and moved is
   working data offloaded from VRAM, drawn green inside RAM. A planned part is lighter than a measured one. */
function drawTrack(track, total, parts, keep = 0) {
  const limit = total - keep;
  const spans = [];
  let at = 0;
  for (const part of parts) {
    if (!(part.gb >= 0.01)) continue;
    const start = at, end = at + part.gb;
    at = end;
    const pieces = part.kind === "other" ? [[start, end, part.kind]] : [[start, Math.min(end, limit), part.kind], [Math.max(start, limit), end, "spill"]];
    for (const [a, b, kind] of pieces) {
      const shown = Math.min(b, total) - a;
      if (shown <= 0.004) continue;
      const span = document.createElement("span");
      span.className = kind + (part.plan && kind !== "spill" ? " plan" : "");
      span.style.width = `${(shown / total) * 100}%`;
      span.title = kind === "spill" ? `${part.label}: ${gb(part.gb)}, more than is free` : `${part.label}: ${gb(part.gb)}`;
      spans.push(span);
    }
  }
  if (keep > 0) {
    const span = document.createElement("span");
    span.className = "keep";
    span.style.width = `${(keep / total) * 100}%`;
    span.title = `Kept free so Windows can draw the screen: ${gb(keep)}`;
    spans.push(span);
  }
  track.replaceChildren(...spans);
  return at;
}

/* The working data an offloaded render keeps in RAM instead of VRAM. */
function movedOf(p) {
  return p?.offload ? (p.ram.parts.find((r) => r.kind === "offloaded")?.gb || 0) : 0;
}

let renderPlan = null;  // the plan of the render that is running, for the part of its RAM that came from VRAM

/* The status bar's VRAM and RAM. Solid is what is in use now: gray for the rest of the PC, and Disco in color while
   it renders. Between renders a hollow outline after it shows what the next render will add. The number is always
   what is in use now. */
function showMemory() {
  const s = lastStatus;
  const busy = !!s?.task?.active && s.task.kind === "render";
  if (!busy) renderPlan = null;
  // The meters show only what is in use now. What the next render needs stays in their hover text, so a meter
  // never shows memory that isn't used yet.
  let ahead = "";
  if (!busy && plan?.vram && plan.idle) {
    const needs = plan.vram.parts.reduce((t, p) => t + p.gb, 0), ramNeeds = plan.ram.parts.reduce((t, p) => t + p.gb, 0);
    ahead = plan.too_big ? "The next render is too big for this GPU even with offloading. Lower the size or use fewer prompt checkers."
      : `The next render needs about ${gb(needs)} of VRAM${plan.offload ? ", with part of its working data offloaded to RAM, which is slower" : ""}, and about ${gb(ramNeeds)} of RAM.`
        + (plan.ram.other + ramNeeds > plan.ram.total ? " That's more RAM than is free, so Windows will use the page file: it still runs, but slower." : "");
  }
  if (busy && !renderPlan) renderPlan = plan;
  const idle = s?.idle || {};
  if (s?.gpu) {
    const other = busy && idle.gpu ? Math.min(idle.gpu.used_gb, s.gpu.used_gb) : s.gpu.used_gb;
    drawTrack($("vramPlan"), s.gpu.total_gb, [{ kind: "other", label: "Other apps and the desktop", gb: other },
      { kind: "vram", label: "Disco, now", gb: s.gpu.used_gb - other }]);
    $("vramText").textContent = `${s.gpu.used_gb.toFixed(1)} / ${s.gpu.total_gb.toFixed(0)} GB`;
    $("vramText").classList.remove("over");
  } else {  // the driver didn't answer this time, so show nothing rather than an old reading
    drawTrack($("vramPlan"), 1, []);
    $("vramText").textContent = "–";
  }
  if (s?.ram) {
    const other = busy && idle.ram ? Math.min(idle.ram.used_gb, s.ram.used_gb) : s.ram.used_gb;
    const disco = s.ram.used_gb - other;
    const moved = Math.min(movedOf(renderPlan), disco);
    drawTrack($("ramPlan"), s.ram.total_gb, [{ kind: "other", label: "Other apps", gb: other },
      { kind: "ram", label: "Disco, now", gb: disco - moved }, { kind: "moved", label: "Offloaded from VRAM", gb: moved }]);
    $("ramText").textContent = `${s.ram.used_gb.toFixed(0)} / ${s.ram.total_gb.toFixed(0)} GB`;
    $("ramText").classList.remove("over");
  }
  $("memPlan").title = $("ramMeter").title = busy ? "What Disco uses now, after what the rest of the PC used before it started." : ahead;
}

/* The images' drive: Disco Nano's own files in color (its environment, models, inputs and images), the rest gray. */
function showDisk(disk) {
  if (!disk) {
    drawTrack($("diskPlan"), 1, []);
    $("diskText").textContent = "Drive not found";
    $("diskMeter").title = "The drive for your images can't be reached";
    return;
  }
  const disco = Math.min(disk.disco_gb ?? 0, disk.used_gb);
  drawTrack($("diskPlan"), disk.total_gb, [{ kind: "other", label: "Everything else", gb: disk.used_gb - disco },
    { kind: "disk", label: "Disco Nano: its models, environment and images", gb: disco }]);
  const low = disk.free_gb < 10;
  $("diskText").textContent = `${disk.free_gb} GB free`;
  $("diskText").classList.toggle("low", low);
  $("diskMeter").title = `The drive that holds ${disk.folder}.${disk.disco_gb == null ? " Measuring Disco's files." : ` Disco Nano uses ${gb(disco)} of it.`}`;
}

const showPlan = showMemory;

function controlsFor(name) {
  if (name === "animation_mode") return [...document.querySelectorAll('[data-anim="3D"]')];
  if (LAB_IMAGES.includes(name) || name === "image_prompts") return [...document.querySelectorAll(`[data-drop="${name}"]`)];
  if (name === "init_image") return [$("initDrop")];
  if (name === "handoff_file") return [$("handoffPick"), $("handoffOn")];
  if (name === "tile_scale") return [$("tilePick"), $("tilesOn")];
  const found = [$(name), $(`adv-${name}`), ...document.querySelectorAll(`[data-for="${name}"]`)];
  return found.filter((el) => el && el.type !== "hidden");
}

function rowOf(el) {
  return el.matches("[data-anim]") ? el : el.closest(".guide, .line, .slider-row, .check, .adv-field, .motion-row, .start-block") || el;
}

let ruled = [];

/* Grays out what the rules rule out, with the reason on hover. A setting that is already on stays usable and
   turns red instead, so it can be turned off. */
function showBlocked() {
  for (const { el, row, title } of ruled) {
    if (el.dataset.ruleOff) { el.disabled = false; delete el.dataset.ruleOff; }
    row.classList.remove("blocked", "clash");
    if (title == null) row.removeAttribute("title"); else row.title = title;
  }
  ruled = [];
  const blocked = plan?.blocked || {};
  const clashing = new Set(plan?.clashing || []);
  for (const [name, reason] of Object.entries(blocked)) {
    for (const el of controlsFor(name)) {
      const row = rowOf(el);
      if (ruled.some((r) => r.row === row)) continue;
      ruled.push({ el, row, title: row.getAttribute("title") });
      row.title = worded(reason);
      if (clashing.has(name)) { row.classList.add("clash"); continue; }
      row.classList.add("blocked");
      for (const input of row.matches("input, select, button") ? [row] : row.querySelectorAll("input, select, button")) {
        if (!input.disabled) { input.disabled = true; input.dataset.ruleOff = "1"; ruled.push({ el: input, row: input, title: input.getAttribute("title") }); }
      }
    }
  }
  const labOff = ["time_travel", "two_painters", "borrow_attention"].every((n) => blocked[n] && !clashing.has(n));
  const sum = $("sum-lab");
  sum.classList.toggle("blocked-sum", labOff);
  if (labOff) { sum.textContent = "Unavailable"; sum.title = worded(blocked.time_travel); } else sum.removeAttribute("title");
}
