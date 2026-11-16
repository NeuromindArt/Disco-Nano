/* Regional prompts: boxes on the frame, each with a prompt of its own, set in their own card. They are kept in
   state.regions as fractions of the frame, so they fit any size, and go with every render while the card's switch
   is on. Opened while a render runs, the editor pauses it on the preview, and Continue sends the boxes to it. */
const REGION_COLORS = ["#00ff66", "#38bdf8", "#f59e0b", "#e879f9", "#f43f5e", "#a3e635"];
const REGION_MIN = 0.03;  // a drag smaller than this, as a share of the frame, is a click
let regionDraft = null;   // the boxes being edited: { boxes: [{ x, y, w, h, text }] }
let regionMode = null;    // "set" before a render, "run" while one is paused for it
let regionPicked = -1;
let regionHeld = false;   // the render this editor paused has reached its pause, or been asked for one

function regionsState() {
  state.regions ??= { on: false, strength: 0.75, boxes: [] };
  return state.regions;
}

function regionsSet() { return !!state.regions?.on && state.regions.boxes.some((b) => b.text.trim()); }

/* The regions a render gets: only boxes with a prompt, and none while the switch is off. */
function regionsFor() {
  if (!regionsSet()) return null;
  return { strength: state.regions.strength, boxes: state.regions.boxes.filter((b) => b.text.trim()) };
}

/* The card: its switch, Strength, the boxes as a list, and the button that opens the editor. */
function showRegionsButton(t) {
  const r = regionsState();
  const rendering = !!t?.active && t.kind === "render";
  $("regionsOn").checked = r.on;
  $("regionsMore").hidden = !r.on;
  $("regionsOff").hidden = r.on;
  const btn = $("regionsBtn");
  btn.disabled = !!regionMode || eraseMode || (!!t?.active && (!rendering || !t.total));
  btn.textContent = rendering ? "Pause and edit regions" : "Edit regions";
  if (document.activeElement !== $("regionStrength")) $("regionStrength").value = r.strength;
  const range = document.querySelector('[data-for="regionStrength"]');
  range.value = r.strength;
  paintRange(range);
  const boxes = r.boxes.filter((b) => b.text.trim());
  const key = JSON.stringify(boxes.map((b) => b.text));
  const list = $("regionList");
  if (list.dataset.key !== key) {
    list.dataset.key = key;
    list.replaceChildren(...(boxes.length ? boxes.map((b, i) => {
      const item = document.createElement("li");
      item.style.setProperty("--region", REGION_COLORS[i % REGION_COLORS.length]);
      item.textContent = b.text;
      item.title = b.text;
      return item;
    }) : [Object.assign(document.createElement("li"), { className: "none", textContent: "No boxes yet. Edit regions draws them on the frame." })]));
  }
  const sum = $("sum-regions");
  if (sum) sum.textContent = !r.on ? "" : boxes.length === 1 ? "1 box" : `${boxes.length} boxes`;
}

/* Called with every status: keeps the editor in step with the render it paused. */
function showRegionsTask(t) {
  const running = !!t?.active && t.kind === "render";
  if (!regionMode && !eraseMode && running && t.state === "paused" && t.pause_for !== "erase") openRegions();  // paused from another tab, or before a reload
  // The editor is for a paused render, or for setting boxes before one. A render that ends, or one that starts
  // from elsewhere while the boxes are being set, closes it so the preview shows.
  if ((regionMode === "run" && !running) || (regionMode === "set" && running)) closeRegions();
  if (regionMode === "run") {
    const held = t.state === "paused" || t.pausing;
    if (regionHeld && !held) {  // carried on from another tab or a script: this editor's pause is over
      closeRegions();
    } else {
      regionHeld ||= held;
      showRegionBar(t);
    }
  }
  showRegionsButton(t);
}

/* Whether boxes can be drawn now: always before a render, and during one only once it has paused, as with Fix. */
function regionsReady() {
  return regionMode === "set" || lastStatus?.task?.state === "paused";
}

function openRegions() {
  if (regionMode || eraseMode) return;
  const t = lastStatus?.task;
  regionMode = t?.active && t.kind === "render" ? "run" : "set";
  regionHeld = false;
  if (regionMode === "run" && t.state !== "paused") post("/api/pause").catch((e) => { formError(e.message); closeRegions(); });
  regionDraft = { boxes: structuredClone(regionsState().boxes) };
  regionPicked = -1;
  document.querySelector(".canvas").classList.add("editing");
  $("regionEditor").hidden = false;
  $("stageEmpty").hidden = true;
  showRegionBar(t);
  drawRegions();
  requestAnimationFrame(placeRegionFrame);
}

function closeRegions() {
  regionMode = null;
  regionDraft = null;
  $("regionEditor").hidden = true;
  document.querySelector(".canvas").classList.remove("editing");
  $("stageEmpty").hidden = !($("stageImg").hidden && $("stageVideo").hidden);
  showRegionsButton(lastStatus?.task);
}

/* Keeps the editor's boxes as the regions. Boxes with no prompt are left out. */
function keepRegions() {
  regionsState().boxes = regionDraft.boxes.filter((b) => b.text.trim()).map((b) => ({ ...b, text: b.text.trim() }));
  save();
  refreshPlan();
}

async function regionsDone() {
  keepRegions();
  const run = regionMode === "run";
  closeRegions();
  if (run) {
    try { await post("/api/resume", { regions: regionsFor() }); } catch (e) { formError(e.message); }
  }
}

async function regionsCancel() {
  const run = regionMode === "run";
  closeRegions();
  if (run) {
    try { await post("/api/resume", {}); } catch (e) { formError(e.message); }  // carries on with the regions it had
  }
}

/* Until the render has paused, the frame says so and the boxes, Clear and Continue are off. */
function showRegionBar(t) {
  const run = regionMode === "run";
  const ready = regionsReady();
  if (!ready && document.activeElement?.closest?.("#regionFrame")) document.activeElement.blur();
  $("regionFrame").classList.toggle("waiting", !ready);
  $("regionWait").hidden = ready;
  $("regionDone").disabled = $("regionClear").disabled = !ready;
  $("regionDone").textContent = run ? "Continue" : "Done";
  $("regionCancel").textContent = run ? "Cancel and continue" : "Cancel";
  let note = "Drag on the frame to draw a box, then type its prompt.";
  if (run) note = ready ? `Step ${t.step} of ${t.total}. ${note}` : "Pausing after this step…";
  $("regionNote").textContent = note;
}

/* Lays the frame over the preview when it shows this frame's shape, or on an empty frame of that shape. */
function placeRegionFrame() {
  if (!regionMode) return;
  const canvas = document.querySelector(".canvas");
  const frame = $("regionFrame");
  const img = $("stageImg");
  const area = canvas.getBoundingClientRect();
  // With Tiles, the boxes lie over the whole tiled picture, whose shape differs a little from the frame's.
  const aspect = tileGrid() > 1 ? finalSize(state.width, state.height)[0] / finalSize(state.width, state.height)[1] : state.width / state.height;
  const shown = !img.hidden && img.naturalWidth && img.complete;
  const matches = shown && (regionMode === "run" || Math.abs(img.naturalWidth / img.naturalHeight / aspect - 1) < 0.02);
  let box;
  if (matches) {
    box = img.getBoundingClientRect();
  } else {
    const style = getComputedStyle(canvas);
    const room = { w: area.width - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
                   h: area.height - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom) };
    const w = Math.min(room.w, room.h * aspect);
    const h = w / aspect;
    const left = area.left + parseFloat(style.paddingLeft) + (room.w - w) / 2;
    const top = area.top + parseFloat(style.paddingTop) + (room.h - h) / 2;
    box = { left, top, width: w, height: h };
  }
  frame.classList.toggle("blank", !matches);
  Object.assign(frame.style, { left: `${box.left - area.left}px`, top: `${box.top - area.top}px`, width: `${box.width}px`, height: `${box.height}px` });
}

function drawRegions() {
  const frame = $("regionFrame");
  frame.replaceChildren($("regionWait"), ...regionDraft.boxes.map((b, i) => {
    const el = document.createElement("div");
    el.className = "region-box" + (i === regionPicked ? " picked" : "");
    el.dataset.index = i;
    el.style.setProperty("--region", REGION_COLORS[i % REGION_COLORS.length]);
    const text = document.createElement("textarea");
    text.className = "region-text";
    text.rows = 1;
    text.placeholder = "Prompt for this box";
    text.value = b.text;
    text.addEventListener("input", () => { b.text = text.value; });
    text.addEventListener("focus", () => pickRegion(i));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "region-del";
    del.title = "Remove this box";
    del.textContent = "×";
    del.addEventListener("click", () => { regionDraft.boxes.splice(i, 1); regionPicked = -1; drawRegions(); });
    const grip = document.createElement("div");
    grip.className = "region-grip";
    grip.title = "Drag to resize";
    el.append(text, del, grip);
    sizeRegionBox(el, b);
    return el;
  }));
}

function sizeRegionBox(el, b) {
  Object.assign(el.style, { left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` });
}

function pickRegion(i) {
  if (regionPicked === i) return;
  regionPicked = i;
  $("regionFrame").querySelectorAll(".region-box").forEach((el) => el.classList.toggle("picked", Number(el.dataset.index) === i));
}

const clamp01 = (v) => Math.min(Math.max(v, 0), 1);

/* One drag on the frame: draws a new box from empty space, moves a box by its body, or resizes it by its corner. */
function startRegionDrag(e) {
  if (e.button !== 0 || !regionsReady() || e.target.closest(".region-text, .region-del")) return;
  const frame = $("regionFrame");
  const rect = frame.getBoundingClientRect();
  const at = (ev) => ({ x: clamp01((ev.clientX - rect.left) / rect.width), y: clamp01((ev.clientY - rect.top) / rect.height) });
  const start = at(e);
  const boxEl = e.target.closest(".region-box");
  let index, kind, from;
  if (boxEl) {
    index = Number(boxEl.dataset.index);
    kind = e.target.classList.contains("region-grip") ? "size" : "move";
    from = { ...regionDraft.boxes[index] };
  } else {
    regionDraft.boxes.push({ x: start.x, y: start.y, w: 0, h: 0, text: "" });
    index = regionDraft.boxes.length - 1;
    kind = "draw";
    drawRegions();
  }
  pickRegion(index);
  const b = regionDraft.boxes[index];
  const el = frame.querySelector(`.region-box[data-index="${index}"]`);
  e.preventDefault();
  try { frame.setPointerCapture(e.pointerId); } catch { /* a pointer that is already gone */ }
  const move = (ev) => {
    const p = at(ev);
    if (kind === "draw") {
      Object.assign(b, { x: Math.min(start.x, p.x), y: Math.min(start.y, p.y), w: Math.abs(p.x - start.x), h: Math.abs(p.y - start.y) });
    } else if (kind === "move") {
      b.x = clamp01(Math.min(from.x + p.x - start.x, 1 - from.w));
      b.y = clamp01(Math.min(from.y + p.y - start.y, 1 - from.h));
    } else {
      b.w = Math.max(p.x - from.x, REGION_MIN);
      b.h = Math.max(p.y - from.y, REGION_MIN);
    }
    sizeRegionBox(el, b);
  };
  const end = () => {
    frame.removeEventListener("pointermove", move);
    frame.removeEventListener("pointerup", end);
    frame.removeEventListener("pointercancel", end);
    if (kind === "draw") {
      if (b.w < REGION_MIN || b.h < REGION_MIN) {
        regionDraft.boxes.splice(index, 1);
        regionPicked = -1;
        drawRegions();
      } else {
        el.querySelector(".region-text").focus();
      }
    }
  };
  frame.addEventListener("pointermove", move);
  frame.addEventListener("pointerup", end);
  frame.addEventListener("pointercancel", end);
}

function regionKeys(e) {
  if (!regionMode) return;
  if (e.key === "Escape") { e.preventDefault(); regionsCancel(); }
  const typing = e.target.closest?.("textarea, input");
  if (!typing && (e.key === "Delete" || e.key === "Backspace") && regionPicked >= 0 && regionsReady()) {
    regionDraft.boxes.splice(regionPicked, 1);
    regionPicked = -1;
    drawRegions();
  }
  if (typing && e.key === "Enter" && !e.shiftKey && e.target.classList.contains("region-text")) {
    e.preventDefault();
    e.target.blur();
  }
}

/* Render while the editor is open before a render keeps its boxes first. */
function regionsBeforeRender() {
  if (regionMode !== "set") return;
  keepRegions();
  closeRegions();
}

function setUpRegions() {
  regionsState();
  $("regionsBtn").addEventListener("click", openRegions);
  $("regionsOn").addEventListener("change", (e) => {
    e.stopPropagation();
    regionsState().on = e.target.checked;
    if (e.target.checked) openCard("regions");
    save();
    refreshPlan();
    showRegionsButton(lastStatus?.task);
    if (e.target.checked && !state.regions.boxes.length && !lastStatus?.task?.active) openRegions();  // nothing to use yet
  });
  $("regionStrength").addEventListener("change", (e) => {
    e.stopPropagation();
    const value = Math.min(Math.max(Number(e.target.value) || 0, 0), 1);
    regionsState().strength = value;
    save();
    refreshPlan();
    showRegionsButton(lastStatus?.task);
  });
  $("regionDone").addEventListener("click", () => { if (regionsReady()) regionsDone(); });
  $("regionCancel").addEventListener("click", regionsCancel);
  $("regionClear").addEventListener("click", () => { if (!regionsReady()) return; regionDraft.boxes = []; regionPicked = -1; drawRegions(); });
  $("regionFrame").addEventListener("pointerdown", startRegionDrag);
  document.addEventListener("keydown", regionKeys);
  $("stageImg").addEventListener("load", placeRegionFrame);
  new ResizeObserver(placeRegionFrame).observe(document.querySelector(".canvas"));
  showRegionsButton(lastStatus?.task);
}
