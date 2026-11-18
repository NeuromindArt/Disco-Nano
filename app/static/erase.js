/* Fix: erase something by painting over it, have a model of your own inpaint it, or clone one part of the picture
   over another.

   During a render it pauses, you paint over the preview, and the render fixes what you painted as it carries on. Erase
   has LaMa fill the area and the steps left repaint it, so it can be done again when the prompt grows something back.
   Inpaint has the Inpaint model in Settings repaint the area, from LaMa's fill, from noise or from what's there, with
   an optional prompt. Clone copies from a spot you Alt-click, like a clone stamp. Inpaint and Clone keep the picture's
   own detail, so they are the ones to use late in a render, when the steps left can't add detail to an erased area.
   All three can be used in one pause: the erase is done first, then the inpaint, and the clones go over them. It is
   for stills: in a clip, what it fixes would show until the frame it's fixed on.

   On the start image it works before a render, so the render never has the thing to bring back. The start image
   shows on the stage, and each Apply saves a new image beside it that becomes the start image. */
let eraseMode = null;     // "render" while the editor is over a render it paused, "image" over the start image
let eraseHeld = false;    // the render has reached its pause, or been asked for one
let erasePainted = false;
let eraseBusy = false;    // the start image is being fixed
let fixImage = "";        // the start image the editor is over, so a new one dropped meanwhile closes it
const eraseUndo = [];     // each stroke, with what it changed as it was before, for Undo
const ERASE_UNDO = 20;
const ERASE_BRUSH = { min: 0.5, max: 25, start: 6, step: 0.5 };  // the brush's width, as a share of the picture's width in %
// The share of fresh noise a fixed area gets is all of it up to `from` of the schedule, none from `to`, and falls in a
// straight line between: erase.py's FRESH_FROM and FRESH_TO, which explain why. From `to`, an erased area no longer
// goes back to be drawn again, so it comes out soft, and the page suggests Clone.
const ERASE_FRESH = { from: 0.7, to: 0.8 };
const FIX_LATE_SEEN = "nano.fixLateSeen";
let eraseFreshStep = null;  // the paused step the Noise slider was set for, so a drag isn't undone by the next status

let fixTool = "erase";
// One layer per source picked: how far right and down the source is from the brush, in the picture's pixels, and a
// mask of how much was cloned where. Every layer copies from the picture as it was when the pause began.
const cloneLayers = [];
let cloneSource = null;   // where the last Alt-click was, in the picture's pixels
let cloneOffset = null;   // from the brush to the source, fixed by the first stroke after a new source, then kept
let cloneNewLayer = true; // the next clone stroke starts a layer of its own
const clonePicture = document.createElement("canvas");  // the picture the clones copy from
let clonePictureTaken = false;
const cloneStampCanvas = document.createElement("canvas");

function eraseCanvas() { return $("eraseCanvas"); }
function cloneCanvas() { return $("cloneCanvas"); }
function inpaintCanvas() { return $("inpaintCanvas"); }
function inpaintModel() { return lookValue("inpaint_file"); }
const INPAINT_START = { mode: "fresh", strength: 85, tries: 2 };  // what Inpaint starts from, how much it repaints, how many tries
// Inpaint's tries, during a pause or on the start image: {id, key, made: [url], painting, error}, or null. `key` is
// what was asked, so a change to the area, the prompt or the settings shows the tries are out of date. `tryShown` is
// the try on the stage, or null for the picture without it.
let inpaintTries = null;
let tryShown = null;
let inpaintEdits = 0;  // counts changes to the inpaint area, for `key`
let otherEdits = 0;    // and to the erase and clones, which a start image's tries include

function erasable(t) {
  return !!t?.active && t.kind === "render" && !animating();
}

/* Whether the brush can paint now: a render has to have paused, and the start image can't be mid-fix. */
function eraseReady() {
  return eraseMode === "image" ? !eraseBusy : lastStatus?.task?.state === "paused";
}

/* The editor as the stage knows it: over a render it paused, or over the start image. */
function fixEditor(mode) {
  return {
    name: "fix", followsRender: mode === "render", picture: mode === "image" ? state.init_url : null,
    keys: eraseKeys, beforeRender: fixBeforeRender,
  };
}

/* The picture the editor fixes, without a try on it: the paused preview, or the start image. */
function fixPicture() {
  return eraseMode === "image" ? state.init_url : stage.shows.src;
}

function openErase() {
  const t = lastStatus?.task;
  if (!erasable(t) || !openStageEditor(fixEditor("render"))) return;
  eraseMode = "render";
  eraseHeld = false;
  if (t.state !== "paused") post("/api/pause?why=erase").catch((e) => { formError(e.message); closeErase(); });
  showEraseEditor();
  sizeEraseCanvas();
}

/* The start image on the stage, with the brush over it. */
function openImageErase() {
  if (!state.init_url || lastStatus?.task?.active || !openStageEditor(fixEditor("image"))) return;
  eraseMode = "image";
  eraseBusy = false;
  fixImage = state.init_image;
  showEraseEditor();
  const img = $("stageImg");
  if (img.complete && img.naturalWidth) sizeEraseCanvas();  // otherwise it is sized when it loads
}

function showEraseEditor() {
  dropTries();
  erasePainted = false;
  eraseUndo.length = 0;
  eraseFreshStep = null;
  $("eraseEditor").hidden = false;
  setFixTool(fixTool);
  requestAnimationFrame(placeEraseFrame);
  showEraseButton(lastStatus?.task);
}

function closeErase() {
  if (!eraseMode) return;
  const was = eraseMode;
  dropTries();
  if (was === "image") post("/api/image-tries/close", {}).catch(() => {});  // the Inpaint model can go; it ends by itself too
  eraseMode = null;
  $("eraseEditor").hidden = true;
  $("eraseRing").hidden = true;
  $("cloneMark").hidden = true;
  closeStageEditor("fix");
  showEraseButton(lastStatus?.task);
}

/* Render, with the start image open in Fix: what's painted and not applied would be lost, so it waits for that. */
function fixBeforeRender() {
  if (eraseMode !== "image") return "";
  if (eraseBusy) return "The start image is still being fixed.";
  if (erasePainted) return "Apply or clear what's painted on the start image first.";
  closeErase();
  return "";
}

/* The painting is kept at the picture's own size, so the masks line up with it whatever the window is.
   `clear` wipes it; otherwise it is only reset when the picture's size changed. Either way the clones copy from the
   picture on the stage now. */
function sizeEraseCanvas(clear = true) {
  const img = $("stageImg");
  const canvas = eraseCanvas();
  const w = img.naturalWidth || state.width, h = img.naturalHeight || state.height;
  clonePictureTaken = false;
  if (!clear && canvas.width === w && canvas.height === h) return;
  if (canvas.width !== w || canvas.height !== h) cloneSource = cloneOffset = null;
  for (const c of [canvas, cloneCanvas(), inpaintCanvas()]) {
    c.width = w;
    c.height = h;
    c.getContext("2d").clearRect(0, 0, w, h);
  }
  cloneLayers.length = 0;
  cloneNewLayer = true;
  erasePainted = false;
  eraseUndo.length = 0;
  inpaintEdits++;
  otherEdits++;
  showEraseBar(lastStatus?.task);
}

function placeEraseFrame() {
  if (!eraseMode) return;
  const area = document.querySelector(".canvas").getBoundingClientRect();
  const img = $("stageImg");
  const box = img.hidden ? area : img.getBoundingClientRect();
  Object.assign($("eraseFrame").style, { left: `${box.left - area.left}px`, top: `${box.top - area.top}px`, width: `${box.width}px`, height: `${box.height}px` });
  placeCloneMark();
}

function setFixTool(tool) {
  if (tool === "inpaint" && !inpaintModel()) tool = "erase";
  fixTool = tool;
  $("fixErase").setAttribute("aria-pressed", String(tool === "erase"));
  $("fixClone").setAttribute("aria-pressed", String(tool === "clone"));
  $("fixInpaint").setAttribute("aria-pressed", String(tool === "inpaint"));
  $("inpaintOptions").hidden = tool !== "inpaint";
  $("eraseFrame").classList.toggle("cloning", tool === "clone");
  placeCloneMark();
  showEraseBar(lastStatus?.task);
}

function brushWidth() {
  return (Number($("eraseBrush").value) / 100) * eraseCanvas().width;
}

/* Grows or shrinks the brush by `steps` of the slider. The [ and ] keys and the mouse wheel do this too. */
function sizeBrush(steps) {
  const brush = $("eraseBrush");
  brush.value = Math.min(Math.max(Number(brush.value) + steps * ERASE_BRUSH.step, ERASE_BRUSH.min), ERASE_BRUSH.max);
  showBrushSize();
}

function showBrushSize() {
  $("eraseBrushSize").textContent = `${Math.round(brushWidth())} px`;
  const ring = $("eraseRing");
  if (!ring.hidden && ring.dataset.x) placeEraseRing(Number(ring.dataset.x), Number(ring.dataset.y));
}

/* One stroke of the brush, from where the pointer went down to where it comes up. With Clone, an Alt-click picks
   where to copy from instead. */
function startErasePaint(e) {
  if (e.button !== 0 || !eraseMode || !eraseReady()) return;
  const canvas = eraseCanvas();
  const rect = canvas.getBoundingClientRect();
  const at = (ev) => [((ev.clientX - rect.left) / rect.width) * canvas.width, ((ev.clientY - rect.top) / rect.height) * canvas.height];
  e.preventDefault();
  showTry(null);  // painting is on the picture as it paused
  if (fixTool === "clone") {
    if (e.altKey) {
      const [x, y] = at(e);
      cloneSource = { x, y };
      cloneOffset = null;
      cloneNewLayer = true;
      placeCloneMark();
      showEraseBar(lastStatus?.task);
      return;
    }
    if (!cloneSource) {
      $("eraseNote").textContent = "Alt-click where to copy from first.";
      return;
    }
  }
  const paint = fixTool === "clone" ? cloneStroke(at(e)) : eraseStroke(at(e), fixTool === "inpaint" ? inpaintCanvas() : eraseCanvas());
  erasePainted = true;
  showEraseBar(lastStatus?.task);
  followPointer(canvas, e, (ev) => paint(at(ev)));
}

function keepUndo(record) {
  eraseUndo.push(record);
  if (eraseUndo.length > ERASE_UNDO) eraseUndo.shift();
}

/* An erase or inpaint stroke starting at `start`, on that tool's layer: solid on a see-through layer, so overlaps
   stay even, red to erase and violet to inpaint. Returns what paints the stroke on to each next point. */
function eraseStroke(start, canvas) {
  const ctx = canvas.getContext("2d");
  const color = canvas === inpaintCanvas() ? "#a855f7" : "#ff3b3b";
  if (canvas === inpaintCanvas()) inpaintEdits++; else otherEdits++;
  keepUndo({ canvas, before: ctx.getImageData(0, 0, canvas.width, canvas.height) });
  Object.assign(ctx, { strokeStyle: color, fillStyle: color, lineCap: "round", lineJoin: "round", lineWidth: brushWidth() });
  ctx.beginPath();
  ctx.arc(start[0], start[1], ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fill();
  let last = start;
  return (p) => {
    ctx.beginPath();
    ctx.moveTo(...last);
    ctx.lineTo(...p);
    ctx.stroke();
    last = p;
  };
}

/* A clone stroke starting at `start`: soft dabs of the picture from the source, shown as they will look, and added
   to the layer's mask. The first stroke after a new source sets how far away it copies from, and the strokes after
   it keep that, so the source moves with the brush. */
function cloneStroke(start) {
  if (!clonePictureTaken) {
    const img = $("stageImg");
    clonePicture.width = eraseCanvas().width;
    clonePicture.height = eraseCanvas().height;
    clonePicture.getContext("2d").drawImage(img, 0, 0, clonePicture.width, clonePicture.height);
    clonePictureTaken = true;
  }
  if (!cloneOffset) cloneOffset = { dx: Math.round(cloneSource.x - start[0]), dy: Math.round(cloneSource.y - start[1]) };
  otherEdits++;
  const shown = cloneCanvas().getContext("2d");
  const record = { shown: shown.getImageData(0, 0, cloneCanvas().width, cloneCanvas().height) };
  if (cloneNewLayer || !cloneLayers.length) {
    const mask = document.createElement("canvas");
    mask.width = eraseCanvas().width;
    mask.height = eraseCanvas().height;
    cloneLayers.push({ ...cloneOffset, mask });
    cloneNewLayer = false;
    record.added = true;
  } else {
    const mask = cloneLayers[cloneLayers.length - 1].mask;
    record.mask = mask.getContext("2d").getImageData(0, 0, mask.width, mask.height);
  }
  keepUndo(record);
  const layer = cloneLayers[cloneLayers.length - 1];
  const radius = brushWidth() / 2;
  const spacing = Math.max(radius / 4, 1);
  let last = start;
  cloneDab(layer, start, radius);
  return (p) => {
    const length = Math.hypot(p[0] - last[0], p[1] - last[1]);
    const dabs = Math.floor(length / spacing);
    for (let k = 1; k <= dabs; k++) {
      const f = (k * spacing) / length;
      cloneDab(layer, [last[0] + (p[0] - last[0]) * f, last[1] + (p[1] - last[1]) * f], radius);
    }
    if (dabs) {
      const f = (dabs * spacing) / length;
      last = [last[0] + (p[0] - last[0]) * f, last[1] + (p[1] - last[1]) * f];
    }
    placeCloneMark(p);
  };
}

/* A soft round brush: solid in the middle, fading to nothing at its edge. */
function softBrush(ctx, x, y, radius) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, radius);
  g.addColorStop(0, "rgb(255 255 255 / 1)");
  g.addColorStop(0.5, "rgb(255 255 255 / 1)");
  g.addColorStop(1, "rgb(255 255 255 / 0)");
  return g;
}

/* One dab of the clone brush at `p`. */
function cloneDab(layer, p, radius) {
  const size = Math.ceil(radius * 2) + 2;
  const left = Math.round(p[0] - size / 2), top = Math.round(p[1] - size / 2);
  const stamp = cloneStampCanvas;
  if (stamp.width !== size) stamp.width = stamp.height = size;
  const s = stamp.getContext("2d");
  s.globalCompositeOperation = "copy";
  s.drawImage(clonePicture, left + layer.dx, top + layer.dy, size, size, 0, 0, size, size);
  s.globalCompositeOperation = "destination-in";
  s.fillStyle = softBrush(s, p[0] - left, p[1] - top, radius);
  s.fillRect(0, 0, size, size);
  cloneCanvas().getContext("2d").drawImage(stamp, left, top);
  const m = layer.mask.getContext("2d");
  m.fillStyle = softBrush(m, p[0], p[1], radius);
  m.fillRect(left, top, size, size);
}

function undoErasePaint() {
  const stroke = eraseUndo.pop();
  if (!stroke) return;
  showTry(null);
  if (stroke.canvas === inpaintCanvas()) inpaintEdits++; else otherEdits++;
  if (stroke.canvas) stroke.canvas.getContext("2d").putImageData(stroke.before, 0, 0);
  else {
    cloneCanvas().getContext("2d").putImageData(stroke.shown, 0, 0);
    const layer = cloneLayers[cloneLayers.length - 1];
    if (stroke.added) {
      cloneLayers.pop();
      cloneNewLayer = true;  // the next stroke starts the layer again, from the same source
    } else layer.mask.getContext("2d").putImageData(stroke.mask, 0, 0);
  }
  erasePainted = cloneLayers.length > 0 || paintedOn(eraseCanvas()) || paintedOn(inpaintCanvas());
  showEraseBar(lastStatus?.task);
}

function paintedOn(canvas) {
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) if (data[i]) return true;
  return false;
}

/* The ring that shows the brush's size where it would paint. */
function placeEraseRing(x, y) {
  const ring = $("eraseRing");
  const canvas = eraseCanvas();
  const size = (brushWidth() / canvas.width) * canvas.getBoundingClientRect().width;
  const frame = $("eraseFrame").getBoundingClientRect();
  Object.assign(ring.style, { width: `${size}px`, height: `${size}px`, left: `${x - frame.left - size / 2}px`, top: `${y - frame.top - size / 2}px` });
  Object.assign(ring.dataset, { x, y });
  ring.hidden = false;
  if (cloneOffset) {  // the source follows the brush
    const rect = canvas.getBoundingClientRect();
    placeCloneMark([((x - rect.left) / rect.width) * canvas.width, ((y - rect.top) / rect.height) * canvas.height]);
  }
}

/* The crosshair on the clone source: where the brush at `brush` copies from, or the picked spot before a stroke. */
function placeCloneMark(brush) {
  const mark = $("cloneMark");
  const at = brush && cloneOffset ? { x: brush[0] + cloneOffset.dx, y: brush[1] + cloneOffset.dy } : !cloneOffset ? cloneSource : null;
  if (!eraseMode || fixTool !== "clone" || !cloneSource) { mark.hidden = true; return; }
  if (!at) return;  // keeps its place until the pointer moves
  const canvas = eraseCanvas();
  const scale = canvas.getBoundingClientRect().width / canvas.width;
  Object.assign(mark.style, { left: `${at.x * scale}px`, top: `${at.y * scale}px` });
  mark.hidden = false;
}

/* A painted layer as a mask: white where it is, black elsewhere, with its soft edge as greys. */
function maskOf(canvas, solid) {
  const mask = document.createElement("canvas");
  mask.width = canvas.width;
  mask.height = canvas.height;
  const m = mask.getContext("2d");
  m.fillStyle = "#000";
  m.fillRect(0, 0, mask.width, mask.height);
  if (solid) {  // the erase and inpaint layers are one color: white wherever they were painted
    const white = document.createElement("canvas");
    white.width = canvas.width;
    white.height = canvas.height;
    const w = white.getContext("2d");
    w.drawImage(canvas, 0, 0);
    w.globalCompositeOperation = "source-in";
    w.fillStyle = "#fff";
    w.fillRect(0, 0, white.width, white.height);
    m.drawImage(white, 0, 0);
  } else m.drawImage(canvas, 0, 0);
  return mask.toDataURL("image/png");
}

/* What was painted, as /api/erase and /api/erase-image take it. Inpaint goes with the model and settings Settings
   has now, so a model chosen during the pause is the one that paints. */
function fixBody() {
  const canvas = eraseCanvas();
  const inpainted = inpaintModel() && paintedOn(inpaintCanvas());
  return {
    mask: paintedOn(canvas) ? maskOf(canvas, true) : null,
    clones: cloneLayers.map((l) => ({ mask: maskOf(l.mask, false), dx: l.dx / canvas.width, dy: l.dy / canvas.height })),
    inpaint: inpainted ? {
      mask: maskOf(inpaintCanvas(), true), prompt: $("inpaintPrompt").value, mode: $("inpaintMode").value,
      strength: $("inpaintMode").value === "fresh" ? 1 : Number($("inpaintStrength").value) / 100,
      file: inpaintModel(), steps: Number(lookValue("inpaint_steps")), guidance: Number(lookValue("inpaint_guidance")),
      context: Number(lookValue("inpaint_context")), grow: Number(lookValue("inpaint_grow")), blend: Number(lookValue("inpaint_blend")),
      ...(triesCurrent() && tryShown !== null ? { chosen: { id: inpaintTries.id, k: tryShown } } : {}),
    } : null,
  };
}

// ---- inpaint tries ----
/* What an ask for tries depends on, so tries painted for another area, prompt or setting show as out of date. */
function askKey() {
  return JSON.stringify([inpaintEdits, $("inpaintPrompt").value, $("inpaintMode").value, $("inpaintStrength").value,
    ...["inpaint_file", "inpaint_steps", "inpaint_guidance", "inpaint_context", "inpaint_grow", "inpaint_blend"].map(lookValue),
    ...(eraseMode === "image" ? [otherEdits, state.init_image] : [])]);  // a start image's tries are the whole fix
}

/* Tries come for an inpaint painted now, during a paused render or on the start image. */
function usesTries() {
  return !!eraseMode && !!inpaintModel() && paintedOn(inpaintCanvas());
}

function triesCurrent() {
  return !!inpaintTries && inpaintTries.key === askKey();
}

/* Puts try `k` on the stage in place of the picture it was painted on, or that picture back for null. */
function showTry(k) {
  if (k !== null && !inpaintTries?.made[k]) k = null;
  if (k !== null || tryShown !== null) showEditorPicture(k === null ? fixPicture() : inpaintTries.made[k]);
  tryShown = k;
  $("eraseFrame").classList.toggle("trying", k !== null);
  $("eraseFrame").classList.toggle("whole", eraseMode === "image");  // a start image's try has the erase and clones in it
  showTryRow();
  showEraseBar(lastStatus?.task);
}

function dropTries() {
  if (tryShown !== null) showEditorPicture(fixPicture());
  inpaintTries = null;
  tryShown = null;
  $("eraseFrame").classList.remove("trying");
  showTryRow();
}

/* A start image's tries are the whole fix, erase and clones too, so the one chosen becomes the new start image. */
async function askTries(more) {
  const body = fixBody();
  if (!body.inpaint) return;
  delete body.inpaint.chosen;
  const count = Number($("inpaintCount").value);
  more = !!more && triesCurrent();
  if (!more) showTry(null);
  try {
    const r = eraseMode === "image" ? await post("/api/image-tries", { path: state.init_image, ...body, count, more })
      : await post("/api/inpaint-tries", { inpaint: body.inpaint, count, more });
    inpaintTries = more && r.id === inpaintTries.id ? { ...inpaintTries, painting: true, error: null }
      : { id: r.id, key: askKey(), made: [], painting: true, error: null, auto: true };
  } catch (e) { formError(e.message); }
  showTryRow();
  showEraseBar(lastStatus?.task);
}

/* The tries in a status, as they come: the render's own, or the start image's. The first one goes on the stage by
   itself. */
function takeTries(sent) {
  if (!inpaintTries || !sent || sent.id !== inpaintTries.id) return;
  const before = inpaintTries.made.length;
  Object.assign(inpaintTries, { made: sent.made, painting: sent.painting, error: sent.error, message: sent.message || "" });
  if (sent.error && sent.error !== inpaintTries.shownError) { inpaintTries.shownError = sent.error; formError(sent.error); }
  if (inpaintTries.auto && before === 0 && sent.made.length) { inpaintTries.auto = false; showTry(0); }
  else if (sent.made.length !== before || !sent.painting) showTryRow();
}

function tryTile(label, src, k) {
  const tile = document.createElement("button");
  tile.type = "button";
  tile.className = "try-tile";
  tile.setAttribute("role", "option");
  tile.setAttribute("aria-selected", String(tryShown === k));
  tile.title = k !== null ? `Use try ${k + 1}` : eraseMode === "image" ? "The start image, without the inpaint" : "The picture as it paused, without the inpaint";
  if (src) {
    const img = document.createElement("img");
    img.src = src;
    img.alt = "";
    tile.append(img);
  }
  const name = document.createElement("span");
  name.className = "try-label";
  name.textContent = label;
  tile.append(name);
  tile.addEventListener("click", () => { if (inpaintTries) inpaintTries.auto = false; showTry(k); });
  return tile;
}

function showTryRow() {
  const shown = !!inpaintTries && !!eraseMode;
  $("inpaintTryRow").hidden = !shown;
  if (!shown) return;
  const tiles = [tryTile("Before", fixPicture(), null)];
  inpaintTries.made.forEach((url, k) => tiles.push(tryTile(`Try ${k + 1}`, url, k)));
  if (inpaintTries.painting) {
    const wait = document.createElement("div");
    wait.className = "try-tile painting";
    wait.title = "Painting a try";
    wait.innerHTML = '<span class="spinner" aria-hidden="true"></span>';
    tiles.push(wait);
  }
  $("inpaintTryList").replaceChildren(...tiles);
  $("inpaintMore").disabled = inpaintTries.painting || !triesCurrent();
}

async function eraseDone() {
  if (usesTries() && !triesCurrent()) return askTries(false);  // the inpaint is painted first, to choose from
  if (usesTries() && inpaintTries.painting && tryShown === null) return;
  if (eraseMode === "image") return eraseImage();
  const body = erasePainted ? fixBody() : null;
  if (body?.inpaint && usesTries() && tryShown === null) body.inpaint = null;  // Before: carry on without it
  const fresh = Number($("eraseFresh").value) / 100;
  closeErase();
  await endPause(async () => {
    if (body && (body.mask || body.clones.length || body.inpaint)) await post("/api/erase", { ...body, fresh });
    await post("/api/resume", {});
  });
}

/* Fixes the start image, which becomes the new image, and leaves the editor open to fix more. With tries, the one on
   the stage is the fix; Before applies the erase and clones without the inpaint. */
async function eraseImage() {
  if (!erasePainted || eraseBusy) return;
  const chosen = usesTries() && tryShown !== null ? { id: inpaintTries.id, k: tryShown } : null;
  const body = fixBody();
  if (usesTries()) body.inpaint = null;
  if (!chosen && !body.mask && !body.clones.length && !body.inpaint) return;
  eraseBusy = true;
  showEraseBar();
  try {
    const r = chosen ? await post("/api/image-tries/use", chosen) : await post("/api/erase-image", { path: state.init_image, ...body });
    dropTries();
    readControls();
    Object.assign(state, { init_image: r.path, init_name: r.name, init_url: r.url || "", init_size: r.size });
    fixImage = r.path;
    save();
    showInit();
    eraseBusy = false;
    showEditorPicture(r.url);
    sizeEraseCanvas();  // what was painted is part of the new image
  } catch (e) {
    eraseBusy = false;
    formError(e.message);
    showEraseBar();
  }
}

async function eraseCancel() {
  const render = eraseMode === "render";
  closeErase();
  if (render) await endPause(() => post("/api/resume", {}));
}

function lateNoteSeen() {
  try { return localStorage.getItem(FIX_LATE_SEEN) === "1"; } catch { return false; }
}

function dismissLateNote() {
  try { localStorage.setItem(FIX_LATE_SEEN, "1"); } catch { /* shown again next time */ }
  $("fixLate").hidden = true;
}

/* Until a render has paused, or while the start image is being fixed, the frame says so and the brush is off. */
function showEraseBar(t) {
  if (inpaintTries && !inpaintTries.painting && !triesCurrent()) dropTries();  // the area, prompt or settings changed
  const image = eraseMode === "image";
  const ready = eraseReady();
  const cloning = fixTool === "clone";
  const late = !image && ready && t?.at >= ERASE_FRESH.to;
  $("eraseFrame").classList.toggle("waiting", !ready);
  $("eraseWait").hidden = ready;
  $("eraseWaitText").textContent = image ? "Fixing…" : "Pausing after this step…";
  for (const id of ["eraseBrush", "eraseFresh", "eraseClear", "eraseUndo", "eraseDone", "fixErase", "fixClone", "inpaintPrompt", "inpaintMode", "inpaintStrength", "inpaintCount"]) $(id).disabled = !ready;
  $("fixInpaint").disabled = !ready || !inpaintModel();
  $("fixInpaint").title = inpaintModel() ? "Paint over something for the Inpaint model to repaint (I)" : "Set an Inpaint model in Settings to inpaint.";
  if (fixTool === "inpaint" && !inpaintModel()) setFixTool("erase");
  $("eraseFreshBox").hidden = image;
  $("fixLate").hidden = !(late && fixTool === "erase" && !lateNoteSeen());
  if (!image && ready && t && eraseFreshStep !== t.step) {
    eraseFreshStep = t.step;
    $("eraseFresh").value = Math.round(freshShare(t.at ?? 0) * 20) * 5;  // in the slider's steps of 5%
    showFresh();
  }
  const trying = ready && usesTries();
  const painting = trying && triesCurrent() && inpaintTries.painting && tryShown === null;
  // Before, on the start image, applies the erase and clones without the inpaint, so with neither there's nothing to apply
  const without = image && trying && tryShown === null && !cloneLayers.length && !paintedOn(eraseCanvas());
  $("eraseDone").textContent = trying && !triesCurrent() ? "Inpaint"
    : painting ? "Painting…"
    : image ? (trying ? (tryShown === null ? "Apply without it" : "Use this") : "Apply")
    : trying && tryShown === null ? "Continue without it"
    : erasePainted ? (trying ? "Use this and continue" : "Apply and continue") : "Continue";
  $("eraseDone").title = !trying || triesCurrent() ? "" : image ? "Paint the tries, then choose one" : "Paint the tries, then choose one before the render carries on";
  $("eraseDone").disabled = !ready || (image && !erasePainted) || painting || (image && triesCurrent() && without);
  $("eraseCancel").textContent = image ? "Close" : "Cancel and continue";
  let note;
  if (image && eraseBusy) note = "Fixing the start image…";
  else if (!ready) note = "Pausing after this step…";
  else if (cloning) note = cloneSource ? "Paint to copy. Alt-click for a new source." : "Alt-click where to copy from, then paint.";
  else if (trying && triesCurrent() && (inpaintTries.made.length || inpaintTries.painting)) {
    if (inpaintTries.painting && !inpaintTries.made.length) note = image && inpaintTries.message ? `Painting the tries: ${inpaintTries.message}` : "Painting the tries…";
    else note = image ? "Choose a try. Before leaves the inpaint out." : "Choose a try. Before carries on without it.";
  } else if (fixTool === "inpaint") note = "Paint where something should go, and say what in the box.";
  else note = image ? "Paint over what to remove from the start image." : "Paint over what to remove.";
  if (!image && ready && t) note = `Step ${t.step} of ${t.total}. ${note}`;
  $("eraseNote").textContent = note;
  showBrushSize();
}

/* The share of fresh noise that suits a fix `at` this far through the schedule, 0 to 1. */
function freshShare(at) {
  return Math.min(Math.max((ERASE_FRESH.to - at) / (ERASE_FRESH.to - ERASE_FRESH.from), 0), 1);
}

function showFresh() {
  $("eraseFreshShown").textContent = `${$("eraseFresh").value}%`;
}

function showEraseButton(t) {
  if (eraseMode === "image" && state.init_image !== fixImage) closeErase();  // another start image: the painting was for the last one
  const btn = $("eraseBtn");
  btn.hidden = !t?.active || t.kind !== "render" || animating();
  btn.disabled = stageEditor() || !erasable(t) || (t?.state === "paused" && t.pause_for !== "erase");
  btn.title = "Pauses after this step, so you can erase or clone over something. The render then carries on from it.";
  const image = $("initErase");
  image.hidden = !state.init_url || dropBusy($("initDrop"));
  image.disabled = stageEditor() || !!t?.active;
  image.title = t?.active ? "Fix the start image when no render is running." : "Erase or clone over things in the start image";
}

/* Called with every status: keeps the editor in step with the render it paused. */
function showEraseTask(t) {
  const running = !!t?.active && t.kind === "render";
  if (eraseMode === "image") {
    if (t?.active) closeErase();  // a render started: the stage is the render's again
    else if (inpaintTries) {
      takeTries(lastStatus?.image_tries);
      showEraseBar();
    }
    showEraseButton(t);
    return;
  }
  if (heldPause(t) && !eraseMode && t.pause_for === "erase") openErase();  // paused from another tab, or before a reload
  if (eraseMode && !running) closeErase();
  if (eraseMode) {
    const held = t.state === "paused" || t.pausing;
    if (eraseHeld && !held) closeErase();  // carried on from another tab: this editor's pause is over
    else {
      if (!eraseHeld && t.state === "paused") sizeEraseCanvas(false);  // the paused step's preview, at its own size
      eraseHeld ||= held;
      takeTries(t.tries);
      showEraseBar(t);
    }
  }
  showEraseButton(t);
}

function eraseKeys(e) {
  if (e.target.closest?.("input, textarea, select")) return;
  const key = e.key.toLowerCase();
  if (e.key === "Escape") { e.preventDefault(); eraseCancel(); }
  else if (e.key === "[" || e.key === "]") { e.preventDefault(); sizeBrush(e.key === "]" ? 2 : -2); }
  else if ((e.ctrlKey || e.metaKey) && key === "z") { e.preventDefault(); undoErasePaint(); }
  else if (!e.ctrlKey && !e.metaKey && !e.altKey && ["e", "c", "i"].includes(key) && eraseReady()) setFixTool({ e: "erase", c: "clone", i: "inpaint" }[key]);
  else if (e.key === "Alt" && fixTool === "clone") e.preventDefault();  // keeps the browser from taking focus to its menu
}

function setUpErase() {
  Object.assign($("eraseBrush"), { min: ERASE_BRUSH.min, max: ERASE_BRUSH.max, step: ERASE_BRUSH.step, value: ERASE_BRUSH.start });
  $("eraseBrush").addEventListener("input", showBrushSize);
  $("eraseFresh").addEventListener("input", showFresh);
  $("eraseBtn").addEventListener("click", openErase);
  $("initErase").addEventListener("click", (e) => { e.stopPropagation(); openImageErase(); });
  $("eraseDone").addEventListener("click", eraseDone);
  $("eraseCancel").addEventListener("click", eraseCancel);
  $("eraseClear").addEventListener("click", () => sizeEraseCanvas());
  $("eraseUndo").addEventListener("click", undoErasePaint);
  $("fixErase").addEventListener("click", () => setFixTool("erase"));
  $("fixClone").addEventListener("click", () => setFixTool("clone"));
  $("fixInpaint").addEventListener("click", () => setFixTool("inpaint"));
  // Inpaint's start and strength stay as last set, in this browser.
  const start = { ...INPAINT_START, ...state.fix_inpaint };
  $("inpaintMode").value = start.mode;
  $("inpaintStrength").value = start.strength;
  $("inpaintCount").value = start.tries;
  const showStart = () => {
    $("inpaintStrengthBox").hidden = $("inpaintMode").value === "fresh";  // Fresh always paints the whole way
    $("inpaintStrengthShown").textContent = `${$("inpaintStrength").value}%`;
    state.fix_inpaint = { mode: $("inpaintMode").value, strength: Number($("inpaintStrength").value), tries: Number($("inpaintCount").value) };
    save();
    if (eraseMode) showEraseBar(lastStatus?.task);
  };
  $("inpaintMode").addEventListener("change", showStart);
  $("inpaintStrength").addEventListener("input", showStart);
  $("inpaintCount").addEventListener("change", showStart);
  $("inpaintPrompt").addEventListener("input", () => { if (eraseMode) showEraseBar(lastStatus?.task); });
  $("inpaintMore").addEventListener("click", () => askTries(true));
  showStart();
  $("fixLateClone").addEventListener("click", () => { dismissLateNote(); setFixTool("clone"); });
  $("fixLateClose").addEventListener("click", dismissLateNote);
  const canvas = eraseCanvas();
  canvas.addEventListener("pointerdown", startErasePaint);
  canvas.addEventListener("pointermove", (e) => placeEraseRing(e.clientX, e.clientY));
  canvas.addEventListener("pointerleave", () => { $("eraseRing").hidden = true; delete $("eraseRing").dataset.x; });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); sizeBrush(e.deltaY < 0 ? 1 : -1); }, { passive: false });
  $("stageImg").addEventListener("load", () => {
    // The start image, once it has loaded: only a new size resets the painting, so strokes made while it loaded stay.
    if (eraseMode === "image" && tryShown === null) sizeEraseCanvas(false);
    placeEraseFrame();
  });
  new ResizeObserver(placeEraseFrame).observe(document.querySelector(".canvas"));
}
