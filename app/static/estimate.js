/* What a render will make and how long it should take: Small first, tiles, the frames of a clip, and the time,
   from speeds measured on an RTX 3070 8GB. */
"use strict";

const EXACT_REPEAT_COST = 1.28;  // measured: 0.84 to 1.08 seconds per step at 512×512
const TIME_TRAVEL_COST = 1.35;  // measured at 512×512 with the default curve
const BORROW_SECONDS = 20;  // measured with Tiny SD at 512×512, loading included
const HANDOFF_SECONDS = 20;  // loading an SD 1.5 checkpoint and its own steps, at 512×512
const HANDOFF_FRAME_SECONDS = 1.8;  // measured: each later clip frame's SD 1.5 steps and swap, at 512×512
const ROUND_SECONDS = 0.25;  // measured: each View round past the first, per step, with the three default checkers

function smallSide() { return Number(lookValue("small_side")) || 768; }

/* Small first's size and steps, or null when it is off or the frame is no larger. As settings.small_frame in
   the renderer: the frame's shape in multiples of 64, its longer side at most Small first size. Of the sizes
   with at least 60% of the area that allows, the largest within 2% of the frame's shape, or else the closest. */
function smallFirstOn() { return Number(state.small_first) > 0 && !state.handoff_file; }

function smallFrame() {
  const share = smallFirstOn() ? Number(state.small_first) : 0, longest = smallSide();
  const long = Math.max(state.width, state.height), short = Math.min(state.width, state.height);
  if (share <= 0 || tileGrid() > 1 || long <= longest) return null;
  const ratio = long / short, found = [], top = 64 * Math.max(Math.floor(longest / 64), 1);
  for (let side = top; side >= 64; side -= 64) {
    const other = Math.max(Math.round(side / ratio / 64), 1) * 64;
    if (side * other >= (0.6 * top * top) / ratio || !found.length) found.push([Math.abs(side / other / ratio - 1), side, other]);
  }
  const close = found.filter((f) => f[0] <= 0.02);
  const [, side, other] = close.length ? close[0] : found.reduce((a, b) => (b[0] < a[0] ? b : a));  // the larger on a tie
  const steps = Math.min(Math.round((state.steps * share) / 100), state.steps - 1);
  return state.width >= state.height ? { width: side, height: other, steps } : { width: other, height: side, steps };
}

/* The handoff model paints the start, finishes, or both. */
function handingOff() {
  return !!state.handoff_file && (Number(state.handoff_share) > 0 || Number(state.handoff_finish) > 0);
}

/* Seconds per step, measured on an RTX 3070 8GB. The memory plan comes from the server. */
function fitFor(mpx, offload) {
  const vitL = state.ViTL14;
  const others = ["ViTB32", "ViTB16", "RN50"].filter((n) => state[n]).length;
  if (state.use_secondary_model) {
    return vitL ? { limit: 0.27, perStep: 0.77 } : { limit: 0.99, perStep: 0.2 + 1.05 * mpx };
  }
  if (offload) {
    return vitL ? { limit: 0.59, perStep: 0.6 + 1.4 * mpx } : { limit: 0.99, perStep: 0.4 + 3.7 * mpx };
  }
  if (vitL) return { limit: 0.27, perStep: 1.0 + 0.08 * others };
  return { limit: 0.59, perStep: 0.1 + 2.7 * mpx };
}

/* Tiles across and down. Each tile is a render at the frame size, so the grid sets the final size. */
function tileGrid() {
  return state.tile_scale && state.tile_scale !== "Off" ? Number(state.tile_scale[0]) : 1;
}

/* The pixels a tile of this side shares with its neighbors, as tiles.overlaps() works it out. */
function tileOverlap(side) {
  const percent = Number(state.advanced.tile_overlap_pct ?? byName.tile_overlap_pct.default);
  return Math.min(Math.max(pyRound(side * percent / 800) * 8, 8), Math.floor(side / 2));
}

/* The final image's size for a frame of this size. */
function finalSize(width, height) {
  const extra = tileGrid() - 1;
  return [width + (width - tileOverlap(width)) * extra, height + (height - tileOverlap(height)) * extra];
}

/* The step Render first's tiles start from, repainting the picture its first pass made. */
function tileSkip() {
  const steps = Math.max(Number(state.steps) || 1, 1);
  return Math.min(pyRound(steps * (1 - Number(state.tile_denoise))), steps - 1);
}

/* Seconds of steps for one picture that Disco paints from step `from`, as the renderer runs it: at the frame
   size with Small first, or once for each tile. `perStep` and `smallPer` are the measured speeds. */
function paintSeconds(from, perStep, smallPer) {
  const steps = Math.max(Number(state.steps) || 1, 1), grid = tileGrid();
  const run = Math.max(steps - from, 0);
  if (grid > 1) return run * grid * grid * perStep;
  const small = smallFrame();
  const smallRun = small ? Math.min(Math.max(small.steps - from, 0), run) : 0;
  const repaint = Number(state.small_denoise) || 0;  // Full-size denoise repaints some small steps at full size
  const redone = smallRun && repaint > 0 ? Math.min(Math.max(small.steps - (steps - pyRound((steps * repaint) / 100)), 0), smallRun) : 0;
  return (run - smallRun + redone) * perStep + smallRun * smallPer;
}

/* Seconds of steps for a still, or a clip's first frame, from where it starts. */
function firstSeconds(perStep, smallPer) {
  const steps = Math.max(Number(state.steps) || 1, 1), grid = tileGrid();
  const video = animating() && state.animation_mode === "Video Input";
  const skip = Math.min(Number(state.skip_steps) || 0, steps - 1);
  const at = (from) => paintSeconds(afterHandoffSkip(from), perStep, smallPer);
  if (video) {
    // Every video frame uses Frame skip; Build up's carried paint first needs a pass of the frame from Denoise.
    return at(frameSkip(steps)) + (Number(state.video_blend) ? at(skip) : 0);
  }
  if (grid > 1 && !state.init_image && state.tile_start === "Render first") {
    const whole = Math.max(steps - afterHandoffSkip(skip), 0) * perStep;  // the frame at tile size, then the tiles
    return whole + paintSeconds(tileSkip(), perStep, smallPer);
  }
  return at(skip);
}

/* Frames a clip paints. With Turbo, a 2D or 3D clip paints every Turbo steps-th frame and the last one. */
function paintedFrames() {
  const frames = Math.max(Number(state.max_frames) || 1, 1);
  const turbo = state.turbo_mode && state.animation_mode !== "Video Input" ? Math.max(Number(state.turbo_steps) || 1, 1) : 1;
  return Math.ceil(frames / turbo) + ((frames - 1) % turbo ? 1 : 0);
}

function showTiles() {
  const grid = tileGrid();
  const renderFirst = state.tile_start === "Render first";
  $("tileMore").hidden = grid === 1;
  const video = animating() && state.animation_mode === "Video Input";
  // Tile denoise is how much Render first's tiles repaint its first pass. A start image or video uses Denoise.
  $("tile_denoise").disabled = grid === 1 || !renderFirst || !!state.init_image || video;
  $("tile_denoise").closest(".slider-row").title = grid > 1 && (state.init_image || video)
    ? worded("A start image or video uses {skip_steps}.") : grid > 1 && !renderFirst ? "Only Render first repaints in tiles from a picture." : "";
  $("tile_start").disabled = !!state.init_image || video;
  // A video's frames all use Frame skip, and Denoise only paints Build up's first pass of the first frame.
  const frameSkipOnly = video && !Number(state.video_blend);
  // Below Handoff 100%, Disco carries on from where the checkpoint stopped, so there's no share left to choose.
  const carryOn = handingStart() && Number(state.handoff_share) < 100;
  for (const id of ["denoise", "skip_steps"]) {
    const row = $(id).closest(".slider-row");
    row.classList.toggle("unused", frameSkipOnly || carryOn);
    row.title = frameSkipOnly ? worded("Every video frame uses {frames_skip_steps}, so {skip_steps} isn't used.")
      : carryOn ? worded("With {handoff_share} below 100, Disco carries on from where the checkpoint stopped, so {skip_steps} isn't used.") : "";
  }
}

/* The frame closest to the start image's shape, and never larger than the frame Match began from,
   which is the size known to fit. With tiles, it is the final image's shape that counts. */
function matchFrame() {
  if (state.init_fit !== "match" || !state.init_size) return;
  const shape = state.init_size[0] / state.init_size[1];
  state.match_area ||= state.width * state.height;
  const area = state.match_area;
  let best = null, lowest = Infinity;
  for (const width of SIZES) {
    for (const height of SIZES) {
      if (width * height > area) continue;
      const [w, h] = finalSize(width, height);
      const score = Math.abs(Math.log(w / h / shape)) + 0.2 * Math.abs(Math.log((width * height) / area));
      if (score < lowest) { lowest = score; best = [width, height]; }
    }
  }
  if (best[0] === state.width && best[1] === state.height) return;
  [state.width, state.height] = best;
  followSize();
}

function updateFit() {
  showTiles();
  showIdle();
  showSummaries();
  showOpts();
  showImagePrompts();
  showBlocked();
  refreshPlan();
  showTime();
}

/* What the critic experiments add to each step, measured at 512×512 against 67 seconds of steps. */
function criticCost() {
  const share = (text) => (parseSchedule(text) || []).reduce((s, l) => s + (l.v > 0 ? l.n : 0), 0) / 1000;
  let cost = 1;
  if (Number(state.layout_critic)) cost *= lookValue("layout_cuts") === "All" ? 1.15 : 1.12;
  if (String(state.robust_critic).includes("L/14")) cost *= 1.7;
  else if (state.robust_critic && state.robust_critic !== "Off") cost *= 1.07;
  if (Number(state.one_subject)) cost *= 1.04;
  if (Number(state.correct_steps)) cost *= 1 + Number(state.correct_steps) * share(lookValue("correct_curve")) * (lookValue("correct_clip") ? 1.25 : 0.6);
  return cost;
}

/* How long it should take, from the measured speeds. Only for the main model, which they were measured on. */
function showTime() {
  const note = $("timeNote");
  if (state.diffusion_model !== "512x512_diffusion_uncond_finetune_008100") { note.textContent = ""; return; }
  const mpx = (state.width * state.height) / 1e6;
  // Each View round past the first runs the checkers again on new cutouts, which are the same size at any frame size.
  const rounds = Math.max((Number(lookValue("cutn_batches")) || 1) - 1, 0) * ROUND_SECONDS;
  const perStep = fitFor(mpx, !!plan?.offload).perStep + rounds;
  const LOADING_SECONDS = 45;
  // Small first paints the start of what is left after a handoff at its own size and speed.
  const small = smallFrame();
  const smallPer = small ? fitFor((small.width * small.height) / 1e6, !!plan?.offload).perStep + rounds : perStep;
  const costs = (state.exact_repeat ? EXACT_REPEAT_COST : 1) * (state.time_travel ? TIME_TRAVEL_COST : 1) * criticCost();
  const seconds = firstSeconds(perStep, smallPer) * costs
    + LOADING_SECONDS + (usesPartner() ? BORROW_SECONDS : 0) + (handingOff() ? HANDOFF_SECONDS : 0);
  if (animating()) {
    const steps = Math.max(Number(state.steps) || 1, 1);
    // A handoff paints the start of every frame. The checkpoint paints at least 512×512 and at most 768×768.
    const handArea = Math.min(Math.max(state.width * state.height, 512 * 512), 768 * 768) / (512 * 512);
    // measured at Handoff 30%; at a higher share the checkpoint paints more of each frame's steps
    const hand = handingStart() ? HANDOFF_FRAME_SECONDS * handArea * Math.max(Number(state.handoff_share) / 30, 1) : 0;
    const perFrame = paintSeconds(afterHandoffSkip(frameSkip(steps)), perStep, smallPer) * costs + hand;
    note.textContent = `About ${fmtTime(perFrame)} a frame, ${fmtTime(seconds + perFrame * (paintedFrames() - 1))} in all`;
  } else {
    note.textContent = `About ${fmtTime(seconds)} an image`;
  }
}
