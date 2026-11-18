/* The settings panel: its controls and cards, sliders, orientation, Image or Animation, and the settings that do
   nothing with the others as they are. */
"use strict";

// Controls in the main panel. Settings shows the rest.
const QUICK = ["steps", "skip_steps", "clip_guidance_scale", "n_batches", "set_seed", "diffusion_model",
  "use_secondary_model", "exact_repeat", "init_scale", "tile_scale", "tile_denoise", "tile_start", "tile_whole_view",
  "time_travel", "prompt_graft", "live_ghost", "depth_target", "two_painters", "borrow_attention", "detail_bands", "shaped_guidance", "gradient_momentum", "evidence_map", "layout_critic", "robust_critic", "one_subject", "correct_steps", "route_prompts", ...CLIPS.map((c) => c[0]),
  "animation_mode", "max_frames", "fps", "in_between", "loop_mode", "frames_skip_steps", "video_init_path", "extract_nth_frame", "video_blend",
  "color_hold", "turbo_mode", "turbo_steps", "frame_ghost", "handoff_file", "handoff_share", "handoff_start_denoise", "handoff_finish",
  "small_first", "small_denoise", "image_prompts", "image_prompt_weight", "edge_control", "depth_control", "pose_control", "person_control", "face_control", "face_shape", "face_detail"];
// Camera moves in the Animation section: the modes each one works in, and its slider's range.
const MOTION = {
  zoom: { modes: ["2D"], min: 0.9, max: 1.1, step: 0.005 },
  angle: { modes: ["2D"], min: -5, max: 5, step: 0.1 },
  translation_x: { modes: ["2D", "3D"], min: -20, max: 20, step: 0.5 },
  translation_y: { modes: ["2D", "3D"], min: -20, max: 20, step: 0.5 },
  translation_z: { modes: ["3D"], min: -30, max: 30, step: 0.5 },
  rotation_3d_x: { modes: ["3D"], min: -3, max: 3, step: 0.05 },
  rotation_3d_y: { modes: ["3D"], min: -3, max: 3, step: 0.05 },
  rotation_3d_z: { modes: ["3D"], min: -3, max: 3, step: 0.05 },
};
QUICK.push(...Object.keys(MOTION));
const HANDLED = new Set(QUICK);
/* Settings drawn inside the cards, each kept with the option it shapes. A row in the page marked data-sub="key"
   gets a settings toggle that opens CARD_FIELDS[key] under it. Inside a list, a setting that is itself a key here
   gets the same toggle for its own settings, as Limit each step does for Step limit. */
const CARD_FIELDS = {
  // Whole cards that are a list of settings
  cutouts: ["cut_ic_pow", "cut_overview", "cut_innercut", "cut_icgray_p", "views_shapes", "views_fit", "cutn_batches", "skip_augs"],
  guidance: ["tv_scale", "sat_scale", "range_scale", "clamp_grad", "fuzzy_prompt",
    "diffusion_sampling_mode", "clip_denoised", "use_horizontal_symmetry", "use_vertical_symmetry", "transformation_percent"],
  clamp_grad: ["clamp_max"],
  fuzzy_prompt: ["rand_mag"],
  diffusion_sampling_mode: ["eta"],
  // Model
  checkers: ["RN101", "RN50x4", "RN50x16", "RN50x64"],
  // Handoff
  checkpoint: ["handoff_steps", "handoff_guidance", "handoff_sampler", "handoff_karras", "handoff_clip_skip", "handoff_few_steps", "handoff_negative", "handoff_keep"],
  small_first: ["small_hold", "small_side"],
  small_hold: ["small_hold_curve"],
  // Tiles
  tile_scale: ["tile_overlap_pct"],
  // Animation
  prompt_changes: ["prompt_fade"],
  clip_style: ["video_init_seed_continuity"],
  video_blend: ["video_flow"],
  loop_mode: ["loop_frames"],
  fps: ["save_mp4", "save_webp", "save_gif"],
  // Keyframe shape shapes Frame skip keyframes in every clip mode, and the camera's curves, whose editor has it too.
  frames_skip_steps: ["frame_skip_keys", "interp_spline", "frames_scale"],
  camera: ["fov", "padding_mode", "sampling_mode", "frame_sharpen"],
  // Guide images
  ghost_image: ["ghost_lowfreq", "ghost_structure", "ghost_crops", "ghost_curve"],
  style_image: ["style_strength"],
  // Controls
  edge_control: ["edge_curve"],
  depth_control: ["depth_curve"],
  pose_control: ["pose_curve"],
  person_control: ["person_curve"],
  face_control: ["face_curve"],
  face_shape: ["face_shape_curve"],
  // Experiments
  time_travel: ["time_travel_curve"],
  prompt_graft: ["graft_curve"],
  live_ghost: ["sketch_steps", "partner_keep_sketch"],
  two_painters: ["painters_curve", "painters_strength", "painters_color", "painters_path"],
  borrow_attention: ["borrow_grids", "borrow_curve"],
  detail_bands: ["detail_curve"],
  shaped_guidance: ["guidance_curve"],
  layout_critic: ["layout_cuts", "layout_margin", "layout_curve"],
  robust_critic: ["robust_mode", "robust_curve"],
  one_subject: ["subject_words", "one_subject_curve"],
  correct_steps: ["correct_curve", "correct_snr", "correct_clip"],
  route_prompts: ["route_split", "route_curve"],
  partner: ["partner_model", "partner_file", "partner_every", "partner_follow"],
};

/* Settings that do nothing with the others as they are: grayed, with the reason on hover. */
const IDLE_WHEN = [
  ["color_hold", () => state.animation_mode === "Video Input" && !Number(state.video_blend), "With Video Input it holds only the paint {video_blend} carries over, so at 0 it does nothing."],
  ["eta", () => lookValue("diffusion_sampling_mode") === "plms", "plms doesn't use it. It's for ddim."],
  ["transformation_percent", () => !lookValue("use_horizontal_symmetry") && !lookValue("use_vertical_symmetry"), "Only with a mirror on."],
  ["sketch_steps", () => lookValue("live_ghost") !== "Sketch first", "Only with {live_ghost} set to Sketch first."],
  ["fov", () => state.animation_mode !== "3D", "3D only."],
  ["handoff_sampler", () => flowModel(), "Z-Image and Flux use their own sampler."],
  ["handoff_karras", () => flowModel(), "Z-Image and Flux use their own schedule."],
  ["handoff_few_steps", () => flowModel(), "Z-Image and Flux use their own schedule."],
  ["handoff_clip_skip", () => flowModel(), "For SD 1.5 and SDXL only."],
  ["handoff_negative", () => flowModel() === "flux" || distilledKlein()
     || (["zimage", "flux2"].includes(flowModel()) && Number(lookValue("handoff_guidance")) <= 1),
   "Flux and the distilled Kleins don't read one, and Z-Image and Klein Base read it only with CFG above 1."],
  ["handoff_guidance", () => (flowModel() === "flux" && checkpointNow()?.variant === "schnell") || distilledKlein(),
   "Flux schnell and the distilled Kleins have no guidance."],
];
function checkpointNow() { return checkpointList.find((c) => c.path === state.handoff_file); }
function flowModel() { const kind = checkpointNow()?.kind; return ["zimage", "flux", "flux2"].includes(kind) ? kind : ""; }
function distilledKlein() { return flowModel() === "flux2" && checkpointNow()?.variant === "klein"; }
function showIdle() {
  for (const [name, idle, why] of IDLE_WHEN) {
    const row = document.querySelector(`.controls .adv-field[data-name="${name.toLowerCase()}"]`);
    if (!row) continue;
    const off = idle();
    row.classList.toggle("unused", off);
    row.title = off ? worded(why) : "";
  }
  const hold = $("color_hold").closest(".slider-row"), holdIdle = IDLE_WHEN[0][1]();
  hold.classList.toggle("unused", holdIdle);
  hold.title = holdIdle ? worded(IDLE_WHEN[0][2]) : "";
}

// ---- main panel ----
function buildControls() {
  for (const id of ["width", "height"]) SIZES.forEach((s) => $(id).append(option(s)));
  Object.entries(LOOKS).forEach(([key, l]) => {
    $("look").append(option(key, l.label));
    const term = document.createElement("dt");
    term.textContent = l.label;
    $("helpLooks").append(term, withWords(document.createElement("dd"), l.hint));
  });
  $("look").append(option("custom", "Custom"));
  Object.entries(CLIP_STYLES).forEach(([key, l]) => $("clip_style").append(option(key, l.label)));
  $("clip_style").append(option("custom", "Custom"));
  for (const [name, spec] of Object.entries(MOTION)) {
    const row = document.createElement("div");
    row.className = "motion-row";
    row.dataset.modes = spec.modes.join(" ");
    const label = withWords(document.createElement("label"), `{${name}}`);
    label.className = "field-name";
    label.htmlFor = name;
    const slider = document.createElement("input");
    Object.assign(slider, { type: "range", id: `${name}_slider`, min: spec.min, max: spec.max, step: spec.step });
    slider.setAttribute("aria-label", term(name));
    slider.addEventListener("input", () => { $(name).value = `0:(${fmtValue(Number(slider.value))})`; showMotion(); });
    const keys = document.createElement("input");
    Object.assign(keys, { type: "text", id: name, className: "keys", spellcheck: false });
    const button = document.createElement("button");
    Object.assign(button, { type: "button", className: "icon-link", title: "Edit the curve" });
    button.setAttribute("aria-label", "Edit the curve");
    button.innerHTML = CURVE_ICON;
    button.addEventListener("click", () => openCurve(name));
    row.append(label, slider, keys, button);
    $("motionRows").append(row);
  }
  for (const [name] of CLIPS) {
    const row = document.createElement("label");
    row.className = "check";
    const box = document.createElement("input");
    box.type = "checkbox"; box.id = name;
    row.append(box, " ", withWords(document.createElement("span"), `{${name}}`));
    $("clipChecks").append(row);
  }
}

function controlValue(name) {
  const el = $(name);
  if (el.type === "checkbox") return el.checked;
  return el.type === "number" ? Number(el.value) : el.value;
}

function fillControls() {
  $("prompt").value = state.prompt;
  $("batch_name").value = state.batch_name;
  $("width").value = state.width;
  $("height").value = state.height;
  showOrientation();
  for (const name of QUICK) {
    const el = $(name);
    const value = state[name] ?? byName[name]?.default;
    if (el.type === "checkbox") el.checked = !!value;
    else if (name === "set_seed") el.value = seedOf(value) ? String(value) : "";
    else el.value = value ?? "";
  }
  showLook();
  showSchedulePreviews();  // a start image's skip shades the views schedules
  showInit();
  showAnimation();
  showPromptChanges();
  syncSliders();
  updateFit();
}

/* A seed of 0, an empty box or anything that isn't a whole number means a new random seed each render. */
function seedOf(value) {
  const n = Number(String(value ?? "").trim());
  return Number.isInteger(n) && n > 0 && n < 2 ** 32 ? n : 0;
}

function readControls() {
  state.prompt = $("prompt").value;
  state.batch_name = $("batch_name").value;
  state.width = Number($("width").value);
  state.height = Number($("height").value);
  for (const name of QUICK) {
    state[name] = name === "set_seed" ? (seedOf($("set_seed").value) ? String(seedOf($("set_seed").value)) : "random_seed") : controlValue(name);
  }
  save();
}

// ---- denoise ----
/* Plain names show skip_steps as Denoise, the share of the steps that are painted. */
function denoiseOf(skip, steps) {
  return steps > 0 ? Math.round(100 - (100 * skip) / steps) : 100;
}

function skipFor(denoise, steps) {
  return Math.min(Math.max(Math.round((steps * (100 - denoise)) / 100), 0), Math.max(steps - 1, 0));
}

/* A start image or a video keeps its layout only when part of the steps are skipped, so its arrival
   raises Denoise to 50%. Text alone needs every step to lay out the scene, so its departure puts that back. */
function startSkip(on) {
  const half = Math.round(state.steps / 2);
  if (on && state.skip_steps < half) state.skip_steps = half;
  else if (!on && !state.init_image && state.animation_mode !== "Video Input") state.skip_steps = 0;
}

// ---- orientation ----
/* The toggle remembers its choice through square sizes, which have no orientation of their own. */
function matchOrientation() {
  const wide = state.width > state.height;
  if (state.width !== state.height && wide !== (state.orientation === "landscape")) {
    [state.width, state.height] = [state.height, state.width];
  }
}

function followSize() {
  if (state.width !== state.height) state.orientation = state.width > state.height ? "landscape" : "portrait";
}

function showOrientation() {
  const square = state.width === state.height;
  for (const btn of document.querySelectorAll("[data-orientation]")) {
    btn.disabled = square;
    btn.setAttribute("aria-pressed", String(!square && btn.dataset.orientation === state.orientation));
  }
}

function setOrientation(orientation) {
  if (state.init_fit === "match") state.init_fit = "crop";
  state.orientation = orientation;
  matchOrientation();
  fillControls();
  save();
}

function onControlChange(e) {
  if (e.target.id === "look") { applyLook(e.target.value); return; }
  if (e.target.id === "clip_style") { applyPreset(CLIP_STYLES[e.target.value], STYLE_KEYS); return; }
  if (["width", "height"].includes(e.target.id) && state.init_fit === "match") state.init_fit = "crop";
  if (e.target.id === "denoise") $("skip_steps").value = skipFor(Number(e.target.value), Number($("steps").value));
  // Under plain names Denoise is what was chosen, so it holds when the steps change.
  if (e.target.id === "steps" && state.names === "plain") {
    $("skip_steps").value = skipFor(denoiseOf(state.skip_steps, state.steps), Number(e.target.value));
  }
  readControls();
  followSize();
  if (e.target.id === "tile_scale") { matchFrame(); $("width").value = state.width; $("height").value = state.height; }
  if (e.target.id === "max_frames") showPromptChanges();  // a change after the last frame says so
  if (e.target.id === "animation_mode") {
    // A video wants more repainting than a camera move, so each keeps its own tested default.
    const video = state.animation_mode === "Video Input";
    if (video && state.frames_skip_steps === CAMERA_SKIP) {
      state.frames_skip_steps = VIDEO_SKIP;
      // with the default blend too, so a new clip starts as Build up rather than Custom
      if (Number(state.video_blend) === byName.video_blend.default) state.video_blend = CLIP_STYLES.build.values.video_blend;
      $("video_blend").value = state.video_blend;
    }
    if (!video && state.frames_skip_steps === VIDEO_SKIP) state.frames_skip_steps = CAMERA_SKIP;
    $("frames_skip_steps").value = state.frames_skip_steps;
    startSkip(video);
    $("skip_steps").value = state.skip_steps;
  }
  showOrientation();
  showInit();
  showAnimation();
  syncSliders();
  save();
  showLook();
  showSchedulePreviews();
  updateFit();
}

// ---- animation ----
const CURVE_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 19c4 0 5-12 9-12s5 8 9 8"/><circle cx="12" cy="7" r="1.6"/></svg>';
const NUMBER = /^\s*[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?\s*$/i;
const fmtValue = (v) => String(Number(v.toFixed(4)));
const CAMERA_SKIP = 75, VIDEO_SKIP = 55;

function animating() { return state.animation_mode && state.animation_mode !== "None"; }

function showAnimation() {
  const mode = state.animation_mode || "None";
  const camera = ["2D", "3D"].includes(mode);
  document.body.classList.toggle("animating", mode !== "None");
  document.querySelectorAll("[data-make]").forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.make === "animation") === (mode !== "None"))));
  document.querySelectorAll("[data-anim]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.anim === mode)));
  $("videoRow").hidden = mode !== "Video Input";
  $("cameraBlock").hidden = !camera;
  $("turboCheck").hidden = !camera;
  $("turboEvery").hidden = !camera || !state.turbo_mode;
  document.querySelectorAll(".motion-row").forEach((row) => { row.hidden = !row.dataset.modes.split(" ").includes(mode); });
  showMotion();
  showVideo();
}

/* Image or Animation. Animation comes back as the kind last used. */
function setMake(make) {
  if (make === "animation" && !animating()) setAnimation(state.anim_last || "3D");
  if (make === "image" && animating()) setAnimation("None");
}

function setAnimation(mode) {
  if (mode !== "None") state.anim_last = mode;
  const select = $("animation_mode");
  select.value = mode;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

// ---- cards ----
const CARDS_OPEN = { size: true, render: true, model: false, cutouts: false, guidance: false, handoff: false, animation: true, tiles: false, guides: false, lab: false };

function cardOpen(name) { return state.cards?.[name] ?? CARDS_OPEN[name]; }

function showCards() {
  document.querySelectorAll(".card").forEach((card) => {
    const open = cardOpen(card.dataset.card);
    card.classList.toggle("closed", !open);
    card.querySelector(".card-head").setAttribute("aria-expanded", String(open));
  });
}

function openCard(name, open = true) {
  state.cards = { ...state.cards, [name]: open };
  save();
  showCards();
}

/* Tiles and Handoff are each one thing, on or off, so a switch on the card's title row turns them on and off.
   Off remembers the last size or model, which the dropdown keeps showing, and on brings it back. */
const FIRST_TILES = "2x, 4 tiles";

function showCardSwitches() {
  $("tilesOn").checked = tileGrid() > 1;
  $("tilePick").value = tileGrid() > 1 ? state.tile_scale : state.tile_last || FIRST_TILES;
  $("handoffOn").checked = !!state.handoff_file;
}

function setTiles(value) {
  if (tileGrid() > 1) state.tile_last = state.tile_scale;
  $("tile_scale").value = value;
  $("tile_scale").dispatchEvent(new Event("change", { bubbles: true }));
}

function switchHandoff(on) {
  if (!on) {
    if (state.handoff_file) state.handoff_last = state.handoff_file;
    pickCheckpoint("");
    return;
  }
  openCard("handoff");
  if (state.handoff_last) return pickCheckpoint(state.handoff_last);
  $("handoffOn").checked = false;  // it needs a model first, so the list opens to choose one
  $("handoffPick").focus();
  try { $("handoffPick").showPicker(); } catch { /* the list is in view to choose from */ }
}

/* Each card's title row says what it holds, so a closed card still shows its settings. */
function showSummaries() {
  const model = catalog?.find((m) => m.name === state.diffusion_model);
  const on = ["time_travel", "two_painters", "borrow_attention", "detail_bands", "shaped_guidance"].filter((n) => state[n]).length
    + ["prompt_graft", "depth_target", "gradient_momentum", "evidence_map", "layout_critic", "one_subject", "correct_steps"].filter((n) => Number(state[n])).length
    + ["live_ghost", "robust_critic", "route_prompts"].filter((n) => state[n] && state[n] !== "Off").length;
  const kind = { "2D": "2D", "3D": "3D", "Video Input": "Video" }[state.animation_mode];
  const sums = {
    size: `${state.width} × ${state.height}`,
    render: `${state.steps} steps`,
    model: model ? (state.names === "official" ? model.name : model.label) : "",
    animation: kind ? `${kind}, ${state.max_frames} frames` : "",
    tiles: tileGrid() === 1 ? "" : `${String(state.tile_scale).split(",")[0]}, ${finalSize(state.width, state.height).join("×")}`,
    lab: on ? `${on} on` : "Off",
    controls: controlsOn() ? `${controlsOn()} on` : "Off",
    handoff: [handingOff() && [Number(state.handoff_share) > 0 && `${state.handoff_share}%`, Number(state.handoff_finish) > 0 && !animating() && `finish ${state.handoff_finish}%`,
      modelName(state.handoff_file)].filter(Boolean).join(", "),
      smallFirstOn() && `small first ${state.small_first}%`].filter(Boolean).join("; "),
    cutouts: changedIn("cutouts") ? `${changedIn("cutouts")} changed` : model?.cuts ? "The model's own" : "Close-up ramp",
    guidance: changedIn("guidance") ? `${changedIn("guidance")} changed` : "Default",
  };
  for (const [name, text] of Object.entries(sums)) $(`sum-${name}`).textContent = text;
  showCardSwitches();
  for (const id of ["handoff_share", "handoff_start_denoise", "handoff_finish"]) $(id).closest(".slider-row").hidden = !state.handoff_file;
  // Start denoise sets where the checkpoint starts on a start image or a clip's frames, so from noise it does nothing.
  const fromStart = !!state.init_image || animating();
  const startRow = $("handoff_start_denoise").closest(".slider-row");
  startRow.classList.toggle("unused", !fromStart || !(Number(state.handoff_share) > 0));
  startRow.title = !fromStart ? "Only with a start image or on a clip's frames." : !(Number(state.handoff_share) > 0) ? worded("Only with {handoff_share} above 0.") : "";
  document.querySelectorAll(".handoff-part").forEach((el) => { el.hidden = !state.handoff_file; });
  // Small first is Disco's handoff to itself, so a checkpoint's handoff takes its place: grayed, saying why.
  const smallRow = $("small_first").closest(".slider-row");
  smallRow.classList.toggle("unused", !!state.handoff_file);
  smallRow.title = state.handoff_file ? "A handoff checkpoint already paints the start at its own size, so Small first is off while one is chosen." : "";
  const small = smallFrame(), where = $("smallWhere");
  where.hidden = !smallFirstOn() || tileGrid() > 1;
  // Small first paints only the steps before small.steps that aren't skipped: Denoise, then a clip's Frame skip.
  const smallFor = (skip) => (small ? Math.max(small.steps - skip, 0) : 0);
  const first = smallFor(Math.min(Number(state.skip_steps) || 0, state.steps - 1));
  const later = animating() ? smallFor(frameSkip(Math.max(Number(state.steps) || 1, 1))) : first;
  const video = animating() && state.animation_mode === "Video Input";
  const opening = video ? later : first;  // a video's first frame uses Frame skip too
  const size = `Paints ${small?.width} × ${small?.height} first`;
  let text;
  if (!small) text = `Does nothing at ${state.width} × ${state.height}: the frame is no larger than Small first size, ${smallSide()}.`;
  else if (!opening && !later) text = `Does nothing here: the skipped steps already cover its first ${small.steps} of ${state.steps}.`;
  else if (!animating()) text = `${size}, for ${first} of ${state.steps} steps.`;
  else if (opening === later) text = `${size}, for ${later} of ${state.steps} steps a frame.`;
  else if (!later) text = `${size} on the first frame only, for ${opening} of ${state.steps} steps.`;
  else text = `${size}, for ${opening} of ${state.steps} steps on the first frame and ${later} on each after it.`;
  where.textContent = text;
  $("controlWhere").textContent = controlSource();
  if (model) setHelp($("diffusion_model"), model.about);
}

/* How many of the Controls card's controls are on. */
function controlsOn() { return ["edge_control", "depth_control", "pose_control", "person_control", "face_control", "face_shape", "face_detail"].filter((n) => Number(state[n]) > 0).length; }

/* What the controls follow, as the renderer chooses it. */
function controlSource() {
  if (animating() && state.animation_mode === "Video Input") return "Follows each frame of the video, as it is before any painting.";
  if (animating()) return state.init_image ? "Follows the start image, then each frame's own start." : "Follows each frame's own start, from the second frame.";
  if (state.advanced.control_image) return "Follows the Control image.";
  return state.init_image ? "Follows the start image. Drop a Control image to follow another." : "Drop a Control image, or a start image, for these to follow.";
}

/* How many settings in a group are changed, counting the settings of options inside it. */
function changedIn(card) {
  return CARD_FIELDS[card].reduce((n, name) => n + (name in state.advanced) + (name !== card && CARD_FIELDS[name] ? changedIn(name) : 0), 0);
}

// ---- sliders ----
/* A slider and its number box are one setting. The number stays the source of truth. */
function syncSliders() {
  $("denoise").value = denoiseOf(Number($("skip_steps").value), Number($("steps").value));
  document.querySelectorAll("input[type=range][data-for]").forEach((range) => {
    const box = $(range.dataset.for);
    if (range.dataset.for === "skip_steps") range.max = Math.max(Number(state.steps) - 1, 0);
    if (range.dataset.for === "set_seed") {
      range.value = seedOf(box.value);
      if (!seedOf(box.value)) box.value = "";
      box.classList.toggle("changed", seedOf(box.value) > 0);
    } else {
      range.value = box.value;
      const fallback = range.dataset.for === "denoise" ? 100 : byName[range.dataset.for]?.default;
      box.classList.toggle("changed", fallback != null && Number(box.value) !== Number(fallback));
    }
    paintRange(range);
  });
  document.querySelectorAll(".motion-row input[type=range], .adv-range").forEach(paintRange);
}

/* The filled part of a slider's track, up to the thumb. */
function paintRange(range) {
  const low = Number(range.min) || 0, high = Number(range.max) || 100;
  const at = high > low ? (Math.min(Math.max(Number(range.value), low), high) - low) / (high - low) : 0;
  range.style.setProperty("--fill", `${(at * 100).toFixed(2)}%`);
}

function wireSliders(root = document) {
  root.querySelectorAll("input[type=range]").forEach((range) => range.addEventListener("input", () => paintRange(range)));
  root.querySelectorAll("input[type=range][data-for]").forEach((range) => {
    range.addEventListener("input", () => {
      const box = $(range.dataset.for);
      box.value = range.dataset.for === "set_seed" && range.value === "0" ? "" : range.value;
      box.dispatchEvent(new Event("change", { bubbles: true }));
    });
    range.addEventListener("change", (e) => e.stopPropagation());  // the box already reported it
  });
}

/* A steady value drives its slider. Keyframes leave the slider showing the first one, dimmed. */
function showMotion() {
  for (const name of Object.keys(MOTION)) {
    const keys = parseKeys($(name).value);
    const steady = keys.length === 1 && NUMBER.test(keys[0].expr);
    const first = keys.find((k) => NUMBER.test(k.expr));
    if (first) $(`${name}_slider`).value = Number(first.expr);
    paintRange($(`${name}_slider`));
    $(`${name}_slider`).closest(".motion-row").classList.toggle("keyed", !steady);
  }
}

/* The settings panel folds to a strip, which keeps Render and Stop, so the picture and images get the room. */
const STACKED = matchMedia("(max-width: 720px)");  // the panels stack, so the settings can't fold
function showControlsPanel() {
  const closed = !!state.controls_closed && !STACKED.matches;
  document.querySelector(".workspace").classList.toggle("controls-closed", closed);
  $("controlsToggle").title = closed ? "Show settings" : "Hide settings";
  $("controlsToggle").setAttribute("aria-label", $("controlsToggle").title);
  $("controlsToggle").setAttribute("aria-expanded", String(!closed));
}

function setUpPanel() {
  document.querySelector(".controls").addEventListener("change", onControlChange);
  $("prompt").addEventListener("input", readControls);
  document.querySelectorAll("[data-orientation]").forEach((btn) =>
    btn.addEventListener("click", () => setOrientation(btn.dataset.orientation)));
  $("controlsToggle").addEventListener("click", () => {
    state.controls_closed = !state.controls_closed;
    showControlsPanel();
    save();
  });
  document.querySelectorAll(".card-head").forEach((head) => head.addEventListener("click", () => {
    const name = head.closest(".card").dataset.card;
    openCard(name, !cardOpen(name));
  }));
  document.querySelectorAll("[data-make]").forEach((b) => b.addEventListener("click", () => setMake(b.dataset.make)));
  document.querySelectorAll("[data-anim]").forEach((b) => b.addEventListener("click", () => setAnimation(b.dataset.anim)));
  wireSliders();
  showCards();
  $("handoffOn").addEventListener("change", (e) => { e.stopPropagation(); switchHandoff(e.target.checked); });
  $("tilesOn").addEventListener("change", (e) => {
    e.stopPropagation();
    if (e.target.checked) openCard("tiles");
    setTiles(e.target.checked ? $("tilePick").value : "Off");
  });
  $("tilePick").addEventListener("change", (e) => { e.stopPropagation(); state.tile_last = e.target.value; setTiles(e.target.value); });
}
