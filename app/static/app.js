"use strict";
const $ = (id) => document.getElementById(id);
const STORE = "dd-ui-v1";
const SIZES = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280];
const CLIPS = [["ViTB32", "ViT-B/32"], ["ViTB16", "ViT-B/16"], ["ViTL14", "ViT-L/14"], ["RN50", "RN50"]];

// Controls in the main panel. All settings shows the rest.
const QUICK = ["steps", "skip_steps", "clip_guidance_scale", "n_batches", "set_seed", "diffusion_model",
  "use_secondary_model", "exact_repeat", "init_scale", "tile_scale", "tile_denoise", "tile_start", "tile_whole_view",
  "time_travel", "prompt_graft", "live_ghost", "depth_target", "two_painters", "borrow_attention", "detail_bands", "shaped_guidance", "gradient_momentum", "evidence_map", ...CLIPS.map((c) => c[0]),
  "animation_mode", "max_frames", "fps", "in_between", "frames_skip_steps", "video_init_path", "extract_nth_frame", "video_blend",
  "color_hold", "turbo_mode", "turbo_steps", "frame_ghost", "handoff_file", "handoff_share", "image_prompts", "image_prompt_weight"];
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
// Settings drawn inside the cards, beside the controls they shape. All settings points to them.
const CARD_FIELDS = {
  cutouts: ["cut_ic_pow", "cut_overview", "cut_innercut", "cut_icgray_p", "cutn_batches", "skip_augs"],
  guidance: ["tv_scale", "sat_scale", "range_scale", "clamp_grad", "clamp_max", "fuzzy_prompt", "rand_mag",
    "diffusion_sampling_mode", "eta", "clip_denoised", "use_horizontal_symmetry", "use_vertical_symmetry", "transformation_percent"],
  model: ["RN101", "RN50x4", "RN50x16", "RN50x64"],
  handoff: ["handoff_steps", "handoff_guidance", "handoff_few_steps", "handoff_negative", "handoff_keep"],
  tiles: ["tile_overlap"],
  render: ["cpu_threads", "display_rate"],
  animation: ["frames_scale", "frame_sharpen", "prompt_fade", "frame_skip_keys", "interp_spline", "fov", "padding_mode", "sampling_mode",
    "video_init_seed_continuity", "video_flow", "save_mp4", "save_webp", "save_gif"],
  guides: ["ghost_lowfreq", "ghost_structure", "ghost_crops", "ghost_curve", "style_strength"],
  lab: ["time_travel_curve", "graft_curve", "borrow_curve", "detail_curve", "guidance_curve", "painters_curve", "painters_strength",
    "painters_color", "painters_path", "partner_model", "partner_file", "sketch_steps", "partner_every", "partner_follow", "partner_keep_sketch"],
};
const CARD_TITLES = { cutouts: "{Cutouts}", guidance: "Guidance", model: "Model", render: "Render", animation: "Animation", handoff: "Handoff", tiles: "Tiles", guides: "Guide images", lab: "Experiments" };
const CARD_OF = Object.fromEntries(Object.entries(CARD_FIELDS).flatMap(([card, names]) => names.map((n) => [n, card])));

const EXACT_REPEAT_COST = 1.28;  // measured: 0.84 to 1.08 seconds per step at 512×512
const TIME_TRAVEL_COST = 1.35;  // measured at 512×512 with the default curve
const BORROW_SECONDS = 20;  // measured with Tiny SD at 512×512, loading included
const HANDOFF_SECONDS = 20;  // loading an SD 1.5 checkpoint and its own steps, at 512×512
/* The share of Disco's steps left after a handoff, measured with 200 steps: 30% left 98 of 200, 50% left 69. */
function afterHandoff() {
  if (!handingOff()) return 1;
  const share = Math.min(Math.max(Number(state.handoff_share) / 100, 0), 1);
  const points = [[0, 1], [0.3, 0.49], [0.5, 0.345], [1, 0.005]];
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1], [x1, y1] = points[i];
    if (share <= x1) return y0 + ((share - x0) / (x1 - x0)) * (y1 - y0);
  }
  return points[points.length - 1][1];
}

/* A look sets how the prompt shapes the picture, apart from its size. Settings it leaves out go back to
   their defaults. Each was compared with the others on the same prompts and seeds. */
const LOOKS = {
  balanced: { label: "Balanced", hint: "Lays out the whole scene first, then looks closer toward the end for detail, using the close-up ramp in the {Cutouts} card. Strong single subjects. The default.", values: {} },
  detail: { label: "Fine detail", hint: "The prompt checkers look closely from start to end. The crispest subjects, faces and textures, with busier scenes.", values: { cut_ic_pow: "[80]*1000" } },
  shapes: { label: "Big shapes", hint: "Looks at the whole picture from start to end. Bold, simple scenes and dioramas, with softer detail.", values: { cut_ic_pow: "[1]*1000" } },
  classic: {
    label: "Classic 2022", hint: "Settings from 2022 Disco renders: a gentler prompt, no smoothing, five rounds of {cuts_lower} and the {use_secondary_model}. Dreamy and loose. About twice as long.",
    values: { clip_guidance_scale: 5000, use_secondary_model: true, tv_scale: 0, sat_scale: 0, cutn_batches: 5, clamp_max: 0.05, cut_ic_pow: "[1]*1000" },
  },
};
const LOOK_KEYS = [...new Set(Object.values(LOOKS).flatMap((l) => Object.keys(l.values)))];

const INIT_SIZING = {
  match: "Sets the frame to the image's shape, at about the same size. The image then fills it.",
  fit: "Keeps the whole image. Spare space gets a blurred copy of it.",
  fill: "Stretches the whole image to the frame.",
  crop: "Fills the frame without stretching, cutting off what doesn't fit.",
};

let fields = [];
let byName = {};
let terms = {};
let state = { names: "plain", prompt: "", prompt_changes: [], batch_name: "Disco", width: 768, height: 512, init_image: "", init_name: "", init_fit: "crop", advanced: {} };
let lastStatus = null;
let selected = null;
let shownTask = null;
let previewRev = -1;

function load() {
  try { Object.assign(state, JSON.parse(localStorage.getItem(STORE)) || {}); } catch { /* start fresh */ }
}
function save() { localStorage.setItem(STORE, JSON.stringify(state)); }

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}
const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });

function fmtTime(s) {
  if (s == null) return "";
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${Math.round(s % 60)}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

function option(value, label) {
  const o = document.createElement("option");
  o.value = value; o.textContent = label ?? value;
  return o;
}

// ---- setting names ----
/* Text can hold {setting} tokens, which become the setting's plain or official name. */
const TOKEN = /\{([^{}]+)\}/g;

function term(name) {
  const official = state.names === "official";
  // {plain words|official words} picks one, for sentences that read differently under each name.
  if (name.includes("|")) return name.split("|")[official ? 1 : 0];
  return terms[name]?.[official ? 1 : 0] ?? name;
}

function worded(text) {
  return String(text).replace(TOKEN, (_, name) => term(name));
}

function withWords(el, text) {
  el.dataset.words = text;
  el.textContent = worded(text);
  return el;
}

function applyNames() {
  document.querySelectorAll("[data-words]").forEach((el) => { el.textContent = worded(el.dataset.words); });
  showHoverHelp();
  document.querySelectorAll("[data-names]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.names === state.names)));
  document.querySelectorAll("[data-shown-names]").forEach((el) => { el.hidden = el.dataset.shownNames !== state.names; });
}

function usesPartner() {
  return state.prompt_graft || state.two_painters || state.borrow_attention || state.live_ghost !== "Off";
}

// ---- accent color ----
const ACCENTS = [
  ["#00ff66", "Neon green"], ["#22d3ee", "Cyan"], ["#4c8dff", "Blue"], ["#b45cff", "Purple"],
  ["#ff4fa3", "Pink"], ["#ff6b4a", "Coral"], ["#ffb020", "Amber"], ["#e4e4e7", "White"],
];

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/* Sets the accent and the colors made from it: a lighter hover, a soft ring, and readable text on it. */
function applyAccent(hex) {
  const [r, g, b] = hexToRgb(hex);
  const lift = (c) => Math.round(c + (255 - c) * 0.3);
  const light = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  const root = document.documentElement.style;
  root.setProperty("--accent", hex);
  root.setProperty("--accent-hover", `rgb(${lift(r)} ${lift(g)} ${lift(b)})`);
  root.setProperty("--accent-soft", `rgb(${r} ${g} ${b} / .16)`);
  root.setProperty("--on-accent", light > 0.45 ? "#07140c" : "#ffffff");
  document.querySelectorAll(".swatch[data-color]").forEach((s) => s.setAttribute("aria-pressed", String(s.dataset.color === hex)));
  const custom = document.querySelector(".swatch.custom");
  if (custom) {
    custom.setAttribute("aria-pressed", String(!ACCENTS.some(([c]) => c === hex)));
    custom.querySelector("input").value = hex;
  }
}

function setAccent(hex) {
  state.accent = hex;
  save();
  applyAccent(hex);
}

function buildAccents() {
  const box = $("accentSwatches");
  for (const [hex, name] of ACCENTS) {
    const b = document.createElement("button");
    Object.assign(b, { type: "button", className: "swatch", title: name });
    b.dataset.color = hex;
    b.style.setProperty("--swatch", hex);
    b.setAttribute("aria-label", name);
    b.addEventListener("click", () => setAccent(hex));
    box.append(b);
  }
  const custom = document.createElement("label");
  custom.className = "swatch custom";
  custom.title = "Any color";
  const pick = document.createElement("input");
  Object.assign(pick, { type: "color" });
  pick.setAttribute("aria-label", "Any color");
  pick.addEventListener("input", () => setAccent(pick.value));
  custom.append(pick);
  box.append(custom);
  applyAccent(state.accent || ACCENTS[0][0]);
}

/* The setting a label names: the control it is for, or the one inside it. */
function settingOf(label) {
  let id = label.htmlFor || label.querySelector("input:not([type=hidden]), select")?.id || "";
  if (id.startsWith("adv-")) id = id.slice(4);
  if (id === "denoise") id = "skip_steps";
  return byName[id] ? id : null;
}

/* Resting the pointer on a setting's name shows its help, unless that is turned off in All settings. */
function showHoverHelp() {
  const on = state.hover_help !== false;
  if ($("hoverHelp")) $("hoverHelp").checked = on;
  document.body.classList.toggle("hover-help", on);
  document.querySelectorAll(".controls label, .controls .guide > .field-name").forEach((label) => {
    const name = label.matches("label") ? settingOf(label) : label.dataset.words?.match(/^\{(\w+)\}$/)?.[1];
    const help = name && byName[name]?.help;
    if (!help) return;
    if (on) label.title = worded(help); else label.removeAttribute("title");
  });
}

function setNames(names) {
  state.names = names;
  save();
  applyNames();
  loadModels();  // model names follow the plain or official choice
  updateFit();
}

// ---- help ----
function helpValue(value) {
  if (value == null) return "\u2014";
  if (value === "random_seed") return "Random";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (typeof value === "number") return value.toLocaleString("en-US");
  return Array.isArray(value) ? JSON.stringify(value) : String(value);
}

function helpTable(list) {
  const table = document.createElement("table");
  table.className = "ranges";
  const cells = (kind, values, parent) => values.forEach((value) => {
    parent.append(withWords(document.createElement(kind), value));
  });
  const head = table.createTHead().insertRow();
  cells("th", ["Setting", "Lowest", "Default", "Highest"], head);
  const body = table.createTBody();
  for (const f of list) {
    const row = body.insertRow();
    const name = document.createElement("th");
    name.scope = "row";
    withWords(name, f.name ? `{${f.name}}` : f.label);
    row.append(name);
    const ends = f.kind === "boolean" ? [false, true] : [f.low, f.high];
    cells("td", [helpValue(ends[0]), helpValue(f.shown_default ?? f.default), helpValue(ends[1])], row);
    const note = body.insertRow().insertCell();
    note.colSpan = 4;
    note.className = "note";
    withWords(note, f.help);
  }
  return table;
}

function buildHelpSettings() {
  const sections = new Map([["Main", [
    { label: "Width", kind: "number", low: 256, default: 768, high: 1280, help: "Image width in pixels, in steps of 64." },
    { label: "Height", kind: "number", low: 256, default: 512, high: 1280, help: "Image height in pixels, in steps of 64." },
  ]]]);
  for (const f of fields) {
    if (!sections.has(f.section)) sections.set(f.section, []);
    sections.get(f.section).push(f);
  }
  const box = $("helpSettings");
  for (const [title, list] of sections) {
    const heading = document.createElement("h4");
    withWords(heading, `{${title}}`);
    box.append(heading, helpTable(list));
  }
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

// ---- prompts that change during a clip ----
/* The prompt box holds the first frame's prompt. Each change is its own box with the frame it starts at. The
   renderer reads them as lines that start with the frame number, such as "120: a city at night". */
const FRAME_LINE = /^\s*(\d+)\s*:\s*(.*)$/;

/* Moves any "120:" lines out of the prompt box into changes, for prompts saved or reused in that form. */
function splitPromptChanges() {
  const first = [], changes = new Map((state.prompt_changes || []).map((c) => [c.frame, c.text.split("\n")]));
  let at = 0;
  for (const line of String(state.prompt || "").split("\n")) {
    const m = line.match(FRAME_LINE);
    if (m) { at = Number(m[1]); if (at > 0 && !changes.has(at)) changes.set(at, []); if (!m[2].trim()) continue; }
    if (at === 0) first.push(m ? m[2] : line);
    else changes.get(at).push(m ? m[2] : line);
  }
  state.prompt = first.join("\n").replace(/\n+$/, "");
  state.prompt_changes = [...changes].sort((a, b) => a[0] - b[0]).map(([frame, lines]) => ({ frame, text: lines.join("\n").trim() }));
}

/* The prompt lines a render gets. A still uses only the first prompt. */
function promptLines() {
  const lines = state.prompt.split("\n").map((p) => p.trim()).filter(Boolean);
  if (!animating()) return lines;
  for (const change of [...(state.prompt_changes || [])].sort((a, b) => a.frame - b.frame)) {
    const own = change.text.split("\n").map((p) => p.trim()).filter(Boolean);
    own.forEach((line, i) => lines.push(i ? line : `${change.frame}: ${line}`));
  }
  return lines;
}

function showPromptChanges() {
  const box = $("promptChanges");
  const frames = Number(state.max_frames) || 0;
  box.replaceChildren(...(state.prompt_changes || []).map((change, i) => {
    const row = document.createElement("div");
    row.className = "change";
    const head = document.createElement("div");
    head.className = "change-head";
    const label = document.createElement("label");
    label.className = "field-name";
    label.textContent = "From frame";
    const frame = document.createElement("input");
    Object.assign(frame, { type: "number", min: 1, step: 1, value: change.frame, className: "readout frame" });
    frame.setAttribute("aria-label", "Frame this prompt starts at");
    label.htmlFor = frame.id = `change-${i}`;
    const late = document.createElement("span");
    late.className = "hint late";
    late.textContent = frames && change.frame >= frames ? `after the last frame (${frames})` : "";
    const remove = document.createElement("button");
    Object.assign(remove, { type: "button", className: "icon-link", title: "Remove this change" });
    remove.setAttribute("aria-label", "Remove this change");
    remove.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
    head.append(label, frame, late, remove);
    const text = document.createElement("textarea");
    Object.assign(text, { rows: 2, value: change.text, placeholder: "The prompt from this frame on" });
    text.setAttribute("aria-label", `Prompt from frame ${change.frame}`);
    text.addEventListener("input", () => { change.text = text.value; save(); });
    frame.addEventListener("change", () => {
      const wanted = Math.max(1, Math.round(Number(frame.value) || 1));
      change.frame = state.prompt_changes.some((c) => c !== change && c.frame === wanted) ? wanted + 1 : wanted;
      state.prompt_changes.sort((a, b) => a.frame - b.frame);
      save();
      showPromptChanges();
    });
    frame.addEventListener("change", (e) => e.stopPropagation(), { capture: false });
    remove.addEventListener("click", () => { state.prompt_changes.splice(state.prompt_changes.indexOf(change), 1); save(); showPromptChanges(); });
    row.append(head, text);
    return row;
  }));
}

function addPromptChange() {
  readControls();
  const changes = state.prompt_changes || (state.prompt_changes = []);
  const frames = Math.max(Number(state.max_frames) || 60, 2);
  const last = changes.length ? changes.at(-1).frame : 0;
  // Halfway through what is left of the clip, so a new change lands inside it.
  let frame = Math.max(last + 1, Math.round((last + frames) / 2));
  while (changes.some((c) => c.frame === frame)) frame += 1;
  changes.push({ frame, text: "" });
  save();
  showPromptChanges();
  $("promptChanges").querySelector(".change:last-child textarea").focus();
}

// ---- looks ----
function lookValue(name) {
  if (HANDLED.has(name)) return state[name] ?? byName[name].default;
  return name in state.advanced ? state.advanced[name] : byName[name].default;
}

/* The look whose settings all match, so a look changed by hand shows as Custom. */
function currentLook() {
  const match = Object.entries(LOOKS).find(([, l]) =>
    LOOK_KEYS.every((name) => JSON.stringify(lookValue(name)) === JSON.stringify(l.values[name] ?? byName[name].default)));
  return match ? match[0] : "custom";
}

function showLook() {
  const key = currentLook();
  $("look").value = key;
  $("lookAbout").textContent = key === "custom" ? "Some of its settings were changed. Choose a look to set them back." : worded(LOOKS[key].hint);
}

function applyLook(key) {
  const look = LOOKS[key];
  if (!look) return;
  for (const name of LOOK_KEYS) {
    const value = look.values[name] ?? byName[name].default;
    if (HANDLED.has(name)) state[name] = value;
    else if (JSON.stringify(value) === JSON.stringify(byName[name].default)) delete state.advanced[name];
    else state.advanced[name] = value;
  }
  resetAdvancedPanel();
  fillControls();
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
  if (["width", "height"].includes(e.target.id) && state.init_fit === "match") state.init_fit = "crop";
  if (e.target.id === "denoise") $("skip_steps").value = skipFor(Number(e.target.value), Number($("steps").value));
  // Under plain names Denoise is what was chosen, so it holds when the steps change.
  if (e.target.id === "steps" && state.names === "plain") {
    $("skip_steps").value = skipFor(denoiseOf(state.skip_steps, state.steps), Number(e.target.value));
  }
  readControls();
  followSize();
  if (e.target.id === "tile_scale") { matchFrame(); $("width").value = state.width; $("height").value = state.height; }
  if (e.target.id === "animation_mode") {
    // A video wants more repainting than a camera move, so each keeps its own tested default.
    const video = state.animation_mode === "Video Input";
    if (video && state.frames_skip_steps === CAMERA_SKIP) state.frames_skip_steps = VIDEO_SKIP;
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
  updateFit();
}

// ---- animation ----
const CURVE_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 19c4 0 5-12 9-12s5 8 9 8"/><circle cx="12" cy="7" r="1.6"/></svg>';
const NUMBER = /^\s*[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?\s*$/i;
const fmtValue = (v) => String(Number(v.toFixed(4)));
const CAMERA_SKIP = 75, VIDEO_SKIP = 55;

function animating() { return state.animation_mode && state.animation_mode !== "None"; }

const ANIM_ABOUT = {
  "2D": "Zooms, turns and pans a flat picture.",
  "3D": "Flies a camera into the scene. Needs Depth Anything from Models.",
  "Video Input": "Repaints every frame of a video.",
};

function showAnimation() {
  const mode = state.animation_mode || "None";
  const camera = ["2D", "3D"].includes(mode);
  document.body.classList.toggle("animating", mode !== "None");
  document.querySelectorAll("[data-make]").forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.make === "animation") === (mode !== "None"))));
  document.querySelectorAll("[data-anim]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.anim === mode)));
  $("animAbout").textContent = ANIM_ABOUT[mode] || "";
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

/* Each card's title row says what it holds, so a closed card still shows its settings. */
function showSummaries() {
  const model = catalog?.find((m) => m.name === state.diffusion_model);
  const on = ["time_travel", "two_painters", "borrow_attention", "detail_bands", "shaped_guidance"].filter((n) => state[n]).length
    + ["prompt_graft", "depth_target", "gradient_momentum", "evidence_map"].filter((n) => Number(state[n])).length
    + (state.live_ghost && state.live_ghost !== "Off");
  const kind = { "2D": "2D", "3D": "3D", "Video Input": "Video" }[state.animation_mode];
  const sums = {
    size: `${state.width} × ${state.height}`,
    render: `${state.steps} steps`,
    model: model ? (state.names === "official" ? model.name : model.label) : "",
    animation: kind ? `${kind}, ${state.max_frames} frames` : "",
    tiles: state.tile_scale === "Off" ? "Off" : String(state.tile_scale).split(",")[0],
    lab: on ? `${on} on` : "Off",
    handoff: handingOff() ? `${state.handoff_share}%, ${baseName(state.handoff_file).replace(/\.safetensors$/i, "")}` : "Off",
    cutouts: changedIn("cutouts") ? `${changedIn("cutouts")} changed` : "Close-up ramp",
    guidance: changedIn("guidance") ? `${changedIn("guidance")} changed` : "Default",
  };
  for (const [name, text] of Object.entries(sums)) $(`sum-${name}`).textContent = text;
  $("handoff_share").closest(".slider-row").hidden = !state.handoff_file;
  if (model) $("modelAbout").textContent = model.about;
}

function changedIn(card) {
  return CARD_FIELDS[card].filter((n) => n in state.advanced).length;
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
  const text = keysText(curve.keys);
  if (document.activeElement !== $("curveText")) $("curveText").value = text;
  const frames = Math.max(Number(state.max_frames) || 1, curve.keys.at(-1).f + 1, 2);
  try {
    const r = await api(`/api/curve?text=${encodeURIComponent(text)}&frames=${frames}&shape=${curveShape()}`);
    $("curveError").hidden = !r.error;
    $("curveError").textContent = r.error || "";
    if (!r.error) {
      curve.values = r.values;
      if (rescale || !curve.range) curve.range = curveRange();
      drawCurve();
    }
  } finally {
    curveBusy = false;
    if (curveAgain) { curveAgain = false; refreshCurve(); }
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

function handingOff() {
  return !!state.handoff_file && Number(state.handoff_share) > 0;
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

function tileOverlap(width, height) {
  return Math.min(state.advanced.tile_overlap ?? byName.tile_overlap.default, Math.floor(Math.min(width, height) / 2));
}

/* The final image's size for a frame of this size. */
function finalSize(width, height) {
  const overlap = tileOverlap(width, height);
  const extra = tileGrid() - 1;
  return [width + (width - overlap) * extra, height + (height - overlap) * extra];
}

/* Steps the GPU runs, counting each tile's turn at a step as one. */
function stepsRun() {
  const grid = tileGrid();
  const free = Math.max(state.steps - state.skip_steps, 0);
  if (grid === 1) return free;
  const painted = Math.round(state.steps * state.tile_denoise) * grid * grid;
  if (state.init_image || (animating() && state.animation_mode === "Video Input")) return painted;
  return state.tile_start === "Render first" ? free + painted : free * grid * grid;
}

function showTiles() {
  const grid = tileGrid();
  const renderFirst = state.tile_start === "Render first";
  $("tileMore").hidden = $("tileNote").hidden = grid === 1;
  const video = animating() && state.animation_mode === "Video Input";
  $("tile_denoise").disabled = grid === 1 || !(state.init_image || renderFirst || video);
  $("tile_start").disabled = !!state.init_image || video;
  const [width, height] = finalSize(state.width, state.height);
  // With tiles, a start image is repainted by Tile denoise, so the start image's own Denoise has no say.
  const tiledStart = grid > 1 && !!(state.init_image || video);
  for (const id of ["denoise", "skip_steps"]) {
    const row = $(id).closest(".slider-row");
    row.classList.toggle("unused", tiledStart);
    row.title = tiledStart ? worded("Tiles use Tile denoise for the start image, so {skip_steps} isn't used.") : "";
  }
  $("tileNote").textContent = `Final size ${width}×${height}, from ${grid * grid} tiles at the frame size.`
    + (tiledStart ? " Tile denoise sets how much of the start image is repainted." : "");
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
  showSummaries();
  showImagePrompts();
  showBlocked();
  refreshPlan();
  showTime();
}

/* How long it should take, from the measured speeds. Only for the main model, which they were measured on. */
function showTime() {
  const note = $("timeNote");
  if (state.diffusion_model !== "512x512_diffusion_uncond_finetune_008100") { note.textContent = ""; return; }
  const mpx = (state.width * state.height) / 1e6;
  const { perStep } = fitFor(mpx, !!plan?.offload);
  const LOADING_SECONDS = 45;
  const seconds = stepsRun() * afterHandoff() * perStep * (state.exact_repeat ? EXACT_REPEAT_COST : 1) * (state.time_travel ? TIME_TRAVEL_COST : 1)
    + LOADING_SECONDS + (usesPartner() ? BORROW_SECONDS : 0) + (handingOff() ? HANDOFF_SECONDS : 0);
  if (animating()) {
    const later = Math.max(state.steps - Math.round((state.steps * state.frames_skip_steps) / 100), 1);
    const frames = Number(state.max_frames) || 1;
    const painted = state.turbo_mode && state.animation_mode !== "Video Input" ? Math.ceil(frames / Math.max(state.turbo_steps, 2)) : frames;
    const perFrame = (seconds - LOADING_SECONDS - (handingOff() ? HANDOFF_SECONDS : 0)) * ((later * tileGrid() ** 2) / Math.max(stepsRun(), 1));
    note.textContent = `About ${fmtTime(perFrame)} a frame, ${fmtTime(seconds + perFrame * (painted - 1))} in all`;
  } else {
    note.textContent = `About ${fmtTime(seconds)} an image`;
  }
}

// ---- memory plan and setting rules ----
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

/* The status bar's VRAM and RAM. Gray on the left is the rest of the PC. Disco follows in color: before a render,
   lighter, as the plan for the next one; while one runs, solid, as what it really uses. */
function showMemory() {
  const s = lastStatus;
  const busy = !!s?.task?.active && s.task.kind === "render";
  if (!busy) renderPlan = null;
  // A plan worked out while the last render was still running has its memory in "other", so it waits for a new one.
  if (!busy && plan?.vram && plan.idle) {
    const { vram, ram } = plan;
    const moved = movedOf(plan);
    const discoVram = vram.parts.map((p) => ({ ...p, kind: "vram", plan: true, label: `Next render: ${p.label}` }));
    const used = drawTrack($("vramPlan"), vram.total, [{ kind: "other", label: "Other apps and the desktop", gb: vram.other }, ...discoVram], vram.keep);
    $("vramPlan").setAttribute("aria-label", `The next render needs ${gb(used - vram.other)} of VRAM, on top of ${gb(vram.other)} in use`);
    $("vramText").textContent = `${used.toFixed(1)} / ${vram.total.toFixed(0)} GB`;
    $("vramText").classList.toggle("over", !!plan.too_big);
    const discoRam = ram.parts.filter((p) => p.kind !== "offloaded").map((p) => ({ ...p, kind: "ram", plan: true, label: `Next render: ${p.label}` }));
    const movedPart = moved ? [{ kind: "moved", plan: true, label: "Offloaded from VRAM, so it fits: the render runs slower", gb: moved }] : [];
    const ramUsed = drawTrack($("ramPlan"), ram.total, [{ kind: "other", label: "Other apps", gb: ram.other }, ...discoRam, ...movedPart]);
    $("ramPlan").setAttribute("aria-label", `The next render takes ${gb(ramUsed - ram.other)} of system RAM${moved ? `, ${gb(moved)} of it offloaded from VRAM` : ""}`);
    $("ramText").textContent = `${ramUsed.toFixed(0)} / ${ram.total.toFixed(0)} GB`;
    $("ramText").classList.toggle("over", ramUsed > ram.total);
    $("memPlan").title = plan.too_big ? "Too big for this GPU even with offloading. Lower the size or use fewer prompt checkers."
      : moved ? "It doesn't all fit in VRAM, so part of the working data moves to RAM, shown green there. Slower, but it works." : "";
    $("ramMeter").title = $("memPlan").title;
    return;
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
  $("memPlan").title = $("ramMeter").title = busy ? "What Disco uses now, after what the rest of the PC used before it started." : "";
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
  if (name === "handoff_file") return [$("handoffPick")];
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

// ---- all settings ----
function buildAdvanced() {
  const body = $("advBody");
  const sections = new Map();
  for (const f of fields) {
    if (HANDLED.has(f.name) && !CARD_OF[f.name]) continue;
    if (!sections.has(f.section)) sections.set(f.section, []);
    sections.get(f.section).push(f);
  }
  for (const [title, list] of sections) {
    const d = document.createElement("details");
    d.className = "adv-section";
    const s = document.createElement("summary");
    withWords(s, `{${title}} (${list.length})`);
    d.append(s);
    list.forEach((f) => d.append(CARD_OF[f.name] ? cardLink(f) : advField(f)));
    body.append(d);
  }
  makeSwitches(body);
}

/* A setting shown in a card is listed here too, so a search finds it, with a way to it. */
function cardLink(f) {
  const wrap = document.createElement("div");
  wrap.className = "adv-field moved";
  const everyName = (text) => text.replace(TOKEN, (_, name) => (terms[name] ?? [name]).join(" "));
  wrap.dataset.text = everyName(`${f.name} {${f.name}} ${f.help}`).toLowerCase();
  const name = withWords(document.createElement("span"), `{${f.name}}`);
  name.className = "adv-name";
  const go = document.createElement("button");
  go.type = "button";
  go.className = "link";
  withWords(go, `In the ${CARD_TITLES[CARD_OF[f.name]]} card`);
  go.addEventListener("click", () => showInCard(f.name));
  const cell = document.createElement("div");
  cell.className = "adv-control";
  cell.append(go);
  wrap.append(name, cell);
  return wrap;
}

function showInCard(name) {
  $("advanced").close();
  openCard(CARD_OF[name]);
  const row = $(`adv-${name}`)?.closest(".adv-field");
  if (!row) return;
  const more = row.closest("details");
  if (more) more.open = true;
  row.scrollIntoView({ block: "center", behavior: "smooth" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1600);
}

/* Fills each card's settings. Called again whenever saved settings change all at once. */
function buildCardFields() {
  document.querySelectorAll(".card-fields[data-fields]").forEach((box) => {
    box.replaceChildren(...CARD_FIELDS[box.dataset.fields].filter((n) => byName[n]).map((n) => advField(byName[n], true)));
    makeSwitches(box);
  });
  showMoreCounts();
  showHoverHelp();
}

function showMoreCounts() {
  document.querySelectorAll("details.card-more .card-fields").forEach((box) => {
    const n = changedIn(box.dataset.fields);
    box.closest("details").querySelector("summary").textContent = n ? `More settings (${n} changed)` : "More settings";
  });
}

/* A small drawing of a schedule, from the start of the render on the left to the end on the right. */
function schedulePicture(name, text) {
  const spec = { low: 0, ...SCHEDULES[name] };
  const levels = parseSchedule(text) || [];
  const hi = Math.max(spec.top, ...levels.map((l) => l.v));
  const lowest = spec.log ? spec.low : Math.min(spec.low, ...levels.map((l) => l.v));
  const share = (v) => (spec.log ? Math.log(Math.max(v, lowest) / lowest) / Math.log(hi / lowest) : (v - lowest) / (hi - lowest || 1));
  const W = 200, H = 28;
  const y = (v) => (H - 2 - share(v) * (H - 4)).toFixed(1);
  let at = 0, line = "";
  for (const l of levels) {
    const x0 = ((at / 1000) * W).toFixed(1), x1 = (((at + l.n) / 1000) * W).toFixed(1);
    line += `${line ? "L" : "M"}${x0} ${y(l.v)}L${x1} ${y(l.v)}`;
    at += l.n;
  }
  const values = levels.map((l) => fmtValue(l.v));
  const sum = values.length > 4 ? `${values[0]} … ${values.at(-1)}` : values.join(" → ");
  return { svg: `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true"><path class="sched-fill" d="${line}L${W} ${H}L0 ${H}Z"/><path class="sched-line" d="${line}"/></svg>`, sum };
}

/* Every schedule's graph, in the cards and in All settings, redrawn from the saved settings. */
function showSchedulePreviews() {
  document.querySelectorAll(".sched-preview[data-field]").forEach((button) => {
    const name = button.dataset.field;
    const text = name in state.advanced ? state.advanced[name] : byName[name].default;
    const { svg, sum } = schedulePicture(name, text);
    button.innerHTML = `${svg}<span class="sched-sum">${sum}</span>`;
    button.title = `${text}. Click to change it.`;
  });
}

// ---- schedules ----
/* Schedules hold a value for each part of the render, written [12]*400+[4]*600 over 1000 parts. The editor
   draws them as levels to drag. top is the highest value shown; log draws a wide range evenly. */
const SCHEDULES = {
  cut_overview: { top: 32, step: 1 }, cut_innercut: { top: 32, step: 1 },
  cut_ic_pow: { top: 100, step: 0.5, low: 0.5, log: true }, cut_icgray_p: { top: 1, step: 0.05 },
  time_travel_curve: { top: 4, step: 1 }, guidance_curve: { top: 3, step: 0.05 }, detail_curve: { top: 1, step: 0.05 },
  painters_curve: { top: 128, step: 4 }, graft_curve: { top: 1, step: 0.05 }, borrow_curve: { top: 1, step: 0.05 },
  ghost_curve: { top: 1, step: 0.05 },
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
  const current = name in state.advanced ? state.advanced[name] : f.default;
  sched = { name, spec: { low: 0, ...SCHEDULES[name] }, levels: parseSchedule(current) || parseSchedule(f.default), drag: null };
  withWords($("scheduleTitle"), `{${name}}`);
  withWords($("scheduleAbout"), f.help);
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
  if (text === scheduleText(parseSchedule(f.default))) delete state.advanced[sched.name];
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

/* Every changed setting in All settings has a reset button. */
function showResets() {
  document.querySelectorAll(".adv-field").forEach((wrap) => {
    const name = wrap.querySelector("[data-field]")?.dataset.field;
    const button = wrap.querySelector(".adv-reset");
    if (button) button.hidden = !(name in state.advanced);
  });
}

function resetField(name) {
  delete state.advanced[name];
  const f = byName[name];
  const input = $(`adv-${name}`);
  if (input) {
    if (f.kind === "boolean") input.checked = !!f.default;
    else input.value = f.kind === "raw" ? JSON.stringify(f.default) : f.default ?? "";
    input.dispatchEvent(new Event("change"));
  }
}

/* Slider ends for All settings, where the useful range in Help doesn't suit a slider. */
const SLIDER_ENDS = { cpu_threads: [0, 16], display_rate: [1, 100] };

/* [low, high, step] for a number setting's slider, or null when it has no sensible range. */
function sliderRange(f) {
  const [low, high] = SLIDER_ENDS[f.name] || [f.low, f.high];
  if (typeof low !== "number" || typeof high !== "number" || high <= low) return null;
  const rough = (high - low) / 100;
  const power = 10 ** Math.floor(Math.log10(rough));
  let step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= rough);
  if (Number.isInteger(f.default) && Number.isInteger(low) && Number.isInteger(high)) step = Math.max(1, Math.round(step));
  return [low, high, step];
}

/* Combinable choices, such as which checkers compare the image or which files to save, stay checkboxes.
   Every other on and off setting is a switch. */
const CHECKBOXES = new Set(["ViTB32", "ViTB16", "ViTL14", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "save_mp4", "save_webp", "save_gif"]);

function makeSwitches(root = document) {
  root.querySelectorAll('input[type="checkbox"]').forEach((box) => {
    const name = box.dataset.field || box.id;
    if (CHECKBOXES.has(name) || box.classList.contains("switch")) return;
    box.classList.add("switch");
    box.setAttribute("role", "switch");
    box.closest(".check")?.classList.add("toggle");
  });
}

function advField(f, compact = false) {
  const wrap = document.createElement("div");
  wrap.className = compact ? "adv-field card-field" : "adv-field";
  wrap.dataset.name = f.name.toLowerCase();
  const everyName = (text) => text.replace(TOKEN, (_, name) => (terms[name] ?? [name]).join(" "));
  wrap.dataset.text = everyName(`${f.name} {${f.name}} ${f.help}`).toLowerCase();
  const name = document.createElement("label");
  name.className = compact ? "field-name" : "adv-name";
  withWords(name, `{${f.name}}`);
  name.htmlFor = "adv-" + f.name;
  let input;
  const current = f.name in state.advanced ? state.advanced[f.name] : f.default;
  if (compact && SCHEDULES[f.name]) {
    input = document.createElement("input");  // the schedule as text, kept for the editor and resets
    Object.assign(input, { type: "hidden", value: current });
    wrap.append(name, input);
  } else if (f.kind === "boolean") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !!current;
    const box = document.createElement("div");
    box.className = "adv-bool";
    box.append(input);
    wrap.append(name, box);
  } else if (f.kind === "string" && f.options) {
    input = document.createElement("select");
    f.options.forEach((o) => input.append(option(o)));
    input.value = current;
    wrap.append(name, input);
  } else if (f.kind === "number" && sliderRange(f)) {
    // A slider over the useful range, with a box for typing any value, including past either end.
    const [low, high, step] = sliderRange(f);
    input = document.createElement("input");
    Object.assign(input, { type: "number", step: "any", value: current ?? "" });
    input.className = "readout";
    const range = document.createElement("input");
    Object.assign(range, { type: "range", min: low, max: high, step, value: current ?? f.default, className: "adv-range" });
    range.setAttribute("aria-label", term(f.name));
    range.addEventListener("input", () => { input.value = range.value; paintRange(range); });
    range.addEventListener("change", () => input.dispatchEvent(new Event("change")));
    input.addEventListener("change", () => { range.value = input.value; paintRange(range); });
    paintRange(range);
    const row = document.createElement("div");
    row.className = "adv-slider";
    row.append(range, input);
    wrap.append(name, row);
  } else {
    input = document.createElement("input");
    input.type = f.kind === "number" ? "number" : "text";
    if (f.kind === "number") input.step = "any";
    input.value = f.kind === "raw" ? JSON.stringify(current) : current ?? "";
    wrap.append(name, input);
  }
  input.id = "adv-" + f.name;
  input.dataset.field = f.name;
  input.addEventListener("change", () => advChanged(f, input, wrap));
  wrap.classList.toggle("changed", f.name in state.advanced);
  // The control, then an editor for schedules and a way back to the default for anything changed.
  const cell = document.createElement("div");
  cell.className = "adv-control";
  cell.append(wrap.lastChild);
  if (SCHEDULES[f.name]) {
    const edit = document.createElement("button");
    Object.assign(edit, { type: "button", className: "sched-preview" });
    edit.dataset.field = f.name;
    edit.setAttribute("aria-label", `Edit ${term(f.name)}`);
    edit.addEventListener("click", () => openSchedule(f.name));
    if (compact) cell.replaceChildren(edit, input); else cell.append(edit);
  }
  if (f.name === "partner_file") {
    const pick = document.createElement("button");
    Object.assign(pick, { type: "button", className: "default", textContent: "Browse" });
    pick.addEventListener("click", async () => {
      const path = await chooseCheckpoint(input.value);
      if (path) { input.value = path; input.dispatchEvent(new Event("change")); }
    });
    cell.append(pick);
  }
  const reset = document.createElement("button");
  Object.assign(reset, { type: "button", className: "icon-link adv-reset", title: "Back to the default" });
  reset.setAttribute("aria-label", "Back to the default");
  reset.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>';
  reset.hidden = !(f.name in state.advanced);
  reset.addEventListener("click", () => resetField(f.name));
  cell.append(reset);
  wrap.append(cell);
  if (f.help && !compact) {  // in a card, the help shows on hover instead
    const h = document.createElement("div");
    h.className = "adv-help";
    withWords(h, f.help);
    wrap.append(h);
  }
  return wrap;
}

function parseAdv(f, input) {
  if (f.kind === "boolean") return input.checked;
  if (f.kind === "number") return input.value === "" ? f.default : Number(input.value);
  if (f.kind === "raw") return JSON.parse(input.value);
  return input.value;
}

function advChanged(f, input, wrap) {
  wrap.classList.remove("bad");
  try {
    const value = parseAdv(f, input);
    if (JSON.stringify(value) === JSON.stringify(f.default)) delete state.advanced[f.name];
    else state.advanced[f.name] = value;
  } catch {
    wrap.classList.add("bad");
    return;
  }
  wrap.classList.toggle("changed", f.name in state.advanced);
  $("changedCount").textContent = changedLabel();
  showLook();
  if (LAB_IMAGES.includes(f.name)) showLabImages();
  wrap.querySelector(".adv-reset").hidden = !(f.name in state.advanced);
  showSchedulePreviews();
  showMoreCounts();
  save();
  updateFit();
}

const changedLabel = () => {
  const n = Object.keys(state.advanced).length;
  return n ? `(${n} changed)` : "";
};

function resetAdvancedPanel() {
  $("advBody").replaceChildren();
  buildAdvanced();
  buildCardFields();
  showSchedulePreviews();
  $("changedCount").textContent = changedLabel();
}

function resetAdvanced() {
  state.advanced = {};
  resetAdvancedPanel();
  save();
}

function filterAdvanced() {
  const q = $("advSearch").value.trim().toLowerCase();
  $("namesBox").hidden = !!q && !"setting names plain official".includes(q);
  $("lookBox").hidden = !!q && !"accent color colour theme".includes(q);
  $("foldersBox").hidden = !!q && !"save images folders upscalers".includes(q);
  $("modelsBox").hidden = !!q && !"models downloads upscalers".includes(q);
  document.querySelectorAll(".adv-section").forEach((sec) => {
    let any = false;
    sec.querySelectorAll(".adv-field").forEach((row) => {
      const hit = !q || row.dataset.text.includes(q);
      row.hidden = !hit;
      row.style.display = hit ? "" : "none";
      any = any || hit;
    });
    sec.style.display = any ? "" : "none";
    sec.open = !!q && any;
  });
}

// ---- folders ----
function folderError(msg) {
  $("folderError").textContent = msg || "";
  $("folderError").hidden = !msg;
}

async function applyFolders(output_dir, upscale_dirs, checkpoint_dirs) {
  const before = (await api("/api/folders")).output_dir;
  const f = await post("/api/folders", { output_dir, upscale_dirs, ...(checkpoint_dirs ? { checkpoint_dirs } : {}) });
  await loadFolders();
  await loadUpscalers();
  await loadCheckpoints();
  if (f.output_dir !== before) {
    selected = null;
    $("imageActions").hidden = true;
    await loadGallery();
    await poll();
  }
}

async function loadFolders() {
  const f = await api("/api/folders");
  $("outputDir").value = f.output_dir;
  $("upscaleDirs").value = f.upscale_dirs.join("\n");
  $("checkpointDirs").value = (f.checkpoint_dirs || []).join("\n");
  return f;
}

async function loadUpscalers() {
  const { models } = await api("/api/upscalers");
  $("upscaler").replaceChildren(...models.map((m) => option(m, m.replace(/\.(pth|safetensors)$/, ""))));
  $("upscaleBtn").hidden = $("upscaler").hidden = !models.length;
}

async function saveFolders() {
  folderError("");
  try {
    await applyFolders($("outputDir").value, $("upscaleDirs").value.split("\n"), $("checkpointDirs").value.split("\n"));
  } catch (e) { folderError(e.message); }
}

async function browseFolder(title, start, kind = "folder") {
  if (!isLocal()) return browseRemote({ title, kind, start });
  const { path } = await post("/api/pick-folder", { title, start });
  return path;
}

// ---- choosing on the PC that renders ----
/* A browser on another PC can't open Windows' dialogs over there, so it walks that PC's folders here. */
let browsing = null;
const FOLDER_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" aria-hidden="true"><path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z"/></svg>';
const FILE_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" aria-hidden="true"><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/></svg>';

function browseRemote({ title, kind, start }) {
  if (browsing) finishBrowse(null);
  return new Promise((resolve) => {
    browsing = { kind, resolve, path: "" };
    $("browseTitle").textContent = title;
    $("browsePick").hidden = kind === "checkpoint";
    $("browseDialog").showModal();
    browseTo(start || "");
  });
}

async function browseTo(path) {
  if (!browsing) return;
  const kind = browsing.kind;
  $("browseError").hidden = true;
  let r;
  try {
    r = await api(`/api/browse?kind=${kind}&path=${encodeURIComponent(path)}`);
  } catch (e) {
    $("browseError").textContent = e.message;
    $("browseError").hidden = false;
    if (path && !browsing.path) browseTo("");  // a saved folder that is gone: start from the drives
    return;
  }
  if (!browsing) return;
  browsing.path = r.path;
  browsing.parent = r.parent;
  $("browsePath").value = r.path;
  $("browseUp").disabled = r.parent == null;
  $("browsePick").disabled = !r.path;
  $("browseWhere").textContent = kind === "checkpoint"
    ? `Folders on ${r.machine}, the PC that renders. Choose a .safetensors checkpoint.`
    : `Folders on ${r.machine}, the PC that renders. Open a folder, then use it.`;
  const rows = [];
  for (const f of r.folders) {
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = FOLDER_ICON;
    b.append(f.name);
    b.addEventListener("click", () => browseTo(f.path));
    rows.push(b);
  }
  for (const f of r.files) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "file";
    b.innerHTML = FILE_ICON;
    const size = document.createElement("span");
    size.className = "size";
    size.textContent = `${f.gb}GB`;
    b.append(f.name, size);
    if (kind === "checkpoint") b.addEventListener("click", () => finishBrowse(f.path));
    else b.disabled = true;  // shown so you can tell the folder holds models
    rows.push(b);
  }
  $("browseList").replaceChildren(...rows.map((b) => { const li = document.createElement("li"); li.append(b); return li; }));
  if (!rows.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = kind === "checkpoint" ? "No folders or checkpoints here." : "No folders here.";
    $("browseList").append(li);
  }
}

function finishBrowse(path) {
  const b = browsing;
  browsing = null;
  if ($("browseDialog").open) $("browseDialog").close();
  b?.resolve(path || null);
}

// ---- models ----
let catalog = [];
let downloads = null;
let downloadKey = "";

async function loadModels() {
  catalog = (await api("/api/models")).models;
  const painters = catalog.filter((m) => m.kind === "diffusion");
  const select = $("diffusion_model");
  const chosen = select.value || state.diffusion_model;
  select.replaceChildren(...painters.map((m) => {
    const shown = state.names === "official" ? m.name : m.label;
    const o = option(m.name, m.downloaded ? shown : `${shown} (not downloaded)`);
    o.disabled = !m.downloaded;
    return o;
  }));
  select.value = chosen;
  if (!select.value) select.value = byName.diffusion_model.default;  // a saved name that no longer exists
  $("getModels").hidden = painters.every((m) => m.downloaded);
  buildModelRows();
}

function textBlock(name, about) {
  const block = document.createElement("div");
  block.className = "model-text";
  const title = document.createElement("div");
  title.className = "adv-name";
  title.textContent = name;
  const help = document.createElement("div");
  help.className = "adv-help";
  help.textContent = about;
  block.append(title, help);
  return block;
}

const sizeLabel = (gb) => (gb < 1 ? `${Math.round(gb * 1000)}MB` : `${gb}GB`);

function buildModelRows() {
  $("modelList").replaceChildren(...catalog.map((m) => {
    const row = document.createElement("div");
    row.className = "model-row";
    const size = document.createElement("span");
    size.className = "value";
    size.textContent = sizeLabel(m.size_gb);
    const status = document.createElement("span");
    status.className = "value model-state";
    status.dataset.model = m.name;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "default";
    button.dataset.model = m.name;
    button.addEventListener("click", () => requestDownloads([m.name]));
    row.append(textBlock(m.name, m.about), size, status, button);
    return row;
  }));
  const missing = catalog.filter((m) => !m.downloaded).length;
  $("modelsSummary").textContent = missing ? `Models (${missing} not downloaded)` : "Models";
  showModelStates();
}

function modelState(m) {
  if (m.downloaded) return "Installed";
  if (downloads?.current === m.name) return downloads.percent == null ? "Checking" : `${downloads.percent}%`;
  return downloads?.waiting.includes(m.name) ? "Waiting" : "";
}

function showModelStates() {
  for (const m of catalog) {
    const text = modelState(m);
    const failed = !!downloads?.errors[m.name];
    document.querySelector(`.model-state[data-model="${m.name}"]`).textContent = text;
    const button = document.querySelector(`button[data-model="${m.name}"]`);
    button.hidden = !!text;
    button.textContent = failed ? "Try again" : "Download";
  }
  const problem = Object.values(downloads?.errors ?? {})[0];
  if (problem) modelError(problem);
}

function modelError(message) {
  $("modelError").textContent = message;
  $("modelError").hidden = !message;
}

async function requestDownloads(names) {
  modelError("");
  try {
    await post("/api/downloads", { names });
    await poll();
  } catch (e) { modelError(e.message); }
}

async function showDownloads(d) {
  const key = JSON.stringify([d.current, d.waiting, d.errors]);
  const changed = key !== downloadKey;
  downloadKey = key;
  downloads = d;
  if (changed) {  // something finished, so the lists are out of date
    await loadModels();
    await loadUpscalers();
  }
  showModelStates();
  $("downloadState").hidden = !d.current;
  if (d.current) {
    const more = d.waiting.length ? `, ${d.waiting.length} more` : "";
    $("downloadState").textContent = `Downloading ${d.current} ${d.percent == null ? "" : d.percent + "%"}${more}`;
  }
}

// ---- welcome ----
function welcomeRow(m, ticked) {
  const row = document.createElement("label");
  row.className = "model-row";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.dataset.size = m.size_gb;
  box.checked = m.downloaded || ticked;
  box.disabled = m.downloaded;
  if (!m.downloaded) box.dataset.model = m.name;
  box.addEventListener("change", showWelcomeTotal);
  const size = document.createElement("span");
  size.className = "value";
  size.textContent = m.downloaded ? "Installed" : sizeLabel(m.size_gb);
  row.append(box, textBlock(m.name, m.about), size);
  return row;
}

const welcomeTicked = () => [...document.querySelectorAll("#welcome input[data-model]:checked")];

function showWelcomeTotal() {
  const total = welcomeTicked().reduce((sum, box) => sum + Number(box.dataset.size), 0);
  $("welcomeTotal").textContent = total ? `Downloading ${sizeLabel(Number(total.toFixed(2)))}` : "Nothing to download";
}

async function openWelcome() {
  const f = await api("/api/folders");
  $("welcomeOutput").value = f.output_dir;
  $("welcomeUpscalers").value = f.upscale_dirs.join("\n");
  const main = byName.diffusion_model.default;
  $("welcomeUpscaler").replaceChildren(...catalog.filter((m) => m.kind === "upscaler").map((m) => welcomeRow(m, true)));
  $("welcomeModels").replaceChildren(...catalog.filter((m) => m.kind !== "upscaler").map((m) => welcomeRow(m, m.name === main || m.kind === "helper")));
  showWelcomeTotal();
  $("welcome").showModal();
}

async function finishWelcome() {
  const shown = (message) => { $("welcomeError").textContent = message; $("welcomeError").hidden = !message; };
  shown("");
  try {
    await applyFolders($("welcomeOutput").value, $("welcomeUpscalers").value.split("\n"));
    const names = welcomeTicked().map((box) => box.dataset.model);
    if (names.length) await post("/api/downloads", { names });
    await post("/api/first-run");
    $("welcome").close();
    await poll();
  } catch (e) { shown(e.message); }
}

// ---- drop zones ----
/* Start, ghost, style and prompt images, and the video, are dropped onto a box or chosen by clicking it. They are
   uploaded, so a browser on another PC works the same as one on this PC. An image dragged from Images works too. */
const PREVIEW_CACHE = {};

/* Uploads one file, reporting progress, because a video can take a while. */
function upload(url, file, progress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.addEventListener("progress", (e) => { if (e.lengthComputable && progress) progress(e.loaded / e.total); });
    xhr.addEventListener("load", () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body); else reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    });
    xhr.addEventListener("error", () => reject(new Error("The upload didn't reach Nano.")));
    xhr.send(file);
  });
}

/* Every file in a drop, including the files inside a dropped folder. */
async function droppedFiles(transfer) {
  const entries = [...(transfer.items || [])].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.some((e) => e.isDirectory)) return [...transfer.files];
  const files = [];
  const walk = async (entry) => {
    if (entry.isFile) { files.push(await new Promise((ok, fail) => entry.file(ok, fail))); return; }
    const reader = entry.createReader();
    for (;;) {
      const batch = await new Promise((ok, fail) => reader.readEntries(ok, fail));
      if (!batch.length) break;
      for (const child of batch) await walk(child);
    }
  };
  for (const entry of entries) await walk(entry);
  return files;
}

function wireDrop(zone, { input, onFiles, onImage, onClear }) {
  const open = (e) => {
    if (e.target.closest(".drop-clear") || zone.closest(".blocked") || zone.classList.contains("busy")) return;
    input.click();
  };
  zone.addEventListener("click", open);
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(e); } });
  input.addEventListener("change", () => { if (input.files.length) onFiles([...input.files]); input.value = ""; });
  zone.querySelector(".drop-clear")?.addEventListener("click", (e) => { e.stopPropagation(); onClear(); });
  zone.addEventListener("dragover", (e) => {
    const types = e.dataTransfer.types;
    if (zone.closest(".blocked") || !(types.includes("Files") || (onImage && types.includes(DRAG_TYPE)))) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    zone.classList.add("over");
  });
  zone.addEventListener("dragleave", (e) => { if (!zone.contains(e.relatedTarget)) zone.classList.remove("over"); });
  zone.addEventListener("drop", async (e) => {
    zone.classList.remove("over");
    const rel = e.dataTransfer.getData(DRAG_TYPE);
    if (!rel && !e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    if (rel && onImage) { onImage(rel); return; }
    const files = await droppedFiles(e.dataTransfer);
    if (files.length) onFiles(files);
  });
}

/* Shows what a drop zone holds: a picture or video, its name, or the empty prompt. */
function showDrop(zone, { url, name, busy } = {}) {
  const full = !!(url || name || busy);
  zone.classList.toggle("full", full);
  zone.classList.toggle("busy", !!busy);
  zone.querySelector(".drop-empty").hidden = full;
  const media = zone.querySelector(".drop-image");
  if (media) {
    media.hidden = !url;
    if (url && media.getAttribute("src") !== url) media.src = url;
    if (!url) media.removeAttribute("src");
  }
  const label = zone.querySelector(".drop-name");
  if (label) { label.textContent = busy || name || ""; label.title = name || ""; }
  zone.querySelector(".drop-clear").hidden = !full || !!busy;
  zone.title = name || "";
}

/* A preview for a file the settings name, which may have been chosen on an earlier visit or typed in. */
async function previewOf(path) {
  if (!path) return null;
  if (!(path in PREVIEW_CACHE)) {
    try { PREVIEW_CACHE[path] = await api(`/api/input-preview?path=${encodeURIComponent(path)}`); } catch { return null; }
  }
  return PREVIEW_CACHE[path];
}

const baseName = (path) => String(path || "").split(/[\\/]/).filter(Boolean).pop() || "";

// ---- start image ----
function showInit() {
  updateFit();
  showLabImages();
  const zone = $("initDrop");
  if (!zone.classList.contains("busy")) showDrop(zone, { url: state.init_url, name: state.init_image ? state.init_name : "" });
  if (state.init_image && !state.init_url) {
    previewOf(state.init_image).then((p) => {
      if (!state.init_image || zone.classList.contains("busy")) return;
      if (p?.url) { state.init_url = p.url; save(); showDrop(zone, { url: p.url, name: state.init_name }); }
      if (p?.missing) showDrop(zone, { name: `${state.init_name} is missing. Drop it again.` });
    });
  }
  $("initOptions").hidden = !state.init_image;
  $("startBlock").hidden = animating() && state.animation_mode === "Video Input";
  $("matchImage").disabled = !state.init_size;
  for (const btn of document.querySelectorAll("[data-init-fit]")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.initFit === state.init_fit));
  }
  $("initFitHint").textContent = INIT_SIZING[state.init_fit];
}

function setInit(r) {
  readControls();
  state.init_image = r.path; state.init_name = r.name; state.init_size = r.size; state.init_url = r.url || "";
  state.match_area = 0;
  matchFrame();
  startSkip(true);
  fillControls();
  save();
}

async function uploadInit(files) {
  const file = files.find((f) => /\.(png|jpe?g|webp)$/i.test(f.name)) || files[0];
  const zone = $("initDrop");
  formError("");
  showDrop(zone, { busy: "Uploading" });
  try {
    const r = await upload(`/api/init-image?name=${encodeURIComponent(file.name)}`, file, (f) => showDrop(zone, { busy: `Uploading ${Math.round(f * 100)}%` }));
    zone.classList.remove("busy");
    setInit(r);
  } catch (e) { formError(e.message); }
  zone.classList.remove("busy");
  showInit();
}

async function initFromGallery(rel) {
  formError("");
  try { setInit(await post("/api/init-from-output", { rel })); } catch (e) { formError(e.message); }
}

function clearInit() {
  readControls();
  state.init_image = state.init_name = state.init_url = ""; state.init_size = null;
  if (state.init_fit === "match") state.init_fit = "crop";
  startSkip(false);
  fillControls();
  save();
}

// ---- video ----
function showVideo() {
  const path = state.video_init_path || "";
  const zone = $("videoDrop");
  if (zone.classList.contains("busy")) return;
  showDrop(zone, { url: path && state.video_url ? state.video_url : "", name: baseName(path) });
  if (path && !state.video_url) {
    previewOf(path).then((p) => {
      if (state.video_init_path !== path || zone.classList.contains("busy")) return;
      if (p?.url) { state.video_url = p.url; save(); showVideo(); }
      if (p?.missing) showDrop(zone, { name: `${baseName(path)} is missing. Drop it again.` });
    });
  }
}

function setVideo(path, url) {
  state.video_url = url || "";
  $("video_init_path").value = path;
  $("video_init_path").dispatchEvent(new Event("change", { bubbles: true }));
  showVideo();
}

async function uploadVideo(files) {
  const file = files[0];
  const zone = $("videoDrop");
  formError("");
  showDrop(zone, { busy: "Uploading" });
  try {
    const r = await upload(`/api/upload-video?name=${encodeURIComponent(file.name)}`, file, (f) => showDrop(zone, { busy: `Uploading ${Math.round(f * 100)}%` }));
    zone.classList.remove("busy");
    setVideo(r.path, r.url);
  } catch (e) { zone.classList.remove("busy"); formError(e.message); showVideo(); }
}

// ---- guide images ----
/* Ghost and style images live with the other Experimental settings, as full paths. Their boxes upload a copy and
   set the path. Image prompts are a folder: a drop of many images, or of a folder, becomes a new folder of them. */
const LAB_IMAGES = ["ghost_image", "style_image"];

function showLabImages() {
  let count = 0;
  for (const name of LAB_IMAGES) {
    const path = state.advanced[name] || "";
    const zone = document.querySelector(`[data-drop="${name}"]`);
    if (path) count += 1;
    if (zone.classList.contains("busy")) continue;
    const known = PREVIEW_CACHE[path];
    showDrop(zone, { url: known?.url || "", name: baseName(path) });
    if (path && !known) previewOf(path).then((p) => { if (p) showLabImages(); });
  }
  showImagePrompts();
  if (state.image_prompts) count += 1;
  $("sum-guides").textContent = count ? `${count} on` : "None";
}

function setLabImage(name, path) {
  if (path) state.advanced[name] = path;
  else delete state.advanced[name];
  resetAdvancedPanel();
  fillControls();
  save();
}

async function uploadLabImage(name, files) {
  const file = files.find((f) => /\.(png|jpe?g|webp)$/i.test(f.name)) || files[0];
  const zone = document.querySelector(`[data-drop="${name}"]`);
  formError("");
  showDrop(zone, { busy: "…" });
  try {
    const r = await upload(`/api/init-image?name=${encodeURIComponent(file.name)}`, file);
    PREVIEW_CACHE[r.path] = { url: r.url, size: r.size };
    zone.classList.remove("busy");
    setLabImage(name, r.path);
  } catch (e) { zone.classList.remove("busy"); formError(e.message); showLabImages(); }
}

async function labImageFromGallery(name, rel) {
  formError("");
  try {
    const r = await post("/api/init-from-output", { rel });
    PREVIEW_CACHE[r.path] = { url: r.url, size: r.size };
    setLabImage(name, r.path);
  } catch (e) { formError(e.message); }
}

let promptSet = { folder: null, images: [] };

function showImagePrompts() {
  const folder = state.image_prompts || "";
  const zone = document.querySelector('[data-drop="image_prompts"]');
  $("imagePromptWeightRow").hidden = !folder;
  if (zone.classList.contains("busy")) return;
  if (folder !== promptSet.folder) {
    promptSet = { folder, images: [] };
    if (folder) {
      api(`/api/prompt-images?folder=${encodeURIComponent(folder)}`).then((r) => {
        if (promptSet.folder !== folder) return;
        Object.assign(promptSet, { images: r.images, missing: r.missing });
        showImagePrompts();
      }).catch(() => {});
    }
  }
  const stack = zone.querySelector(".drop-stack");
  const shown = promptSet.images.filter((i) => i.url).slice(0, 3);
  stack.replaceChildren(...shown.map((i) => Object.assign(document.createElement("img"), { src: i.url, alt: "", loading: "lazy" })));
  stack.hidden = !shown.length;
  zone.classList.toggle("full", !!folder);
  zone.querySelector(".drop-empty").hidden = !!folder;
  zone.querySelector(".drop-clear").hidden = !folder;
  const n = promptSet.images.length;
  zone.querySelector(".drop-count").textContent = !folder ? "" : promptSet.missing ? "Missing" : `${n} image${n === 1 ? "" : "s"}`;
  zone.title = folder;
}

function setImagePrompts(path) {
  $("image_prompts").value = path;
  $("image_prompts").dispatchEvent(new Event("change", { bubbles: true }));
}

async function uploadPromptImages(files) {
  const images = files.filter((f) => /\.(png|jpe?g|webp)$/i.test(f.name)).slice(0, 200);
  const zone = document.querySelector('[data-drop="image_prompts"]');
  if (!images.length) { formError("Image prompts take PNG, JPG or WEBP images."); return; }
  formError("");
  const group = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
  zone.classList.add("busy", "full");
  zone.querySelector(".drop-empty").hidden = true;
  let folder = "";
  try {
    for (const [i, file] of images.entries()) {
      zone.querySelector(".drop-count").textContent = `${i + 1} of ${images.length}`;
      folder = (await upload(`/api/prompt-images?group=${group}&name=${encodeURIComponent(file.name)}`, file)).folder;
    }
  } catch (e) { formError(e.message); }
  zone.classList.remove("busy");
  if (folder) setImagePrompts(folder); else showImagePrompts();
}

// ---- handoff checkpoints ----
/* The list shows checkpoints in the checkpoint folders on the PC that renders. Browse, on that PC only, picks a
   file anywhere and adds its folder. */
let checkpointList = [];

async function loadCheckpoints() {
  try { checkpointList = (await api("/api/checkpoints")).checkpoints; } catch { checkpointList = []; }
  showCheckpoints();
}

function showCheckpoints() {
  const select = $("handoffPick");
  const current = state.handoff_file || "";
  // Grouped by the subfolder each checkpoint is in, as ComfyUI sorts them.
  const options = [option("", "Off")];
  const groups = new Map();
  for (const c of checkpointList) {
    const label = c.sdxl ? `${c.name} (SDXL)` : c.name;
    if (!c.group) { options.push(option(c.path, label)); continue; }
    if (!groups.has(c.group)) groups.set(c.group, document.createElement("optgroup"));
    groups.get(c.group).label = c.group;
    groups.get(c.group).append(option(c.path, label));
  }
  options.push(...groups.values());
  if (current && !checkpointList.some((c) => c.path === current)) options.push(option(current, baseName(current).replace(/\.safetensors$/i, "")));
  select.replaceChildren(...options);
  select.value = current;
  $("handoffWhere").textContent = checkpointList.length ? ""
    : isLocal() ? "Browse to a checkpoint, or add checkpoint folders in All settings." : "Browse to a checkpoint on the PC that renders, or add checkpoint folders in All settings.";
}

function pickCheckpoint(path) {
  $("handoff_file").value = path;
  $("handoff_file").dispatchEvent(new Event("change", { bubbles: true }));
  showCheckpoints();
}

/* A checkpoint file on the PC that renders: Windows' own dialog there, or the page's browser from another PC.
   Its folder joins the checkpoint folders either way. */
async function chooseCheckpoint(start) {
  if (isLocal()) return (await post("/api/pick-checkpoint", { start: start || "" })).path;
  const folder = start ? start.replace(/[\\/][^\\/]*$/, "") : $("checkpointDirs").value.split("\n")[0].trim();
  const path = await browseRemote({ title: "Choose a Stable Diffusion 1.5 or SDXL checkpoint", kind: "checkpoint", start: folder });
  return path ? (await post("/api/add-checkpoint", { path })).path : null;
}

async function browseCheckpoint() {
  try {
    const path = await chooseCheckpoint(state.handoff_file);
    if (!path) return;
    await loadCheckpoints();
    if (!state.handoff_share) { state.handoff_share = 30; $("handoff_share").value = 30; }
    pickCheckpoint(path);
    loadFolders();
  } catch (e) { formError(e.message); }
}

// ---- render ----
function formError(msg) {
  $("formError").textContent = msg || "";
  $("formError").hidden = !msg;
}

async function startRender() {
  formError("");
  readControls();
  const prompts = promptLines();
  if (!prompts.length || /^\d+\s*:/.test(prompts[0])) { formError("Add a prompt first."); $("prompt").focus(); return; }
  if (animating() && state.prompt_changes?.some((c) => !c.text.trim())) { formError("A prompt change is empty. Write its prompt or remove it."); return; }
  if (!CLIPS.some(([n]) => state[n]) && !Object.keys(state.advanced).length) { formError("Choose at least one CLIP model."); return; }
  if (state.skip_steps >= state.steps) { formError(worded("{Denoise must be above 0%.|skip_steps must be lower than steps.}")); return; }
  if (plan?.conflicts?.length) { formError(worded(plan.conflicts[0])); return; }

  if (animating() && state.animation_mode === "Video Input" && !state.video_init_path) { formError("Drop a video on the Video box for Video Input."); return; }
  const settings = { ...state.advanced };
  for (const name of QUICK) settings[name] = state[name];
  settings.width_height = [state.width, state.height];
  if (state.init_image) settings.init_image = state.init_image;
  const resume_clip = carryOn;
  carryOn = null;
  try {
    renderPlan = plan;
    await post("/api/render", { prompts, settings, batch_name: state.batch_name, resume_clip, options: { init_fit: state.init_fit } });
    shownTask = null; previewRev = -1;
    poll();
  } catch (e) { formError(e.message); }
}

function setStage(src, video) {
  const player = $("stageVideo");
  if (video) {
    if (player.getAttribute("src") !== video) { player.src = video; player.play().catch(() => {}); }
  } else {
    player.pause();
    player.removeAttribute("src");
    $("stageImg").src = src;
  }
  player.hidden = !video;
  $("stageImg").hidden = !!video;
  $("stageEmpty").hidden = true;
}

function renderTask(t) {
  const active = !!t?.active;
  $("renderBtn").disabled = active;
  $("upscaleBtn").disabled = active;
  $("stopBtn").hidden = !active;
  $("progress").hidden = !active;
  if (!t) return;

  if (active) {
    const pct = t.total ? Math.min(100, (t.step / t.total) * 100) : 0;
    $("barFill").style.width = pct + "%";
    const unit = animating() ? "frame" : "image";
    $("progressMsg").textContent = t.n_batches > 1 && t.state === "rendering" ? `${t.message}, ${unit} ${t.batch + 1} of ${t.n_batches}` : t.message;
    const bits = [];
    if (t.total) bits.push(`${t.step} / ${t.total}`);
    if (t.seconds_per_step) bits.push(`${t.seconds_per_step}s per ${t.kind === "upscale" ? "tile" : "step"}`);
    if (t.eta_s != null) bits.push(`${fmtTime(t.eta_s)} left`);
    $("progressNums").textContent = bits.join(" · ");
    $("resultBox").hidden = true;
    $("logBox").hidden = true;
    if (t.kind === "render" && t.preview_rev !== previewRev && t.preview_rev > 0) {
      previewRev = t.preview_rev;
      setStage(`/api/preview/${t.id}?r=${previewRev}`);
      selected = null;
      $("imageActions").hidden = true;
    }
    return;
  }

  if (shownTask === t.id) return;   // already reported this finished task
  shownTask = t.id;
  const box = $("resultBox");
  box.hidden = false;
  box.className = "result";
  if (t.state === "done") {
    const bits = [`Finished in ${fmtTime(t.elapsed_s)}`];
    if (t.seed != null) bits.push(`seed ${t.seed}`);
    if (t.peak_vram_gb) bits.push(`peak ${t.peak_vram_gb}GB VRAM`);
    box.textContent = bits.join(" · ");
  } else {
    box.textContent = t.message + (t.error && t.state !== "oom" ? ` ${t.error}` : "");
    if (t.state !== "stopped") box.classList.add("bad");
  }
  if (t.log?.length) {
    $("logBox").hidden = false;
    $("logText").textContent = t.log.join("\n");
  }
  loadGallery().then(() => {
    if (t.images?.length) selectByUrl(t.images[t.images.length - 1]);
  });
}

async function poll() {
  try {
    const s = await api("/api/status");
    // A task that ended before this page opened is old news, not a result to announce.
    if (!lastStatus && s.task && !s.task.active) shownTask = s.task.id;
    lastStatus = s;
    showStatus(s);
    renderTask(s.task);
    await showDownloads(s.downloads);
  } catch { showReady(false, "Not connected"); }  // the next tick retries
}

function showReady(ready, text) {
  $("statusState").textContent = text;
  $("statusState").classList.toggle("ready", ready);
  $("statusState").classList.toggle("not-ready", !ready);
}

function showStatus(s) {
  showMemory();
  showDisk(s.disk);
  showLocal(s.local !== false);
  showRemote(s);
  showReady(!s.task?.active, s.task?.active ? s.task.message : "Ready");
  $("engineWarn").hidden = !s.other_gpu_app;
  updateFit();
}

/* Some controls open a Windows dialog or folder on the PC that renders, so they only show in a browser on that PC. */
let local = true;
function isLocal() { return local; }

function showLocal(here) {
  if (here === local && document.body.dataset.place) return;
  local = here;
  document.body.dataset.place = here ? "local" : "remote";
  document.querySelectorAll(".local-only").forEach((el) => { el.hidden = !here; });
  // Another PC can choose model folders over there, through the page's own browser. Where images are saved is
  // set on the PC that renders.
  $("outputDir").readOnly = !here;
  for (const id of ["outputBrowse", "openFolder"]) $(id).hidden = !here;
  showCheckpoints();
  loadNetwork();
}

// ---- network ----
/* Share this PC lets a browser on another PC render here. That browser gives a code once. Render on another PC
   opens the other PC's Nano in this browser. */
async function loadNetwork() {
  if (!isLocal()) return;
  try { showNetwork(await api("/api/network")); } catch { /* the server is restarting */ }
}

function showNetwork(n) {
  $("shareToggle").checked = n.on;
  $("shareInfo").hidden = !n.on || !n.addresses.length;
  $("shareAddress").textContent = n.addresses.map((a) => `http://${a}:${n.port}`).join("  or  ");
  $("shareCode").textContent = n.code ? `${n.code.slice(0, 4)}-${n.code.slice(4)}` : "";
  $("shareError").textContent = n.error || "";
  $("shareError").hidden = !n.error;
}

async function setSharing(body) {
  $("shareToggle").disabled = true;
  try { showNetwork(await post("/api/network", body)); } catch (e) { $("shareError").textContent = e.message; $("shareError").hidden = false; }
  $("shareToggle").disabled = false;
}

function openRemote(e) {
  e.preventDefault();
  let address = $("remoteAddress").value.trim();
  if (!address) { $("remoteAddress").focus(); return; }
  if (!/^https?:\/\//i.test(address)) address = `http://${address}`;
  const url = new URL(address);
  if (!url.port) url.port = "7861";
  state.remote_address = $("remoteAddress").value.trim();
  save();
  location.href = url.origin + "/";
}

/* In a browser on another PC, the menu bar says whose GPU it is using. */
function showRemote(s) {
  if (s.version) $("aboutVersion").textContent = `Version ${s.version}.`;
  const remote = s.local === false;
  $("remoteBadge").hidden = !remote;
  $("backHome").hidden = !remote;
  if (remote) {
    $("remoteBadge").textContent = `Rendering on ${s.machine || location.hostname}`;
    $("remoteBadge").title = `This page runs on ${s.machine || location.hostname} at ${location.host}. Renders use its GPU, and images are saved there.`;
  }
}

// ---- gallery ----
function showGalleryPanel() {
  const closed = !!state.gallery_closed;
  document.querySelector(".workspace").classList.toggle("gallery-closed", closed);
  $("galleryToggle").title = closed ? "Show images" : "Hide images";
  $("galleryToggle").setAttribute("aria-label", $("galleryToggle").title);
  $("galleryToggle").setAttribute("aria-expanded", String(!closed));
}

/* Images are grouped by the folder they are in, inside the output folder. Each folder opens and closes, and
   an image dragged onto a folder moves there. Folders made in Explorer show up too. */
const DRAG_TYPE = "application/x-nano-image";

function thumbButton(item) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = item.video ? "thumb video" : "thumb";
  b.title = item.video ? `${item.name} (plays the clip)` : item.name;
  b.dataset.url = item.url;
  b.draggable = true;
  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = item.thumb || item.url;
  img.alt = item.name;
  img.draggable = false;
  b.append(img);
  b.addEventListener("click", () => select(item));
  b.addEventListener("contextmenu", (e) => { select(item); openImageMenu(e); });
  b.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData(DRAG_TYPE, item.rel);
    e.dataTransfer.effectAllowed = "move";
    document.body.classList.add("dragging-image");
  });
  b.addEventListener("dragend", () => document.body.classList.remove("dragging-image"));
  return b;
}

function folderClosed(name) { return !!state.folders_closed?.[name]; }

function folderSection(name, list, count) {
  const section = document.createElement("section");
  section.className = "folder";
  section.dataset.folder = name;
  section.classList.toggle("closed", folderClosed(name));
  const head = document.createElement("button");
  head.type = "button";
  head.className = "folder-head";
  head.setAttribute("aria-expanded", String(!folderClosed(name)));
  // Drawn like a layer group: a chevron, a small picture with a folder mark, the name and the count.
  const chev = document.createElement("span");
  chev.className = "folder-chev";
  chev.setAttribute("aria-hidden", "true");
  chev.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>';
  const cover = document.createElement("span");
  cover.className = "folder-cover";
  if (list[0]) {
    const img = document.createElement("img");
    Object.assign(img, { src: list[0].thumb || list[0].url, alt: "", loading: "lazy", draggable: false });
    cover.append(img);
  }
  const mark = document.createElement("span");
  mark.className = list[0] ? "folder-mark corner" : "folder-mark";
  mark.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>';
  cover.append(mark);
  const title = document.createElement("span");
  title.className = "folder-name";
  title.textContent = name || "Not in a folder";
  const number = document.createElement("span");
  number.className = "folder-count";
  number.textContent = count;
  head.append(chev, cover, title, number);
  head.title = name ? `Drop images here to move them into ${name}` : "Images saved straight into the output folder";
  head.addEventListener("click", () => {
    state.folders_closed = { ...state.folders_closed, [name]: !folderClosed(name) };
    if (!state.folders_closed[name]) delete state.folders_closed[name];
    save();
    section.classList.toggle("closed", folderClosed(name));
    head.setAttribute("aria-expanded", String(!folderClosed(name)));
  });
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.append(...list.map(thumbButton));
  if (!list.length) {
    const empty = document.createElement("p");
    empty.className = "hint folder-empty";
    empty.textContent = "Empty. Drag images here.";
    grid.append(empty);
  }
  section.append(head, grid);
  section.addEventListener("dragover", (e) => {
    if (!e.dataTransfer.types.includes(DRAG_TYPE)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    section.classList.add("drop");
  });
  section.addEventListener("dragleave", (e) => { if (!section.contains(e.relatedTarget)) section.classList.remove("drop"); });
  section.addEventListener("drop", (e) => {
    section.classList.remove("drop");
    const rel = e.dataTransfer.getData(DRAG_TYPE);
    if (rel) { e.preventDefault(); moveImage(rel, name); }
  });
  return section;
}

async function loadGallery() {
  const { items, folders = [], counts = {} } = await api("/api/gallery");
  $("galleryEmpty").hidden = items.length > 0 || folders.length > 0;
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item.folder)) groups.set(item.folder, []);
    groups.get(item.folder).push(item);  // folders with the newest images come first
  }
  for (const name of folders) if (!groups.has(name)) groups.set(name, []);
  $("galleryGrid").replaceChildren(...[...groups].map(([name, list]) => folderSection(name, list, counts[name] || 0)));
  $("galleryGrid").items = items;
  markSelected();
}

async function moveImage(rel, folder) {
  try {
    const wasSelected = selected?.rel === rel;
    const r = await post("/api/gallery/move", { rel, folder });
    await loadGallery();
    if (wasSelected && r.url) selectByUrl(r.url);
  } catch (e) { toast(e.message); }
}

function startNewFolder() {
  if (document.querySelector(".new-folder")) { document.querySelector(".new-folder input").focus(); return; }
  if (state.gallery_closed) { state.gallery_closed = false; save(); showGalleryPanel(); }
  const row = document.createElement("form");
  row.className = "new-folder";
  const input = document.createElement("input");
  Object.assign(input, { type: "text", placeholder: "Folder name", maxLength: 60, spellcheck: false });
  input.setAttribute("aria-label", "New folder name");
  row.append(input);
  const done = () => row.remove();
  row.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = input.value.trim();
    if (!name) { done(); return; }
    try {
      await post("/api/gallery/folder", { name });
      done();
      await loadGallery();
    } catch (err) { toast(err.message); input.focus(); }
  });
  input.addEventListener("keydown", (e) => { if (e.key === "Escape") done(); });
  input.addEventListener("blur", () => setTimeout(() => { if (!input.value.trim()) done(); }, 150));
  $("galleryBody").prepend(row);
  input.focus();
}

function markSelected() {
  document.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("selected", selected?.url === t.dataset.url));
}

function isUpscaled(item) {
  return /_x\d+\.png$/.test(item.name);
}

function select(item) {
  selected = item;
  carryOn = null;  // a new choice cancels a pending Continue the clip
  setStage(item.url, item.video);
  $("imageActions").hidden = false;
  $("reuseBtn").hidden = isUpscaled(item);
  markSelected();
}

function selectByUrl(url) {
  const item = ($("galleryGrid").items || []).find((i) => i.url === url);
  if (item) select(item);
}

async function reuseSettings() {
  if (!selected) return;
  formError("");
  try {
    const s = await api(`/api/image-settings?rel=${encodeURIComponent(selected.rel)}`);
    // Animations keep a prompt set per starting frame. Later sets get their frame number back.
    const sets = Object.entries(s.text_prompts || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
    if (sets.length) {
      state.prompt = sets.map(([at, lines]) => lines.map((line, i) => (Number(at) && !i ? `${at}: ${line}` : line)).join("\n")).join("\n");
      state.prompt_changes = [];
      splitPromptChanges();
    }
    if (s.width_height) { [state.width, state.height] = s.width_height; followSize(); }
    state.tile_scale = "Off";
    for (const name of QUICK) if (name in s) state[name] = s[name];
    state.advanced = {};
    for (const f of fields) {
      if (!HANDLED.has(f.name) && f.name in s && JSON.stringify(s[f.name]) !== JSON.stringify(f.default)) state.advanced[f.name] = s[f.name];
    }
    resetAdvancedPanel();
    // Each image in a run has its own seed: the run's seed plus its position.
    const position = Number(selected.name.match(/\((\d+)\)_(\d+)\.png$/)?.[2] || 0);
    if (s.seed != null) state.set_seed = String((Number(s.seed) + position) % 2 ** 32);
    state.n_batches = 1;
    state.init_fit = s.init_fit || state.init_fit;
    state.init_image = s.init_image || "";
    state.init_size = s.init_size || null;
    state.init_name = s.init_image ? s.init_image.split(/[\\/]/).pop() : "";
    state.init_url = "";
    state.video_url = "";
    fillControls();
    save();
  } catch (e) { formError(e.message); }
}

/* Carries on a clip from its last saved frame, with its own settings. Raise Frames first to lengthen it. */
let carryOn = null;
async function continueClip() {
  if (!selected?.video) return;
  await reuseSettings();
  if (!$("formError").hidden) return;
  carryOn = selected.rel;
  toast("Settings loaded. Raise Frames to lengthen the clip, then press Render.");
  openCard("animation");
  $("max_frames").focus();
}

async function useAsStartImage() {
  if (!selected) return;
  await initFromGallery(selected.rel);
  $("startBlock").scrollIntoView({ block: "nearest" });
}

async function upscale() {
  if (!selected) return;
  formError("");
  try {
    await post("/api/upscale", { rel: selected.rel, model: $("upscaler").value });
    shownTask = null;
    poll();
  } catch (e) { formError(e.message); }
}

// ---- right-click menu ----
let toastTimer = null;

function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 2000);
}

async function copyImage() {
  try {
    const blob = await fetch(selected.url).then((r) => r.blob());
    await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
    toast("Image copied");
  } catch (e) { formError(`Couldn't copy the image. ${e.message}`); }
}

function saveImage() {
  const link = document.createElement("a");
  link.href = selected.url;
  link.download = selected.name;
  link.click();
}

function imageMenuItems() {
  const items = [
    { label: "Open full size", run: () => window.open(selected.url, "_blank", "noopener") },
    ...(selected.video ? [{ label: "Open the clip", run: () => window.open(selected.video, "_blank", "noopener") },
      { label: "Continue the clip", run: continueClip }] : []),
    { label: "Copy image", run: copyImage },
    { label: "Save image", run: saveImage },
    null,
    { label: "Use as start image", run: useAsStartImage },
  ];
  if (!isUpscaled(selected)) items.push({ label: "Reuse settings", run: reuseSettings });
  if ($("upscaler").value && !$("upscaleBtn").hidden) {
    items.push({ label: `Upscale with ${$("upscaler").value}`, run: upscale, disabled: $("upscaleBtn").disabled });
  }
  if (isLocal()) items.push(null, { label: "Open folder", run: () => post("/api/open-output") });
  return items;
}

function closeImageMenu() {
  const menu = $("imageMenu");
  if (menu.hidden) return;
  menu.hidden = true;
  menu.opener?.focus();
}

function openImageMenu(e) {
  if (!selected) return;  // a live preview has no finished image behind it
  e.preventDefault();
  const menu = $("imageMenu");
  menu.opener = e.currentTarget;
  menu.replaceChildren(...imageMenuItems().map((item) => {
    if (!item) return document.createElement("hr");
    const b = document.createElement("button");
    b.type = "button";
    b.role = "menuitem";
    b.textContent = item.label;
    b.title = item.label;
    b.disabled = !!item.disabled;
    b.addEventListener("click", () => { closeImageMenu(); item.run(); });
    return b;
  }));
  menu.hidden = false;
  // The menu key has no pointer position, so it opens beside the element instead.
  const box = e.currentTarget.getBoundingClientRect();
  const x = e.clientX || box.left + 8, y = e.clientY || box.top + 8;
  menu.style.left = `${Math.max(Math.min(x, innerWidth - menu.offsetWidth - 4), 4)}px`;
  menu.style.top = `${Math.max(Math.min(y, innerHeight - menu.offsetHeight - 4), 4)}px`;
  menu.querySelector("button:not(:disabled)").focus();
}

function moveInImageMenu(e) {
  if (e.key === "Escape" || e.key === "Tab") { e.preventDefault(); closeImageMenu(); return; }
  const buttons = [...$("imageMenu").querySelectorAll("button:not(:disabled)")];
  const at = buttons.indexOf(document.activeElement);
  const next = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: buttons.length - 1 }[e.key];
  if (next == null) return;
  e.preventDefault();
  buttons[(next + buttons.length) % buttons.length].focus();
}

// ---- start ----
async function init() {
  load();
  buildAccents();  // first, so the page never shows the wrong color
  ({ fields, terms } = await api("/api/schema"));
  buildHelpSettings();
  byName = Object.fromEntries(fields.map((f) => [f.name, f]));
  buildControls();
  makeSwitches();
  buildAdvanced();
  buildCardFields();
  showSchedulePreviews();
  applyNames();
  for (const name of ["tile_scale", "tile_start", "live_ghost", "animation_mode"]) byName[name].options.forEach((o) => $(name).append(option(o)));
  await loadModels();
  showGalleryPanel();
  // Saved settings with no orientation start in landscape.
  if (!state.orientation) { state.orientation = "landscape"; followSize(); }
  delete state.preset;  // an unused key from older saved settings
  splitPromptChanges();
  fillControls();
  readControls();  // every control's default goes into state
  $("changedCount").textContent = changedLabel();

  await loadFolders();
  await loadUpscalers();
  await loadCheckpoints();

  document.querySelector(".controls").addEventListener("change", onControlChange);
  $("prompt").addEventListener("input", readControls);
  document.querySelectorAll("[data-orientation]").forEach((btn) =>
    btn.addEventListener("click", () => setOrientation(btn.dataset.orientation)));
  $("renderBtn").addEventListener("click", startRender);
  $("stopBtn").addEventListener("click", () => post("/api/stop"));
  $("advancedBtn").addEventListener("click", () => $("advanced").showModal());
  document.querySelectorAll("[data-names]").forEach((btn) =>
    btn.addEventListener("click", () => setNames(btn.dataset.names)));
  $("getModels").addEventListener("click", () => {
    $("advanced").showModal();
    $("modelsBox").open = true;
    $("modelsBox").scrollIntoView();
  });
  $("showWelcome").addEventListener("click", () => { $("help").close(); openWelcome(); });
  $("welcomeStart").addEventListener("click", finishWelcome);
  $("welcomeOutputBrowse").addEventListener("click", async () => {
    const path = await browseFolder("Save images to", $("welcomeOutput").value);
    if (path) $("welcomeOutput").value = path;
  });
  $("welcomeUpscalersBrowse").addEventListener("click", async () => {
    const path = await browseFolder("Add an upscaler folder", "");
    if (path) $("welcomeUpscalers").value = [$("welcomeUpscalers").value.trim(), path].filter(Boolean).join("\n");
  });
  $("advClose").addEventListener("click", () => $("advanced").close());
  $("helpBtn").addEventListener("click", () => $("help").showModal());
  $("helpClose").addEventListener("click", () => $("help").close());
  for (const id of ["advanced", "help"]) {
    $(id).addEventListener("click", (e) => { if (e.target === $(id)) $(id).close(); });
  }
  $("outputDir").addEventListener("change", saveFolders);
  $("upscaleDirs").addEventListener("change", saveFolders);
  $("outputBrowse").addEventListener("click", async () => {
    const path = await browseFolder("Save images to", $("outputDir").value);
    if (path) { $("outputDir").value = path; saveFolders(); }
  });
  $("upscaleBrowse").addEventListener("click", async () => {
    const path = await browseFolder("Add an upscaler folder", "", "upscaler");
    if (path) { $("upscaleDirs").value = [$("upscaleDirs").value.trim(), path].filter(Boolean).join("\n"); saveFolders(); }
  });
  $("galleryToggle").addEventListener("click", () => {
    state.gallery_closed = !state.gallery_closed;
    showGalleryPanel();
    save();
  });
  $("advReset").addEventListener("click", resetAdvanced);
  $("modelsBox").open = !!state.models_open;
  $("modelsBox").addEventListener("toggle", () => { state.models_open = $("modelsBox").open; save(); });
  document.querySelectorAll(".card-head").forEach((head) => head.addEventListener("click", () => {
    const name = head.closest(".card").dataset.card;
    openCard(name, !cardOpen(name));
  }));
  document.querySelectorAll("[data-make]").forEach((b) => b.addEventListener("click", () => setMake(b.dataset.make)));
  document.querySelectorAll("[data-anim]").forEach((b) => b.addEventListener("click", () => setAnimation(b.dataset.anim)));
  wireSliders();
  showCards();
  $("handoffBrowse").addEventListener("click", browseCheckpoint);
  $("handoffPick").addEventListener("change", (e) => {
    if (e.target.value && !Number(state.handoff_share)) { $("handoff_share").value = 30; }
    pickCheckpoint(e.target.value);
  });
  $("addPromptChange").addEventListener("click", addPromptChange);
  $("shareToggle").addEventListener("change", () => setSharing({ on: $("shareToggle").checked }));
  $("shareNewCode").addEventListener("click", () => setSharing({ new_code: true }));
  $("remoteForm").addEventListener("submit", openRemote);
  $("remoteAddress").value = state.remote_address || "";
  wireDrop($("initDrop"), { input: $("initFile"), onFiles: uploadInit, onImage: initFromGallery, onClear: clearInit });
  wireDrop($("videoDrop"), { input: $("videoFile"), onFiles: uploadVideo, onClear: () => setVideo("", "") });
  for (const name of LAB_IMAGES) {
    wireDrop(document.querySelector(`[data-drop="${name}"]`), { input: $(`${name}File`), onFiles: (files) => uploadLabImage(name, files),
      onImage: (rel) => labImageFromGallery(name, rel), onClear: () => setLabImage(name, "") });
  }
  wireDrop(document.querySelector('[data-drop="image_prompts"]'), { input: $("image_promptsFile"), onFiles: uploadPromptImages, onClear: () => setImagePrompts("") });
  $("checkpointDirs").addEventListener("change", saveFolders);
  $("checkpointBrowse").addEventListener("click", async () => {
    const path = await browseFolder("Add a checkpoint folder", "", "checkpoint-folder");
    if (path) { $("checkpointDirs").value = [$("checkpointDirs").value.trim(), path].filter(Boolean).join("\n"); saveFolders(); }
  });
  $("hoverHelp").addEventListener("change", () => { state.hover_help = $("hoverHelp").checked; save(); showHoverHelp(); });
  $("browseUp").addEventListener("click", () => browsing && browseTo(browsing.parent || ""));
  $("browsePath").addEventListener("change", () => browseTo($("browsePath").value.trim()));
  $("browsePick").addEventListener("click", () => browsing?.path && finishBrowse(browsing.path));
  $("browseCancel").addEventListener("click", () => finishBrowse(null));
  $("browseDialog").addEventListener("close", () => { if (browsing) finishBrowse(null); });
  $("curveDone").addEventListener("click", closeCurve);
  $("curveDialog").addEventListener("cancel", (e) => { e.preventDefault(); closeCurve(); });
  $("curveShape").addEventListener("change", setCurveShape);
  $("curveText").addEventListener("change", () => { curve.keys = parseKeys($("curveText").value); if (curve.keys.length) refreshCurve(true); });
  $("curveCanvas").addEventListener("pointerdown", curvePointerDown);
  $("curveCanvas").addEventListener("pointermove", curvePointerMove);
  $("curveCanvas").addEventListener("pointerup", curvePointerUp);
  $("curveCanvas").addEventListener("dblclick", curveDoubleClick);
  $("scheduleCanvas").addEventListener("pointerdown", schedDown);
  $("scheduleCanvas").addEventListener("pointermove", schedMove);
  $("scheduleCanvas").addEventListener("pointerup", schedUp);
  $("scheduleCanvas").addEventListener("dblclick", schedDoubleClick);
  $("scheduleText").addEventListener("change", schedTyped);
  $("scheduleDone").addEventListener("click", () => { $("scheduleDialog").close(); sched = null; });
  $("scheduleDialog").addEventListener("close", () => { sched = null; });
  $("scheduleReset").addEventListener("click", () => { sched.levels = parseSchedule(byName[sched.name].default); commitSchedule(); showSchedule(); });
  $("advSearch").addEventListener("input", filterAdvanced);
  document.querySelectorAll("[data-init-fit]").forEach((btn) => btn.addEventListener("click", () => {
    state.init_fit = btn.dataset.initFit;
    state.match_area = 0;
    matchFrame();
    fillControls();
    save();
  }));
  $("openFolder").addEventListener("click", () => post("/api/open-output"));
  $("newFolder").addEventListener("click", startNewFolder);
  $("reuseBtn").addEventListener("click", reuseSettings);
  $("useInitBtn").addEventListener("click", useAsStartImage);
  $("upscaleBtn").addEventListener("click", upscale);
  document.querySelector(".canvas").addEventListener("contextmenu", openImageMenu);
  $("imageMenu").addEventListener("keydown", moveInImageMenu);
  document.addEventListener("pointerdown", (e) => { if (!$("imageMenu").contains(e.target)) closeImageMenu(); });
  document.addEventListener("scroll", closeImageMenu, true);
  addEventListener("resize", closeImageMenu);
  addEventListener("blur", closeImageMenu);

  await loadGallery();
  await poll();
  setInterval(poll, 1000);
  if ((await api("/api/first-run")).show) await openWelcome();
}

init().catch((e) => formError(`Could not start: ${e.message}`));
