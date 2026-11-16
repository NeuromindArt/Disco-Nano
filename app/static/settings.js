/* The Settings window and the settings drawn in the cards: options with settings of their own, sliders, switches,
   schedule previews, resets and search. */
"use strict";

// ---- all settings ----
/* The Settings window holds how the app behaves, with its folders, models and network. Settings that shape the picture are in the cards. */
const APP_FIELDS = ["keep_models_loaded", "cpu_threads", "display_rate", "save_progress_video"];
// Fix's Inpaint: the model it paints with and how. They shape no render, so they live here, not in a card.
const INPAINT_FIELDS = ["inpaint_file", "inpaint_steps", "inpaint_guidance", "inpaint_context", "inpaint_grow", "inpaint_blend"];

function buildAdvanced() {
  const body = $("advBody");
  for (const [label, names] of [["Rendering", APP_FIELDS], ["Inpaint", INPAINT_FIELDS]]) {
    const box = document.createElement("section");
    box.className = "folders app-fields";
    box.setAttribute("aria-label", label);
    const title = document.createElement("span");
    title.className = "adv-name";
    title.textContent = label;
    box.append(title, ...names.filter((n) => byName[n]).map((n) => advField(byName[n])));
    body.append(box);
  }
  makeSwitches(body);
}

/* Fills each card's settings. Called again whenever saved settings change all at once. */
function buildCardFields() {
  wrapOptions(document);
  document.querySelectorAll(".controls .card-fields[data-fields]").forEach((box) => {
    if (!box.parentElement.closest(".card-fields")) fillFields(box);  // nested lists are filled by their parent
  });
  showMoreCounts();
  showHoverHelp();
}

function fillFields(box) {
  box.replaceChildren(...CARD_FIELDS[box.dataset.fields].filter((n) => byName[n]).map((n) => {
    const row = advField(byName[n], true);
    if (CARD_FIELDS[n]) row.dataset.sub = n;  // a setting with settings of its own
    return row;
  }));
  makeSwitches(box);
  wrapOptions(box);
  box.querySelectorAll(":scope > .opt > .opt-body > .card-fields").forEach(fillFields);
}

// ---- options with their own settings ----
const CHEVRON_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>';

/* Options that are switched on and off. While off, an option is a plain row. Switched on, it becomes a section
   with its settings open under it, which its arrow folds away. An option not listed here is always in use, and
   its arrow starts folded, like More settings. */
const isOn = (name) => { const v = lookValue(name); return !!v && v !== "Off" && Number(v) !== 0; };
const OPT_ON = {
  checkpoint: () => !!state.handoff_file,
  small_first: () => smallFirstOn(),
  small_hold: () => isOn("small_hold"),
  tile_scale: () => tileGrid() > 1,
  edge_control: () => isOn("edge_control"),
  depth_control: () => isOn("depth_control"),
  pose_control: () => isOn("pose_control"),
  person_control: () => isOn("person_control"),
  face_control: () => isOn("face_control"),
  face_shape: () => isOn("face_shape"),
  loop_mode: () => isOn("loop_mode"),
  // The ghost's settings shape any ghost: a Ghost image, Live ghost's sketch, or a clip's last frame.
  ghost_image: () => !!state.advanced.ghost_image || isOn("live_ghost") || (animating() && !!state.frame_ghost),
  style_image: () => !!state.advanced.style_image,
  // Also open while the partner is set away from Tiny SD, so a missing SD 1.5 file can be fixed from here.
  partner: () => !!usesPartner() || lookValue("partner_model") !== byName.partner_model?.default || !!lookValue("partner_file"),
  clamp_grad: () => isOn("clamp_grad"),
  fuzzy_prompt: () => isOn("fuzzy_prompt"),
  ...Object.fromEntries(["time_travel", "prompt_graft", "live_ghost", "two_painters", "borrow_attention", "detail_bands", "shaped_guidance",
    "layout_critic", "robust_critic", "one_subject", "correct_steps", "route_prompts"].map((n) => [n, () => isOn(n)])),
};

function optActive(key) { return OPT_ON[key] ? OPT_ON[key]() : true; }
function optOpen(key) { return state.opts && key in state.opts ? !!state.opts[key] : !!OPT_ON[key]; }

/* Puts each row marked data-sub in a box with an arrow beside it, which opens that option's own settings
   under it. The box takes over the row's place, and hides when the row is hidden. */
function wrapOptions(root) {
  root.querySelectorAll("[data-sub]:not(.opt-main)").forEach((row) => {
    const key = row.dataset.sub;
    const opt = document.createElement("div");
    opt.className = "opt";
    opt.dataset.opt = key;
    for (const c of ["anim-only", "image-only"]) {
      if (row.classList.contains(c)) { row.classList.remove(c); opt.classList.add(c); }
    }
    row.before(opt);
    row.classList.add("opt-main");
    const name = row.dataset.subLabel || (row.querySelector(".field-name, .adv-name") || row).textContent.trim().replace(/\s*\(.*\)$/, "");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "opt-toggle";
    toggle.innerHTML = CHEVRON_ICON;
    toggle.dataset.label = `${name} settings`;
    const body = document.createElement("div");
    body.className = "opt-body";
    body.id = `opt-${key}`;
    const fields = document.createElement("div");
    fields.className = "card-fields";
    fields.dataset.fields = key;
    // Rows marked data-in="key" belong to this option, such as Full-size denoise to Small first, so they move into its section.
    body.append(...document.querySelectorAll(`[data-in="${key}"]`), fields);
    // A slot among them puts the option's own settings there instead of last, as the checkpoint's settings go before
    // the two directions of a handoff that both use them.
    body.querySelector(":scope > .fields-slot")?.replaceWith(fields);
    toggle.setAttribute("aria-controls", body.id);
    toggle.setAttribute("aria-label", toggle.dataset.label);
    const flip = () => { state.opts = { ...state.opts, [key]: !optOpen(key) }; save(); showOpt(opt); };
    toggle.addEventListener("click", flip);
    if (row.classList.contains("group-label")) row.addEventListener("click", flip);  // a heading opens its settings too
    if (OPT_ON[key]) opt.classList.add("switched");
    opt.append(row, toggle, body);
    showOpt(opt);
  });
}

function showOpt(opt) {
  const key = opt.dataset.opt, active = optActive(key), open = active && optOpen(key);
  opt.classList.toggle("active", active);
  opt.classList.toggle("open", open);
  opt.querySelector(":scope > .opt-body").hidden = !open;
  const toggle = opt.querySelector(":scope > .opt-toggle");
  toggle.hidden = !active;
  toggle.setAttribute("aria-expanded", String(open));
  // A heading with nothing to show, such as Partner model with no partner in use, steps aside.
  opt.classList.toggle("unused", !active && opt.querySelector(":scope > .opt-main").classList.contains("group-label"));
}

function showOpts() { document.querySelectorAll(".opt").forEach(showOpt); }

/* Each arrow shows a dot when something inside it is changed. */
function showMoreCounts() {
  showOpts();
  document.querySelectorAll(".opt").forEach((opt) => {
    const toggle = opt.querySelector(":scope > .opt-toggle");
    const n = CARD_FIELDS[opt.dataset.opt] ? changedIn(opt.dataset.opt) : 0;
    toggle.classList.toggle("changed", n > 0);
    toggle.title = n ? `${toggle.dataset.label} (${n} changed)` : toggle.dataset.label;
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
  // The views schedules mark where shapes give way to detail, and shade the part a start image or Frame skip skips.
  let marks = "";
  if (VIEW_SCHEDULES.includes(name) && byName.views_fit) {
    const plan = viewPlan();
    // Up to where Disco's own steps begin: after a skip, and after a handoff model's steps.
    const skipped = plan.disco ? (plan.viewAt(Math.floor((1000 * plan.disco) / plan.steps)) / 1000) * W : 0;
    if (skipped) marks += `<rect class="sched-skip" x="0" y="0" width="${skipped.toFixed(1)}" height="${H}"/>`;
    const shape = shapePoints();
    for (let i = 1; i < 1000; i++) {
      if (shape[i] !== shape[i - 1]) marks += `<line class="sched-split" x1="${((i / 1000) * W).toFixed(1)}" x2="${((i / 1000) * W).toFixed(1)}" y1="0" y2="${H}"/>`;
    }
  }
  return { svg: `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true"><path class="sched-fill" d="${line}L${W} ${H}L0 ${H}Z"/><path class="sched-line" d="${line}"/>${marks}</svg>`, sum };
}

/* Every schedule's graph, in the cards and in Settings, redrawn from the saved settings. */
function showSchedulePreviews() {
  document.querySelectorAll(".sched-preview[data-field]").forEach((button) => {
    const name = button.dataset.field;
    const text = name in state.advanced ? state.advanced[name] : defaultOf(name);
    const { svg, sum } = schedulePicture(name, text);
    button.innerHTML = `${svg}<span class="sched-sum">${sum}</span>`;
    button.title = `${text}. Click to change it.`;
  });
}

/* Every changed setting in Settings has a reset button. */
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

/* Slider ends for Settings, where the useful range in Help doesn't suit a slider. */
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

const LONG_TEXT = new Set(["handoff_negative"]);  // prompts, shown as a box under the name in the cards

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
  } else if (f.name === "inpaint_file") {
    input = document.createElement("select");
    fillModelPick(input, current, "None", true, true);
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
  } else if (compact && LONG_TEXT.has(f.name)) {
    // Prompts get a box under their name, wide enough to read, as the main prompt has.
    input = document.createElement("textarea");
    Object.assign(input, { rows: 2, spellcheck: false, value: current ?? "" });
    wrap.classList.add("stacked");
    wrap.append(name, input);
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
  if (f.name === "inpaint_file") {
    const pick = document.createElement("button");
    Object.assign(pick, { type: "button", className: "default icon-button", title: "Browse for a model file on the PC that renders" });
    pick.setAttribute("aria-label", "Browse for a model file");
    pick.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>';
    pick.addEventListener("click", reporting(async () => {
      const path = await chooseCheckpoint(input.value);
      if (!path) return;
      await loadCheckpoints();
      pickInpaintModel(path);
    }));
    input.addEventListener("change", () => pickInpaintModel(input.value));
    cell.append(pick);
  }
  if (f.name === "partner_file") {
    const pick = document.createElement("button");
    Object.assign(pick, { type: "button", className: "default", textContent: "Browse" });
    pick.addEventListener("click", reporting(async () => {
      const path = await chooseCheckpoint(input.value);
      if (path) { input.value = path; input.dispatchEvent(new Event("change")); }
    }));
    cell.append(pick);
  }
  const reset = document.createElement("button");
  Object.assign(reset, { type: "button", className: "icon-link adv-reset", title: "Back to the default" });
  reset.setAttribute("aria-label", "Back to the default");
  reset.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>';
  reset.hidden = !(f.name in state.advanced);
  reset.addEventListener("click", () => resetField(f.name));
  cell.append(reset);
  wrap.append(cell);  // what it does shows on hover over its name, in the cards and in Settings
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

// Counts only Settings' own settings. Changes in the cards show in each card's title row.
const changedLabel = () => {
  const n = [...APP_FIELDS, ...INPAINT_FIELDS].filter((name) => name in state.advanced).length;
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
  for (const name of [...APP_FIELDS, ...INPAINT_FIELDS]) delete state.advanced[name];
  resetAdvancedPanel();
  save();
  updateFit();
}

function filterAdvanced() {
  const q = $("advSearch").value.trim().toLowerCase();
  $("namesBox").hidden = !!q && !"setting names plain official".includes(q);
  $("lookBox").hidden = !!q && !"accent color colour theme grid background".includes(q);
  $("foldersBox").hidden = !!q && !"save images folders upscalers".includes(q);
  $("modelsBox").hidden = !!q && !"models downloads upscalers".includes(q);
  $("networkBox").hidden = !!q && !"network share this pc render on another pc remote address code".includes(q);
  document.querySelectorAll(".app-fields").forEach((sec) => {
    let any = false;
    sec.querySelectorAll(".adv-field").forEach((row) => {
      const hit = !q || row.dataset.text.includes(q);
      row.style.display = hit ? "" : "none";
      any = any || hit;
    });
    sec.hidden = !any;
  });
  const boxes = ["namesBox", "lookBox", "foldersBox", "modelsBox", "networkBox"].map((id) => $(id));
  $("advNone").hidden = !q || boxes.some((b) => !b.hidden) || [...document.querySelectorAll(".app-fields")].some((s) => !s.hidden);
}

function setUpSettings() {
  $("advancedBtn").addEventListener("click", () => $("advanced").showModal());
  $("getModels").addEventListener("click", () => {
    $("advanced").showModal();
    $("modelsBox").open = true;
    $("modelsBox").scrollIntoView();
  });
  $("advClose").addEventListener("click", () => $("advanced").close());
  $("advReset").addEventListener("click", resetAdvanced);
  $("modelsBox").open = !!state.models_open;
  $("modelsBox").addEventListener("toggle", () => { state.models_open = $("modelsBox").open; save(); });
  $("advSearch").addEventListener("input", filterAdvanced);
  closeOnBackdrop($("advanced"));
}
