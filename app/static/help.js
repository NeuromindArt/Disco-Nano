/* Help: each setting's help shown on hover, plain or official names, and the Help window's list of settings. */
"use strict";

/* The setting a label names: the control it is for, or the one inside it. */
function settingOf(label) {
  let id = label.htmlFor || label.querySelector("input:not([type=hidden]), select")?.id || "";
  if (id.startsWith("adv-")) id = id.slice(4);
  if (id === "denoise") id = "skip_steps";
  if (id === "tilePick") id = "tile_scale";
  return byName[id] ? id : null;
}

/* Resting the pointer on a setting's name shows its help, unless that is turned off in Settings. */
function showHoverHelp() {
  const on = state.hover_help !== false;
  if ($("hoverHelp")) $("hoverHelp").checked = on;
  document.body.classList.toggle("hover-help", on);
  document.querySelectorAll(".controls label, .controls .guide > .field-name, #advanced label, [data-help]").forEach((label) => {
    const name = label.matches("label") ? settingOf(label) : label.dataset.words?.match(/^\{(\w+)\}$/)?.[1];
    const help = label.dataset.help || (name && byName[name]?.help);
    if (!help) return;
    if (on) label.title = worded(help); else label.removeAttribute("title");
  });
}

/* Help for something whose meaning changes, such as the chosen model, shown on hover like the rest. */
function setHelp(el, text) {
  if (text) el.dataset.help = text; else delete el.dataset.help;
  if (text && state.hover_help !== false) el.title = worded(text); else el.removeAttribute("title");
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

/* Every setting, listed under the card it's in, in the cards' order, then the Settings window's. */
function buildHelpSettings() {
  const cards = [...document.querySelectorAll(".controls [data-card]")];
  const TOP = "Prompt and start image";  // above the cards
  const sections = new Map([[TOP, []], ...cards.map((c) => [c.querySelector(".card-title").textContent.trim(), []])]);
  sections.get("Size")?.push(
    { label: "Width", kind: "number", low: 256, default: 768, high: 1280, help: "Image width in pixels, in steps of 64." },
    { label: "Height", kind: "number", low: 256, default: 512, high: 1280, help: "Image height in pixels, in steps of 64." });
  sections.set("Settings", []);
  const other = [];
  for (const f of fields) {
    const el = (MOTION[f.name] && $("motionRows")) || $(f.name) || document.querySelector(`.adv-field[data-name="${f.name.toLowerCase()}"]`)
      || document.querySelector(`[data-drop="${f.name}"]`) || (f.name === "prompt_fade" ? $("addPromptChange") : null);
    const card = el?.closest("[data-card]");
    const where = card ? card.querySelector(".card-title").textContent.trim() : !el ? null : el.closest("#advanced") ? "Settings" : TOP;
    (where && sections.has(where) ? sections.get(where) : other).push(f);
  }
  if (other.length) sections.set("Not shown in the app", other);
  const box = $("helpSettings");
  for (const [title, list] of sections) {
    if (!list.length) continue;
    const heading = document.createElement("h4");
    heading.textContent = title;
    box.append(heading, helpTable(list));
  }
}

function setUpHelp() {
  document.querySelectorAll("[data-names]").forEach((btn) =>
    btn.addEventListener("click", () => setNames(btn.dataset.names)));
  $("showWelcome").addEventListener("click", () => { $("help").close(); openWelcome(); });
  $("helpBtn").addEventListener("click", () => $("help").showModal());
  $("helpClose").addEventListener("click", () => $("help").close());
  $("hoverHelp").addEventListener("change", () => { state.hover_help = $("hoverHelp").checked; save(); showHoverHelp(); });
  closeOnBackdrop($("help"));
}
