/* What every part of the page shares: the settings it keeps in the browser, talking to the server, error lines,
   setting names and the accent color. Each feature has a file of its own, and main.js starts them all. */
"use strict";

const $ = (id) => document.getElementById(id);
const STORE = "dd-ui-v1";
const SIZES = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280];
const CLIPS = [["ViTB32", "ViT-B/32"], ["ViTB16", "ViT-B/16"], ["ViTL14", "ViT-L/14"], ["RN50", "RN50"]];

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
/* The handoff checkpoint's own settings, which each checkpoint keeps: choosing it again brings them back. */
const HANDOFF_MODEL_KEYS = ["handoff_steps", "handoff_guidance", "handoff_sampler", "handoff_karras", "handoff_clip_skip", "handoff_few_steps", "handoff_negative"];

function save() {
  if (state.handoff_file && fields.length) {
    const own = Object.fromEntries(HANDOFF_MODEL_KEYS.filter((n) => byName[n]).map((n) => [n, lookValue(n)]));
    state.handoff_models = { ...state.handoff_models, [state.handoff_file]: own };
  }
  try { localStorage.setItem(STORE, JSON.stringify(state)); } catch { /* storage full or blocked: this visit still works */ }
}

/* The main panel's values are saved in the browser, so a new default would never reach anyone who has opened
   Nano before. A value still at the default it was saved under moves to the new default, and one changed by
   hand stays. QUICK_DEFAULTS_BEFORE stands in for saved settings that have no record of their defaults. */
const QUICK_DEFAULTS_BEFORE = { clip_guidance_scale: 20000 };

function followNewDefaults() {
  const seen = state.quick_defaults || QUICK_DEFAULTS_BEFORE;
  const now = {};
  for (const name of QUICK) {
    const fresh = byName[name]?.default;
    if (fresh === undefined) continue;
    now[name] = fresh;
    if (name in seen && name in state && String(state[name]) === String(seen[name]) && String(seen[name]) !== String(fresh)) state[name] = fresh;
  }
  state.quick_defaults = now;
  save();
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}
/* Keeps a setting in state.advanced only while it differs from `standard`, the setting's default unless given, so
   a default that changes reaches it. */
function setAdvanced(name, value, standard = byName[name]?.default) {
  if (JSON.stringify(value) === JSON.stringify(standard)) delete state.advanced[name];
  else state.advanced[name] = value;
}

const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });

/* An error line under what it's about: shows `msg`, or hides with none. */
function showError(id, msg) {
  $(id).textContent = msg || "";
  $(id).hidden = !msg;
}

/* A click handler that shows what went wrong on `error`'s line, as every button's should. */
const reporting = (run, error = formError) => async (...args) => {
  try { await run(...args); } catch (e) { error(e.message); }
};

function formError(msg) {
  showError("formError", msg);
  if (msg && state.controls_closed) { state.controls_closed = false; showControlsPanel(); save(); }  // so the reason shows
}

/* A dialog that closes on a click on its backdrop, outside its box. */
function closeOnBackdrop(dialog) {
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
}

function fmtTime(s) {
  if (s == null) return "";
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${Math.round(s % 60)}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

/* The last part of a path, and a model file's name without its extension. */
const baseName = (path) => String(path || "").split(/[\\/]/).filter(Boolean).pop() || "";
const modelName = (path) => baseName(path).replace(/\.(safetensors|gguf)$/i, "");

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
  // The grid behind the picture, or plain gray. Set here, with the accent, so the page never shows the wrong one.
  // It starts off: only a grid switched on in Settings, which saves no_grid as false, shows.
  const grid = $("canvasGrid");
  grid.checked = state.no_grid === false;
  document.body.classList.toggle("no-grid", !grid.checked);
  grid.addEventListener("change", () => { state.no_grid = !grid.checked; document.body.classList.toggle("no-grid", state.no_grid); save(); });
}
