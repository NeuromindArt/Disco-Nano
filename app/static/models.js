/* Models to download, in Settings and in the welcome window that opens on the first visit. */
"use strict";

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
  $("getModelsRow").hidden = painters.every((m) => m.downloaded);
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

const modelError = (message) => showError("modelError", message);

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
  size.textContent = m.found_at ? "Found" : m.downloaded ? "Installed" : sizeLabel(m.size_gb);
  if (m.found_at) size.title = `Used where it is: ${m.found_at}`;
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
  $("welcomeCheckpoints").value = f.checkpoint_dirs.join("\n");
  findCheckpointFolders($("welcomeCheckpoints"), $("welcomeCheckpointsHelp")).then(async ({ models }) => {
    if (!models.length) return;
    await loadModels();  // models found in other apps show as found, not as downloads
    showWelcomeModels(new Set(welcomeTicked().map((box) => box.dataset.model)));
  }).catch(() => {});
  showWelcomeModels();
  $("welcome").showModal();
}

/* The models to tick for download. `ticked` keeps the choices made so far when the list is shown again. */
function showWelcomeModels(ticked) {
  const main = byName.diffusion_model.default;
  const wanted = (m, start) => (ticked ? ticked.has(m.name) : start);
  $("welcomeUpscaler").replaceChildren(...catalog.filter((m) => m.kind === "upscaler").map((m) => welcomeRow(m, wanted(m, true))));
  $("welcomeModels").replaceChildren(...catalog.filter((m) => m.kind !== "upscaler").map((m) => welcomeRow(m, wanted(m, m.name === main || m.kind === "helper"))));
  showWelcomeTotal();
}

/* Looks once for the checkpoint folders of ComfyUI and the web UIs, and adds any it finds to a folder list. It
   runs only when asked: when the welcome screen opens, and from Find in other apps. The same search finds models
   those apps already have, so Nano needn't download them. Returns how many folders it added, and those models. */
async function findCheckpointFolders(box, note) {
  const before = note.textContent;
  note.textContent = "Looking for ComfyUI and Stable Diffusion web UI folders, and models they already have…";
  try {
    const { found, models } = await api("/api/find-checkpoint-folders");
    const have = new Set(box.value.split("\n").map((d) => d.trim().toLowerCase()).filter(Boolean));
    const added = found.filter((d) => !have.has(d.toLowerCase()));
    if (added.length) box.value = [box.value.trim(), ...added].filter(Boolean).join("\n");
    const notes = [added.length && `Found ${added.length} folder${added.length > 1 ? "s" : ""} in other apps.`,
      models.length && `Using ${models.join(", ")} from them, so ${models.length > 1 ? "those don't" : "it doesn't"} need downloading.`];
    note.textContent = notes.filter(Boolean).join(" ") || before;
    return { added: added.length, models };
  } catch (e) {
    note.textContent = before;
    throw e;
  }
}

async function finishWelcome() {
  const shown = (message) => showError("welcomeError", message);
  shown("");
  try {
    await applyFolders($("welcomeOutput").value, $("welcomeUpscalers").value.split("\n"), $("welcomeCheckpoints").value.split("\n"));
    const names = welcomeTicked().map((box) => box.dataset.model);
    if (names.length) await post("/api/downloads", { names });
    await post("/api/first-run");
    $("welcome").close();
    await poll();
  } catch (e) { shown(e.message); }
}

function setUpModels() {
  $("welcomeStart").addEventListener("click", finishWelcome);
  const welcomeError = (m) => showError("welcomeError", m);
  $("welcomeOutputBrowse").addEventListener("click", reporting(async () => {
    const path = await browseFolder("Save images to", $("welcomeOutput").value);
    if (path) $("welcomeOutput").value = path;
  }, welcomeError));
  $("welcomeCheckpointsBrowse").addEventListener("click", reporting(async () => {
    const path = await browseFolder("Add a checkpoint folder", "", "checkpoint-folder");
    if (path) addFolderLine($("welcomeCheckpoints"), path);
  }, welcomeError));
  $("welcomeUpscalersBrowse").addEventListener("click", reporting(async () => {
    const path = await browseFolder("Add an upscaler folder", "");
    if (path) addFolderLine($("welcomeUpscalers"), path);
  }, welcomeError));
}
