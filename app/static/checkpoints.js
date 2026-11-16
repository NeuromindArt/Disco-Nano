/* Checkpoints: the model lists for Handoff and Inpaint, and the settings each checkpoint keeps. */
"use strict";

/* The list shows checkpoints in the checkpoint folders on the PC that renders. Browse, on that PC only, picks a
   file anywhere and adds its folder. */
let checkpointList = [];

async function loadCheckpoints() {
  try { checkpointList = (await api("/api/checkpoints")).checkpoints; } catch { checkpointList = []; }
  showCheckpoints();
  showIdle();  // which handoff settings apply depends on the model's kind
}

/* Fills a model list: the checkpoints grouped by the subfolder each is in, as ComfyUI sorts them, with `current`
   kept even when it isn't in a checkpoint folder. `none` labels the empty choice, which `always` keeps; otherwise it
   shows only until a model is chosen. */
/* A model list from the checkpoints, grouped by folder. Models made only for inpainting, such as Flux Fill, are
   left out of Handoff's list and come first in Inpaint's. */
function fillModelPick(select, current, none, always = true, inpainting = false) {
  const options = always || !current ? [option("", none)] : [];
  const groups = new Map();
  const made = document.createElement("optgroup");
  made.label = "Made for inpainting";
  for (const c of checkpointList) {
    if (c.inpaint_only && !inpainting) continue;
    const label = c.label ? `${c.name} (${c.label})` : c.sdxl ? `${c.name} (SDXL)` : c.name;
    if (c.inpaint_only) { made.append(option(c.path, label)); continue; }
    if (!c.group) { options.push(option(c.path, label)); continue; }
    if (!groups.has(c.group)) groups.set(c.group, document.createElement("optgroup"));
    groups.get(c.group).label = c.group;
    groups.get(c.group).append(option(c.path, label));
  }
  if (made.children.length) options.splice(options.length && !options[0].value ? 1 : 0, 0, made);
  options.push(...groups.values());
  if (current && !checkpointList.some((c) => c.path === current)) options.push(option(current, baseName(current).replace(/\.(safetensors|gguf)$/i, "")));
  select.replaceChildren(...options);
  select.value = current;
}

/* Inpaint's model, with the steps and guidance it was made for. */
function pickInpaintModel(path) {
  const own = { inpaint_file: path, ...(path && checkpointList.find((c) => c.path === path)?.inpaint) };
  for (const [name, value] of Object.entries(own)) {
    if (!byName[name]) continue;
    if (JSON.stringify(value) === JSON.stringify(byName[name].default)) delete state.advanced[name];
    else state.advanced[name] = value;
  }
  resetAdvancedPanel();
  save();
  if (eraseMode) showEraseBar(lastStatus?.task);
}

function showCheckpoints() {
  const select = $("handoffPick");
  const current = state.handoff_file || state.handoff_last || "";
  fillModelPick(select, current, "Choose a model", false);
  const inpaint = $("adv-inpaint_file");
  if (inpaint) fillModelPick(inpaint, lookValue("inpaint_file"), "None", true, true);
  $("handoffWhere").textContent = checkpointList.length ? ""
    : isLocal() ? "Browse to a checkpoint, or add checkpoint folders in Settings." : "Browse to a checkpoint on the PC that renders, or add checkpoint folders in Settings.";
}

function pickCheckpoint(path) {
  const first = path && !state.handoff_file;
  if (first && !Number(state.handoff_share)) $("handoff_share").value = state.handoff_share = 100;
  // From noise, Denoise is 100%, which would repaint all of the checkpoint's picture, so a handoff starts at 60%.
  if (first && !state.init_image && !Number(state.skip_steps)) {
    $("skip_steps").value = state.skip_steps = skipFor(60, state.steps);
    $("denoise").value = denoiseOf(state.skip_steps, state.steps);
  }
  // A checkpoint chosen before comes back with its own steps, CFG, sampler and the rest. Z-Image and Flux start
  // with the steps and CFG they were made for.
  const own = path && (state.handoff_models?.[path] || checkpointList.find((c) => c.path === path)?.defaults);
  if (own) {
    for (const [name, value] of Object.entries(own)) {
      if (!byName[name]) continue;
      if (JSON.stringify(value) === JSON.stringify(byName[name].default)) delete state.advanced[name];
      else state.advanced[name] = value;
    }
  }
  $("handoff_file").value = path;
  $("handoff_file").dispatchEvent(new Event("change", { bubbles: true }));
  if (own) { resetAdvancedPanel(); fillControls(); }
  showCheckpoints();
}

/* A checkpoint file on the PC that renders: Windows' own dialog there, or the page's browser from another PC.
   Its folder joins the checkpoint folders either way. */
async function chooseCheckpoint(start) {
  if (isLocal()) return (await post("/api/pick-checkpoint", { start: start || "" })).path;
  const folder = start ? start.replace(/[\\/][^\\/]*$/, "") : $("checkpointDirs").value.split("\n")[0].trim();
  const path = await browseRemote({ title: "Choose an SD 1.5, SDXL, Z-Image, Flux or Flux.2 Klein model", kind: "checkpoint", start: folder });
  return path ? (await post("/api/add-checkpoint", { path })).path : null;
}

const browseCheckpoint = reporting(async () => {
  const path = await chooseCheckpoint(state.handoff_file);
  if (!path) return;
  await loadCheckpoints();
  pickCheckpoint(path);
  loadFolders();
});

function setUpCheckpoints() {
  $("handoffBrowse").addEventListener("click", browseCheckpoint);
  $("handoffPick").addEventListener("change", (e) => {
    e.stopPropagation();
    if (e.target.value) pickCheckpoint(e.target.value);
  });
}
