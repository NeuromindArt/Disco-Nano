/* Drop zones: the start image, the video, the guide images and image prompts, dropped, picked or dragged from Images. */
"use strict";

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

/* A drop zone is busy while what was dropped on it uploads. It takes nothing else until then. */
const dropBusy = (zone) => zone.classList.contains("busy");
const dropClosed = (zone) => !!zone.closest(".blocked") || dropBusy(zone);

/* Runs `work` with `zone` busy, shows what went wrong, and then `after`, which shows what the zone holds now. */
async function whileBusy(zone, work, after) {
  if (dropBusy(zone)) return;
  zone.classList.add("busy");
  formError("");
  try { await work(); } catch (e) { formError(e.message); } finally {
    zone.classList.remove("busy");
    after();
  }
}

function wireDrop(zone, { input, onFiles, onImage, onClear }) {
  const open = (e) => {
    if (e.target.closest(".drop-clear, .drop-erase") || dropClosed(zone)) return;
    input.click();
  };
  zone.addEventListener("click", open);
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(e); } });
  input.addEventListener("change", () => { if (input.files.length) onFiles([...input.files]); input.value = ""; });
  zone.querySelector(".drop-clear")?.addEventListener("click", (e) => { e.stopPropagation(); onClear(); });
  zone.addEventListener("dragover", (e) => {
    const types = e.dataTransfer.types;
    if (dropClosed(zone) || !(types.includes("Files") || (onImage && types.includes(DRAG_TYPE)))) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    zone.classList.add("over");
  });
  zone.addEventListener("dragleave", (e) => { if (!zone.contains(e.relatedTarget)) zone.classList.remove("over"); });
  zone.addEventListener("drop", async (e) => {
    zone.classList.remove("over");
    const rel = e.dataTransfer.getData(DRAG_TYPE);
    if (!rel && !e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();  // the browser would open the file in place of the page
    if (dropClosed(zone)) return;
    if (rel && onImage) { onImage(rel); return; }
    const files = await droppedFiles(e.dataTransfer);
    if (files.length) onFiles(files);
  });
}

/* Shows what a drop zone holds: a picture or video, its name, or the empty prompt. `busy` is what it is doing. */
function showDrop(zone, { url, name, busy } = {}) {
  const full = !!(url || name || busy);
  zone.classList.toggle("full", full);
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
  if (!dropBusy(zone)) showDrop(zone, { url: state.init_url, name: state.init_image ? state.init_name : "" });
  if (state.init_image && !state.init_url) {
    previewOf(state.init_image).then((p) => {
      if (!state.init_image || dropBusy(zone)) return;
      if (p?.url) { state.init_url = p.url; save(); showDrop(zone, { url: p.url, name: state.init_name }); }
      if (p?.missing) showDrop(zone, { name: `${state.init_name} is missing. Drop it again.` });
    });
  }
  $("initOptions").hidden = !state.init_image;
  if (typeof showEraseButton === "function") showEraseButton(lastStatus?.task);  // the start image's Erase button
  $("startBlock").hidden = animating() && state.animation_mode === "Video Input";
  $("matchImage").disabled = !state.init_size;
  for (const btn of document.querySelectorAll("[data-init-fit]")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.initFit === state.init_fit));
  }
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

/* Uploads a file into a drop zone, showing how far it has got. */
function uploadInto(zone, url, file) {
  showDrop(zone, { busy: "Uploading" });
  return upload(`${url}${url.includes("?") ? "&" : "?"}name=${encodeURIComponent(file.name)}`, file,
    (f) => showDrop(zone, { busy: `Uploading ${Math.round(f * 100)}%` }));
}

const isImageFile = (f) => /\.(png|jpe?g|webp)$/i.test(f.name);

function uploadInit(files) {
  const zone = $("initDrop");
  return whileBusy(zone, async () => setInit(await uploadInto(zone, "/api/init-image", files.find(isImageFile) || files[0])), showInit);
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
  if (dropBusy(zone)) return;
  showDrop(zone, { url: path && state.video_url ? state.video_url : "", name: baseName(path) });
  if (path && !state.video_url) {
    previewOf(path).then((p) => {
      if (state.video_init_path !== path || dropBusy(zone)) return;
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

function uploadVideo(files) {
  const zone = $("videoDrop");
  let made = null;
  return whileBusy(zone, async () => { made = await uploadInto(zone, "/api/upload-video", files[0]); },
    () => (made ? setVideo(made.path, made.url) : showVideo()));
}

// ---- guide images ----
/* Ghost and style images live with the other Experimental settings, as full paths. Their boxes upload a copy and
   set the path. Image prompts are a folder: a drop of many images, or of a folder, becomes a new folder of them. */
const LAB_IMAGES = ["ghost_image", "style_image", "control_image"];

function showLabImages() {
  let count = 0;
  for (const name of LAB_IMAGES) {
    const path = state.advanced[name] || "";
    const zone = document.querySelector(`[data-drop="${name}"]`);
    if (path && name !== "control_image") count += 1;
    if (dropBusy(zone)) continue;
    const known = PREVIEW_CACHE[path];
    showDrop(zone, { url: known?.url || "", name: baseName(path) });
    if (path && !known) previewOf(path).then((p) => { if (p) showLabImages(); });
  }
  showImagePrompts();
  if (state.image_prompts && Number(lookValue("image_prompt_weight"))) count += 1;  // weight 0 turns them off
  $("sum-guides").textContent = count ? `${count} on` : "None";
  showOpts();
}

function setLabImage(name, path) {
  if (path) state.advanced[name] = path;
  else delete state.advanced[name];
  resetAdvancedPanel();
  fillControls();
  save();
}

function uploadLabImage(name, files) {
  const zone = document.querySelector(`[data-drop="${name}"]`);
  let made = null;
  return whileBusy(zone, async () => {
    made = await uploadInto(zone, "/api/init-image", files.find(isImageFile) || files[0]);
    PREVIEW_CACHE[made.path] = { url: made.url, size: made.size };
  }, () => (made ? setLabImage(name, made.path) : showLabImages()));
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
  if (dropBusy(zone)) return;
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

function uploadPromptImages(files) {
  const images = files.filter(isImageFile).slice(0, 200);
  const zone = document.querySelector('[data-drop="image_prompts"]');
  if (!images.length) { formError("Image prompts take PNG, JPG or WEBP images."); return; }
  const group = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
  let folder = "";  // the images that made it before anything went wrong are kept
  return whileBusy(zone, async () => {
    zone.classList.add("full");
    zone.querySelector(".drop-empty").hidden = true;
    for (const [i, file] of images.entries()) {
      zone.querySelector(".drop-count").textContent = `${i + 1} of ${images.length}`;
      folder = (await upload(`/api/prompt-images?group=${group}&name=${encodeURIComponent(file.name)}`, file)).folder;
    }
  }, () => (folder ? setImagePrompts(folder) : showImagePrompts()));
}

function setUpDrops() {
  wireDrop($("initDrop"), { input: $("initFile"), onFiles: uploadInit, onImage: initFromGallery, onClear: clearInit });
  wireDrop($("videoDrop"), { input: $("videoFile"), onFiles: uploadVideo, onClear: () => setVideo("", "") });
  for (const name of LAB_IMAGES) {
    wireDrop(document.querySelector(`[data-drop="${name}"]`), { input: $(`${name}File`), onFiles: (files) => uploadLabImage(name, files),
      onImage: (rel) => labImageFromGallery(name, rel), onClear: () => setLabImage(name, "") });
  }
  wireDrop(document.querySelector('[data-drop="image_prompts"]'), { input: $("image_promptsFile"), onFiles: uploadPromptImages, onClear: () => setImagePrompts("") });
  document.querySelectorAll("[data-init-fit]").forEach((btn) => btn.addEventListener("click", () => {
    state.init_fit = btn.dataset.initFit;
    state.match_area = 0;
    matchFrame();
    fillControls();
    save();
  }));
}
