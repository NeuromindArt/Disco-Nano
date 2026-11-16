/* Images: the gallery of renders in their folders, what can be done with the one picked, and its right-click menu. */
"use strict";

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
  showOnStage(item.url, item.video);
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
    // the boxes it ended with, or the switch off and the boxes kept
    state.regions = s.regions?.boxes?.length ? { ...s.regions, on: true } : { ...regionsState(), on: false };
    fillControls();
    save();
    showRegionsButton(lastStatus?.task);
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
  if (isLocal()) items.push(null, { label: "Open folder", run: reporting(() => post("/api/open-output")) });
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

function setUpGallery() {
  $("galleryToggle").addEventListener("click", () => {
    state.gallery_closed = !state.gallery_closed;
    showGalleryPanel();
    save();
  });
  $("openFolder").addEventListener("click", reporting(() => post("/api/open-output")));
  $("newFolder").addEventListener("click", startNewFolder);
  $("reuseBtn").addEventListener("click", reuseSettings);
  $("useInitBtn").addEventListener("click", useAsStartImage);
  $("upscaleBtn").addEventListener("click", upscale);
  $("imageMenu").addEventListener("keydown", moveInImageMenu);
  document.addEventListener("pointerdown", (e) => { if (!$("imageMenu").contains(e.target)) closeImageMenu(); });
  document.addEventListener("scroll", closeImageMenu, true);
  addEventListener("resize", closeImageMenu);
  addEventListener("blur", closeImageMenu);
}
