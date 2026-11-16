/* Folders: where images are saved, and the upscaler and checkpoint folders, chosen with Windows' dialog or, from
   another PC, with the page's own browser of the PC that renders. */
"use strict";

// ---- folders ----
function folderError(msg) {
  showError("folderError", msg);
}

/* Adds a folder as a line of its own to a list of folders, one per line. */
function addFolderLine(list, path) {
  list.value = [list.value.trim(), path].filter(Boolean).join("\n");
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

function setUpFolders() {
  $("outputDir").addEventListener("change", saveFolders);
  $("upscaleDirs").addEventListener("change", saveFolders);
  $("outputBrowse").addEventListener("click", reporting(async () => {
    const path = await browseFolder("Save images to", $("outputDir").value);
    if (path) { $("outputDir").value = path; saveFolders(); }
  }, folderError));
  $("upscaleBrowse").addEventListener("click", reporting(async () => {
    const path = await browseFolder("Add an upscaler folder", "", "upscaler");
    if (path) { addFolderLine($("upscaleDirs"), path); saveFolders(); }
  }, folderError));
  $("checkpointDirs").addEventListener("change", saveFolders);
  $("checkpointFind").addEventListener("click", async () => {
    const button = $("checkpointFind");
    button.disabled = true;
    $("checkpointFound").textContent = "";
    try {
      const { added, models } = await findCheckpointFolders($("checkpointDirs"), $("checkpointFound"));
      if (added) await saveFolders();
      if (models.length) await loadModels();
      if (!added && !models.length) $("checkpointFound").textContent = "No other folders found.";
    } catch (e) { folderError(e.message); }
    button.disabled = false;
  });
  $("checkpointBrowse").addEventListener("click", reporting(async () => {
    const path = await browseFolder("Add a checkpoint folder", "", "checkpoint-folder");
    if (path) { addFolderLine($("checkpointDirs"), path); saveFolders(); }
  }, folderError));
  $("browseUp").addEventListener("click", () => browsing && browseTo(browsing.parent || ""));
  $("browsePath").addEventListener("change", () => browseTo($("browsePath").value.trim()));
  $("browsePick").addEventListener("click", () => browsing?.path && finishBrowse(browsing.path));
  $("browseCancel").addEventListener("click", () => finishBrowse(null));
  $("browseDialog").addEventListener("close", () => { if (browsing) finishBrowse(null); });
}
