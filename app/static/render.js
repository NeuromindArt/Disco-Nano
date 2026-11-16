/* Rendering: Render and Stop, the preview, the progress of each task and the status from the server. */
"use strict";

async function startRender() {
  formError("");
  const waiting = stageBeforeRender();  // an editor open over the stage keeps or asks about its work first
  if (waiting) { formError(waiting); return; }
  readControls();
  const prompts = promptLines();
  if (!prompts.length || /^\d+\s*:/.test(prompts[0])) { formError("Add a prompt first."); $("prompt").focus(); return; }
  if (animating() && state.prompt_changes?.some((c) => !c.text.trim())) { formError("A prompt change is empty. Write its prompt or remove it."); return; }
  if (!CLIPS.some(([n]) => state[n]) && !Object.keys(state.advanced).length) { formError("Choose at least one CLIP model."); return; }
  if (state.skip_steps >= state.steps) { formError(worded("{Denoise must be above 0%.|skip_steps must be lower than steps.}")); return; }
  if (plan?.conflicts?.length) { formError(worded(plan.conflicts[0])); return; }

  if (animating() && state.animation_mode === "Video Input" && !state.video_init_path) { formError("Drop a video on the Video box for Video Input."); return; }
  const settings = currentSettings();
  const resume_clip = carryOn;
  carryOn = null;
  try {
    renderPlan = plan;
    await post("/api/render", { prompts, settings, batch_name: state.batch_name, resume_clip, options: { init_fit: state.init_fit } });
    shownTask = null; previewRev = -1;
    poll();
  } catch (e) { formError(e.message); }
}

/* Loads a preview off-screen and shows it once it's whole, so one that fails to load, or a newer one that
   overtakes it, never blanks the stage. */
function showPreview(url, rev) {
  const next = new Image();
  next.onload = () => { if (rev === previewRev) showOnStage(url, "", { preview: true }); };
  next.src = url;
}

function renderTask(t) {
  const active = !!t?.active;
  showRegionsTask(t);
  showEraseTask(t);
  $("renderBtn").disabled = active;
  $("renderMini").hidden = active;
  $("upscaleBtn").disabled = active;
  $("stopBtn").hidden = !active;
  $("stopMini").hidden = !active;
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
      showPreview(`/api/preview/${t.id}?r=${previewRev}`, previewRev);
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
  } catch { showConnected(false); }  // the next tick retries
}

/* Only whether the server answers: what it is doing shows under the progress bar. */
function showConnected(connected) {
  $("statusState").textContent = connected ? "Connected" : "Not connected";
  $("statusState").classList.toggle("offline", !connected);
}

function showStatus(s) {
  showMemory();
  showDisk(s.disk);
  showLocal(s.local !== false);
  showRemote(s);
  showConnected(true);
  $("engineWarn").hidden = !s.other_gpu_app;
  updateFit();
}

function setUpRender() {
  $("renderBtn").addEventListener("click", startRender);
  const stop = reporting(() => post("/api/stop"));
  $("stopBtn").addEventListener("click", stop);
  $("renderMini").addEventListener("click", startRender);
  $("stopMini").addEventListener("click", stop);
}
