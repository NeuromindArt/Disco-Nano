/* Starts the page: reads the saved settings and the server's list of settings, builds the panels and wires each part. */
"use strict";

async function init() {
  load();
  buildAccents();  // first, so the page never shows the wrong color
  ({ fields, terms } = await api("/api/schema"));
  byName = Object.fromEntries(fields.map((f) => [f.name, f]));
  followNewDefaults();
  buildControls();
  makeSwitches();
  buildAdvanced();
  buildCardFields();
  buildHelpSettings();  // after the cards, so each setting is listed under the card it's in
  showSchedulePreviews();
  applyNames();
  byName.tile_scale.options.filter((o) => o !== "Off").forEach((o) => $("tilePick").append(option(o)));
  for (const name of ["tile_start", "live_ghost", "animation_mode", "robust_critic", "route_prompts", "loop_mode"]) byName[name].options.forEach((o) => $(name).append(option(o)));
  await loadModels();
  showGalleryPanel();
  showControlsPanel();
  STACKED.addEventListener("change", showControlsPanel);
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

  setUpPanel();
  setUpRender();
  setUpStage();
  setUpRegions();
  setUpErase();
  setUpSettings();
  setUpHelp();
  setUpModels();
  setUpFolders();
  setUpGallery();
  setUpPrompts();
  setUpNetwork();
  setUpDrops();
  setUpCheckpoints();
  setUpCurve();
  setUpSchedule();

  await loadGallery();
  await poll();
  setInterval(poll, 1000);
  if ((await api("/api/first-run")).show) await openWelcome();
}

init().catch((e) => formError(`Could not start: ${e.message}`));
