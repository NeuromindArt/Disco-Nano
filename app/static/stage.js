/* The stage: the picture or clip in the middle of the page, and the editors that open over it.

   Outside an editor the stage shows the latest preview, or what was picked in Images. An editor (Fix or Regions)
   takes the stage while it's open: what the gallery picks meanwhile is kept and shows when the editor closes, and
   only a render the editor paused can move the picture under it. The keys and the right-click go to the editor
   while it's open and nothing else, such as a dialog or the image menu, is on top of it.

   An editor is { name, followsRender, picture, keys(e), beforeRender() }:
   - followsRender: the editor is over a render it paused, so that render's previews still show.
   - picture: what the editor puts on the stage, such as the start image, or null to keep what was showing.
   - keys(e): the editor's own keys.
   - beforeRender(): called by Render, returns why a render can't start yet, or nothing. */
const stage = {
  shows: { src: "", video: "" },  // what the stage shows outside an editor; empty until there is something
  editor: null,
  ending: false,  // the page has ended an editor's pause, and the server hasn't shown the render going again yet
};

function stageEditor(name) {
  return name ? stage.editor?.name === name : !!stage.editor;
}

/* Puts a picture or a clip on the stage. A preview passes `preview`, so a render's own editor still sees it. */
function showOnStage(src, video, { preview = false } = {}) {
  stage.shows = { src: src || "", video: video || "" };
  if (stage.editor && !(preview && stage.editor.followsRender)) return;  // shows when the editor closes
  paintStage(stage.shows);
}

function paintStage({ src, video }) {
  const img = $("stageImg"), player = $("stageVideo");
  if (video) {
    if (player.getAttribute("src") !== video) { player.src = video; player.play().catch(() => {}); }
  } else {
    player.pause();
    player.removeAttribute("src");
    if (src) { if (img.getAttribute("src") !== src) img.src = src; } else img.removeAttribute("src");
  }
  player.hidden = !video;
  img.hidden = !!video || !src;
  $("stageEmpty").hidden = !!(src || video) || !!stage.editor;
}

/* Opens `editor` over the stage, unless another one is open. Returns whether it opened. */
function openStageEditor(editor) {
  if (stage.editor) return false;
  stage.editor = editor;
  document.querySelector(".canvas").classList.add("editing");
  closeImageMenu();
  paintStage(editor.picture ? { src: editor.picture, video: "" } : stage.shows);
  return true;
}

/* Changes the picture the open editor shows, such as a try in place of the start image. */
function showEditorPicture(src) {
  if (!stage.editor) return;
  stage.editor.picture = src;
  paintStage({ src, video: "" });
}

/* Closes the editor called `name`, and the stage shows what it would have without it. */
function closeStageEditor(name) {
  if (!stageEditor(name)) return;
  stage.editor = null;
  document.querySelector(".canvas").classList.remove("editing");
  paintStage(stage.shows);
}

/* Ends the pause an editor held, once it has closed: `send` posts what the editor made and the resume. A status
   from before the server had them still shows the pause, and isn't one to open an editor for. */
async function endPause(send) {
  stage.ending = true;
  try { await send(); } catch (e) { stage.ending = false; formError(e.message); }
}

/* Called with every status, before the editors see it: whether `t` is a pause an editor should open over, which
   it is unless it's the one this page is ending. */
function heldPause(t) {
  const paused = !!t?.active && t.kind === "render" && t.state === "paused";
  if (!paused) stage.ending = false;
  return paused && !stage.ending;
}

/* Something else on top of the stage, which keeps the keys to itself: an open dialog or the image menu. */
function stageCovered() {
  return !!document.querySelector("dialog[open]") || !$("imageMenu").hidden;
}

/* Render asks the open editor first. Returns why the render can't start yet, or nothing. */
function stageBeforeRender() {
  return stage.editor?.beforeRender?.() || "";
}

function setUpStage() {
  document.addEventListener("keydown", (e) => {
    if (stage.editor && !e.defaultPrevented && !stageCovered()) stage.editor.keys(e);
  });
  // The image menu is for a finished image, not the editor's picture.
  document.querySelector(".canvas").addEventListener("contextmenu", (e) => { if (stage.editor) e.preventDefault(); else openImageMenu(e); });
}

/* Follows one pointer from its pointerdown `e` on `el` until it lifts or is lost: `move` with each move, then `end`. */
function followPointer(el, e, move, end = () => {}) {
  try { el.setPointerCapture(e.pointerId); } catch { /* a pointer that is already gone */ }
  const stop = () => {
    el.removeEventListener("pointermove", move);
    el.removeEventListener("pointerup", stop);
    el.removeEventListener("pointercancel", stop);
    end();
  };
  el.addEventListener("pointermove", move);
  el.addEventListener("pointerup", stop);
  el.addEventListener("pointercancel", stop);
}
