/* Prompts that change during a clip, each from the frame it starts at. */
"use strict";

/* The prompt box holds the first frame's prompt. Each change is its own box with the frame it starts at. The
   renderer reads them as lines that start with the frame number, such as "120: a city at night". */
const FRAME_LINE = /^\s*(\d+)\s*:\s*(.*)$/;

/* Moves any "120:" lines out of the prompt box into changes, for prompts saved or reused in that form. */
function splitPromptChanges() {
  const first = [], changes = new Map((state.prompt_changes || []).map((c) => [c.frame, c.text.split("\n")]));
  let at = 0;
  for (const line of String(state.prompt || "").split("\n")) {
    const m = line.match(FRAME_LINE);
    if (m) { at = Number(m[1]); if (at > 0 && !changes.has(at)) changes.set(at, []); if (!m[2].trim()) continue; }
    if (at === 0) first.push(m ? m[2] : line);
    else changes.get(at).push(m ? m[2] : line);
  }
  state.prompt = first.join("\n").replace(/\n+$/, "");
  state.prompt_changes = [...changes].sort((a, b) => a[0] - b[0]).map(([frame, lines]) => ({ frame, text: lines.join("\n").trim() }));
}

/* The prompt lines a render gets. A still uses only the first prompt. */
function promptLines() {
  const lines = state.prompt.split("\n").map((p) => p.trim()).filter(Boolean);
  if (!animating()) return lines;
  for (const change of [...(state.prompt_changes || [])].sort((a, b) => a.frame - b.frame)) {
    const own = change.text.split("\n").map((p) => p.trim()).filter(Boolean);
    own.forEach((line, i) => lines.push(i ? line : `${change.frame}: ${line}`));
  }
  return lines;
}

function showPromptChanges() {
  const box = $("promptChanges");
  const frames = Number(state.max_frames) || 0;
  box.replaceChildren(...(state.prompt_changes || []).map((change, i) => {
    const row = document.createElement("div");
    row.className = "change";
    const head = document.createElement("div");
    head.className = "change-head";
    const label = document.createElement("label");
    label.className = "field-name";
    label.textContent = "From frame";
    const frame = document.createElement("input");
    Object.assign(frame, { type: "number", min: 1, step: 1, value: change.frame, className: "readout frame" });
    frame.setAttribute("aria-label", "Frame this prompt starts at");
    label.htmlFor = frame.id = `change-${i}`;
    const late = document.createElement("span");
    late.className = "hint late";
    late.textContent = frames && change.frame >= frames ? `after the last frame (${frames})` : "";
    const remove = document.createElement("button");
    Object.assign(remove, { type: "button", className: "icon-link", title: "Remove this change" });
    remove.setAttribute("aria-label", "Remove this change");
    remove.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
    head.append(label, frame, late, remove);
    const text = document.createElement("textarea");
    Object.assign(text, { rows: 2, value: change.text, placeholder: "The prompt from this frame on" });
    text.setAttribute("aria-label", `Prompt from frame ${change.frame}`);
    text.addEventListener("input", () => { change.text = text.value; save(); });
    frame.addEventListener("change", (e) => {
      e.stopPropagation();
      const wanted = Math.max(1, Math.round(Number(frame.value) || 1));
      change.frame = state.prompt_changes.some((c) => c !== change && c.frame === wanted) ? wanted + 1 : wanted;
      state.prompt_changes.sort((a, b) => a.frame - b.frame);
      save();
      showPromptChanges();
    });
    remove.addEventListener("click", () => { state.prompt_changes.splice(state.prompt_changes.indexOf(change), 1); save(); showPromptChanges(); });
    row.append(head, text);
    return row;
  }));
}

function addPromptChange() {
  readControls();
  const changes = state.prompt_changes || (state.prompt_changes = []);
  const frames = Math.max(Number(state.max_frames) || 60, 2);
  const last = changes.length ? changes.at(-1).frame : 0;
  // Halfway through what is left of the clip, so a new change lands inside it.
  let frame = Math.max(last + 1, Math.round((last + frames) / 2));
  while (changes.some((c) => c.frame === frame)) frame += 1;
  changes.push({ frame, text: "" });
  save();
  showPromptChanges();
  $("promptChanges").querySelector(".change:last-child textarea").focus();
}

function setUpPrompts() {
  $("addPromptChange").addEventListener("click", addPromptChange);
}
