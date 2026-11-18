/* Looks and clip styles: sets of settings picked by name, and the views plan that shows which steps lay out shapes
   and which paint detail. */
"use strict";

/* A look sets how the prompt shapes the picture, apart from its size. Settings it leaves out go back to
   their defaults. */
const LOOKS = {
  balanced: { label: "Shapes, then detail", hint: "The first 40% of the steps lay out the scene, and the rest paint detail. Strong single subjects. The default.", values: {} },
  detail: { label: "Mostly detail", hint: "Most steps paint detail. Crisp textures and busy scenes, and a start image keeps more of its shapes.", values: { views_shapes: 10 } },
  shapes: { label: "Mostly shapes", hint: "Most steps lay out shapes. Bold, simple scenes with softer detail.", values: { views_shapes: 80 } },
  classic: {
    label: "Classic 2022", hint: "Settings from 2022 Disco renders. Dreamy and loose, and about twice as long.",
    values: { clip_guidance_scale: 5000, use_secondary_model: true, tv_scale: 0, sat_scale: 0, cutn_batches: 5, clamp_max: 0.05, cut_ic_pow: "[1]*1000" },
  },
};
// The two views schedules that set where shapes give way to detail belong to every look, so a look's hint holds.
const LOOK_KEYS = [...new Set(["cut_overview", "cut_innercut", ...Object.values(LOOKS).flatMap((l) => Object.keys(l.values))])];

/* A clip style sets how a Video Input clip is repainted: how much of each frame is painted, how much of the last
   frame carries over, and how the views spread over the painted steps. Like a look, one changed by hand shows as
   Custom. */
const CLIP_STYLES = {
  paint: {
    label: "Paint over", hint: "Keeps the video's shapes and paints over them. Abstract and painterly, with Disco's flicker.",
    values: { frames_skip_steps: 40, video_blend: 0, views_fit: "Start images" },
  },
  reimagine: {
    label: "Reimagine", hint: "Lets the prompt turn what's in the video into what it describes. Looser, and further from the video.",
    values: { frames_skip_steps: 40, video_blend: 0, views_fit: "Start images and clip frames" },
  },
  build: {
    label: "Build up", hint: "Carries some of each painted frame into the next, so the clip changes less from frame to frame. Closer to the video.",
    values: { frames_skip_steps: 55, video_blend: 0.3, views_fit: "Start images" },
  },
};
const STYLE_KEYS = [...new Set(Object.values(CLIP_STYLES).flatMap((l) => Object.keys(l.values)))];

// ---- looks ----
/* A setting's default for the chosen model. The pixel art and watercolor models bring views schedules of their
   own, which the renderer uses for any it isn't given. */
function defaultOf(name) {
  const own = catalog.find((m) => m.name === state.diffusion_model)?.cuts;
  return own && name in own ? own[name] : byName[name]?.default;
}

function lookValue(name) {
  if (HANDLED.has(name)) return state[name] ?? defaultOf(name);
  return name in state.advanced ? state.advanced[name] : defaultOf(name);
}

/* The look whose settings all match, so a look changed by hand shows as Custom. The views schedules count too:
   a look's split of shapes and detail holds only for the schedules as they come. */
const sameValue = (a, b) => JSON.stringify(typeof b === "number" ? Number(a) : a) === JSON.stringify(b);

function currentPreset(presets, keys) {
  const match = Object.entries(presets).find(([, l]) =>
    keys.every((name) => byName[name] && sameValue(lookValue(name), l.values[name] ?? defaultOf(name))));
  return match ? match[0] : "custom";
}

/* Python's round, which takes halves to the even side, so step counts match the renderer's. */
function pyRound(x) {
  const r = Math.round(x);
  return Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 ? r - 1 : r;
}

/* Frame skip % for every frame of a clip when Frame skip keyframes are set, as the server works them out, or null.
   The first call for new keyframes asks the server and redraws what depends on them when it answers. */
let skipKeys = { key: "", values: null };
function keyedSkips() {
  const text = String(lookValue("frame_skip_keys") || "").trim();
  if (!animating() || !text) return null;
  const frames = Math.max(Number(state.max_frames) || 1, 1);
  const key = `${text}|${frames}|${curveShape()}`;
  if (skipKeys.key !== key) {
    skipKeys = { key, values: null };
    api(`/api/curve?text=${encodeURIComponent(text)}&frames=${frames}&shape=${encodeURIComponent(curveShape())}`).then((r) => {
      if (skipKeys.key !== key || r.error) return;
      skipKeys.values = r.values.map((v) => Math.min(Math.max(v, 0), 99));
      showLook();
      showSchedulePreviews();
      showTime();
    }).catch(() => {});
  }
  return skipKeys.values;
}

/* Frame skip as steps for a clip's frames after the first, averaged when keyframes set it frame by frame. */
function frameSkip(steps) {
  const toSteps = (share) => Math.min(pyRound((steps * share) / 100), steps - 1);
  const keyed = keyedSkips();
  if (!keyed?.length) return toSteps(Number(state.frames_skip_steps));
  const later = keyed.length > 1 ? keyed.slice(1) : keyed;
  return Math.round(later.reduce((s, v) => s + toSteps(v), 0) / later.length);
}

/* Where Disco starts after a handoff checkpoint paints from step `skip`. At Handoff 100% the checkpoint finishes its
   picture and Disco repaints Denoise of it, so Disco starts at its own skip. Lower, the checkpoint stops partway and
   Disco carries on from there: from noise, measured with 200 steps, 30% left 98 of 200 and 50% left 69; from a
   start, roughly, as the two noise schedules differ. */
function afterHandoffSkip(skip) {
  const steps = Math.max(Number(state.steps) || 1, 1);
  if (!handingStart()) return skip;
  const share = Math.min(Math.max(Number(state.handoff_share) / 100, 0), 1);
  if (share >= 1) return skip;
  if (skip > 0) {
    const repaints = (Number(state.handoff_start_denoise) || 0) / 100 || (steps - skip) / steps;
    return Math.min(Math.max(skip, steps - Math.round(steps * repaints * (1 - share))), steps - 1);
  }
  const points = [[0, 1], [0.3, 0.49], [0.5, 0.345], [1, 0.005]];
  let left = points.at(-1)[1];
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1], [x1, y1] = points[i];
    if (share <= x1) { left = y0 + ((share - x0) / (x1 - x0)) * (y1 - y0); break; }
  }
  return Math.min(steps - Math.round(steps * left), steps - 1);
}

/* The checkpoint paints the start of every image or frame. */
function handingStart() { return !!state.handoff_file && Number(state.handoff_share) > 0; }

function currentLook() { return currentPreset(LOOKS, LOOK_KEYS); }

/* Which steps lay out shapes and which paint detail, worked out as the renderer does. A start image, or a clip's
   Frame skip, skips the first steps. Shapes share spreads the views schedules' first 400 of 1000 points over its
   share of the steps, and Fit views over only the painted ones. Whether a step lays out shapes or paints detail
   then comes from the schedules themselves, so editing them moves the split too. */
const VIEW_SCHEDULES = ["cut_overview", "cut_innercut", "cut_ic_pow", "cut_icgray_p"];
const SHAPES_END = 400;

function viewPlan() {
  const steps = Math.max(Number(state.steps) || 0, 1);
  const clip = animating();
  const keyed = clip && keyedSkips()?.length;
  const skip = clip ? frameSkip(steps)
    : Math.min(Number(state.skip_steps) || 0, steps - 1);
  // A handoff model paints the start after the views are fitted, so Disco's own steps begin later still.
  const disco = afterHandoffSkip(skip);
  const why = keyed ? "Frame skip keyframes, on average" : clip ? `Frame skip ${state.frames_skip_steps}%`
    : `Denoise ${denoiseOf(skip, steps)}%`;
  const mode = lookValue("views_fit");
  const fit = mode === true || mode === "Start images and clip frames" || (!clip && mode === "Start images");
  const shapes = Math.min(Math.max(Number(lookValue("views_shapes") ?? 40) / 100, 0), 1);
  const begin = fit ? Math.min(Math.round((1000 * skip) / steps), 999) : 0;
  const viewAt = (at) => {
    const done = Math.min(Math.max((at - begin) / (1000 - begin), 0), 1);
    return done < shapes ? (done / shapes) * SHAPES_END : SHAPES_END + ((done - shapes) / Math.max(1 - shapes, 1e-6)) * (1000 - SHAPES_END);
  };
  const shape = shapePoints();
  const kinds = [];
  for (let j = 0; j < steps; j++) {
    // Before Disco starts: kept from a start image, or from the checkpoint's picture when one paints the start.
    if (j < disco) kinds.push(handingStart() || j >= skip ? "handoff" : "skip");
    else kinds.push(shape[Math.min(Math.floor(viewAt(Math.floor((1000 * j) / steps))), 999)] ? "shape" : "detail");
  }
  const shaping = kinds.filter((k) => k === "shape").length;
  const renderFirst = !clip && tileGrid() > 1 && !state.init_image && state.tile_start === "Render first";
  return { steps, skip, disco, shaping, kinds, fit, clip, keyed, renderFirst, viewAt, why };
}

/* Each of the 1000 points of the views schedules, as drawn: true where it lays out shapes, which is while
   whole-image views at least match close-ups, and false where close-ups outnumber them and paint detail. */
function shapePoints() {
  const points = (name) => {
    const levels = parseSchedule(lookValue(name)) || parseSchedule(defaultOf(name)) || [{ v: 0, n: 1000 }];
    return levels.flatMap((l) => Array(l.n).fill(l.v));
  };
  const whole = points("cut_overview"), close = points("cut_innercut");
  return whole.map((w, i) => w > 0 && w >= close[i]);
}

function showViews() {
  const box = $("viewsSplit");
  if (!box || !byName.views_fit) return;
  const { steps, kinds, fit, clip, keyed, renderFirst } = viewPlan();
  const count = (kind) => kinds.filter((k) => k === kind).length;
  const skip = count("skip"), handoff = count("handoff"), shaping = count("shape"), detail = count("detail");
  const span = (a, n) => (n === 1 ? `Step ${a}` : `Steps ${a}–${a + n - 1}`);
  const LABELS = { skip: "Skipped", handoff: "Checkpoint", shape: "Shapes", detail: "Detail" };
  const runs = [];
  kinds.forEach((kind, j) => {
    if (runs.length && runs.at(-1).kind === kind) runs.at(-1).n++;
    else runs.push({ kind, n: 1, from: j + 1 });
  });
  const bar = runs.map((p) => `<span class="views-${p.kind}" style="width:${((100 * p.n) / steps).toFixed(2)}%" title="${LABELS[p.kind]}: ${span(p.from, p.n)}"></span>`).join("");
  const parts = [{ kind: "skip", n: skip }, { kind: "handoff", n: handoff }, { kind: "shape", n: shaping }, { kind: "detail", n: detail }].filter((p) => p.n > 0);
  const keys = parts.map((p) => `<span class="views-key"><i class="views-${p.kind}"></i>${LABELS[p.kind]} ${p.n}</span>`).join("");
  const tips = [];
  if (skip && !shaping && !fit) tips.push("Fit views adds shapes");
  if (keyed) tips.push("Frame skip keyframes, on average");
  if (renderFirst) tips.push("The whole frame first; the tiles then use Tile denoise");
  const tip = tips.map((t) => `<span class="views-tip">${t}</span>`).join("");
  const each = clip ? `<span class="views-key views-each">${keyed ? "A frame" : "Each frame"}</span>` : "";
  box.innerHTML = `<div class="views-bar" aria-hidden="true">${bar}</div><div class="views-legend">${each}${keys}${tip}</div>`;
}

function showLook() {
  showViews();
  const key = currentLook();
  $("look").value = key;
  setHelp($("look"), key === "custom" ? "" : LOOKS[key].hint);
  const style = currentPreset(CLIP_STYLES, STYLE_KEYS);
  $("clip_style").value = style;
  setHelp($("clip_style"), style === "custom" ? "" : CLIP_STYLES[style].hint);
}

function applyPreset(preset, keys) {
  if (!preset) return;
  for (const name of keys) {
    if (!byName[name]) continue;  // a setting this server doesn't have
    const value = preset.values[name] ?? defaultOf(name);
    if (HANDLED.has(name)) state[name] = value;
    else setAdvanced(name, value, defaultOf(name));
  }
  resetAdvancedPanel();
  fillControls();
  save();
}
function applyLook(key) { applyPreset(LOOKS[key], LOOK_KEYS); }
