"""Fix: erase, clone and inpaint on the start image, or on a render paused for it, and the inpaint tries to choose
from."""
import base64
import io
import os
import re
import shutil
import subprocess
import tempfile

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image

import atomic
import inpaint
import procs
from web.base import app, lock
from web import checkpoints, jobs, network, places


@app.post("/api/erase-image")
def erase_image(body: dict, request: Request):
    """Fixes a start image before a render: erases what the mask marks, so the render never has it to bring back,
    and clones the layers over it. The result is a new image beside the start images, so the original is kept. LaMa
    runs in a process of its own: on the GPU when nothing is rendering, and on the CPU while a render is, so the two
    don't compete for VRAM."""
    network.no_network_paths(request, body)
    path = start_image(body)
    mask, clones, inpainting = fix_from(body)
    out = fixed_path(path)
    with tempfile.TemporaryDirectory() as temp:
        args = ["--image", path, "--out", out]
        if mask is not None:
            mask.save(os.path.join(temp, "mask.png"))
            args += ["--mask", os.path.join(temp, "mask.png")]
        if clones:
            atomic.write_json(os.path.join(temp, "clones.json"), jobs.save_clones(clones, temp, "clone"))
            args += ["--clones", os.path.join(temp, "clones.json")]
        if inpainting:
            inpainting["mask"].save(os.path.join(temp, "inpaint.png"))
            atomic.write_json(os.path.join(temp, "inpaint.json"), {**inpainting, "mask": os.path.join(temp, "inpaint.png")})
            args += ["--inpaint", os.path.join(temp, "inpaint.json")]
        with lock:
            if inpainting and jobs.current and jobs.current.running():  # the model and a render won't share the GPU, and on the CPU it would take many minutes
                raise HTTPException(409, "Inpaint the start image when no render is running.")
            on_gpu = jobs.fixing_on_gpu()  # a render or upscale waits for it
        try:
            run = procs.run_once("erase.py", *args, *([] if on_gpu else ["--device", "cpu"]), timeout=900)
        except subprocess.TimeoutExpired:
            raise HTTPException(500, "Fixing took too long and was stopped.") from None
        finally:
            if on_gpu:
                with lock:
                    jobs.fixed_on_gpu()
    if run.returncode or not os.path.isfile(out):
        reason = procs.last_line(run)
        raise HTTPException(500, "Fixing failed" + (f": {reason}" if reason else "."))
    return start_image_json(out)


def start_image(body):
    path = str(body.get("path") or "")
    if not os.path.isfile(path) or places.image_size(path) is None:
        raise HTTPException(400, "The start image is missing. Drop it again.")
    return path


def fixed_path(path):
    """Where a fix of the start image at `path` goes: beside the start images, as a new image, so the original is kept."""
    stem = re.sub(r"_(erased|fixed)\d*$", "", os.path.splitext(os.path.basename(path))[0])
    n = 1
    while os.path.exists(os.path.join(places.INIT_DIR, f"{stem}_fixed{n}.png")):
        n += 1
    os.makedirs(places.INIT_DIR, exist_ok=True)
    return os.path.join(places.INIT_DIR, f"{stem}_fixed{n}.png")


def start_image_json(path):
    return {"path": path, "name": os.path.basename(path), "size": places.image_size(path), "url": places.input_url(path)}


@app.post("/api/image-tries")
def image_tries(body: dict, request: Request):
    """Paints `count` tries, 1 to 4, of a fix of the start image with an inpaint, for the page to choose from: each is
    the whole fix, erase and clones too. `more` adds them to the tries already painted for the same ask; otherwise they
    start over. Each shows in the status's image_tries as it's ready."""
    network.no_network_paths(request, body)
    path = start_image(body)
    mask, clones, inpainting = fix_from(body)
    count = body.get("count")
    if inpainting is None or isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4:
        raise HTTPException(400, "Ask for 1 to 4 tries of an inpaint.")
    with lock:
        if jobs.current and jobs.current.running():  # the model and a render won't share the GPU
            raise HTTPException(409, "Inpaint the start image when no render is running.")
        return {"id": jobs.tries_process().ask(path, mask, clones, inpainting, count, bool(body.get("more")))}


@app.post("/api/image-tries/use")
def use_image_try(body: dict):
    """Makes the chosen try the start image, saved beside the start images as a new image."""
    with lock:
        tries = jobs.image_tries
        if tries is None or body.get("id") != tries.id or body.get("k") not in tries.made:
            raise HTTPException(409, "That try is gone. Paint the tries again.")
        made, image = os.path.join(tries.folder, f"try-{tries.id}-{body['k']}.png"), tries.last["image"]
    out = fixed_path(image)
    shutil.copyfile(made, out)
    return start_image_json(out)


@app.post("/api/image-tries/close")
def close_image_tries():
    """The editor closed over the start image: the tries process ends and lets the model go."""
    with lock:
        jobs.close_tries()
    return {"ok": True}


def refuse_erase(task):
    """Erase is for stills: in a clip, what it removes would show until the frame it's erased on, and a video
    brings it back every frame."""
    if jobs.job_settings(task).get("animation_mode", "None") != "None":
        raise HTTPException(400, "Fix is for stills.")


def mask_from(data):
    """A PNG a page painted, sent as a data URL: white where to erase, or how much to clone."""
    found = re.fullmatch(r"data:image/png;base64,([A-Za-z0-9+/=]+)", str(data or ""))
    if not found or len(found.group(1)) > 40 * 2**20:
        raise HTTPException(400, "The mask isn't a PNG.")
    try:
        return Image.open(io.BytesIO(base64.b64decode(found.group(1)))).convert("L")
    except (OSError, ValueError):
        raise HTTPException(400, "The mask isn't a PNG.")


INPAINT_MODES = ("remove", "fresh", "rework")


def inpaint_from(asked):
    """The inpaint a page sent, checked: {"mask": PNG, "prompt", "mode": remove, fresh or rework, "strength": 0 to 1,
    "file": the Inpaint model, "steps", "guidance"}, with the mask as an image. Or None for none."""
    if asked is None:
        return None
    if not isinstance(asked, dict):
        raise HTTPException(400, "The inpaint isn't readable.")
    path = str(asked.get("file") or "")
    if not path:
        raise HTTPException(400, "Choose an Inpaint model in Settings first.")
    checkpoints.checked_model(path, "inpaint")
    mode, strength, steps, guidance = asked.get("mode"), asked.get("strength"), asked.get("steps"), asked.get("guidance")
    if mode not in INPAINT_MODES:
        raise HTTPException(400, "Inpaint starts from remove, fresh or rework.")
    for value, low, high, what in ((strength, 0, 1, "Strength is a share from 0 to 1."), (steps, 1, 200, "Inpaint steps are 1 to 200."),
                                   (guidance, 0, 30, "Inpaint guidance is 0 to 30.")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise HTTPException(400, what)
    shape = {"context": asked.get("context", inpaint.CONTEXT), "grow": asked.get("grow", inpaint.GROW), "blend": asked.get("blend", inpaint.BLEND)}
    for value, low, high, what in ((shape["context"], 1, 4, "Inpaint context is 1 to 4."), (shape["grow"], 0, 200, "Inpaint grow is 0 to 200 pixels."),
                                   (shape["blend"], 0, 200, "Inpaint blend is 0 to 200 pixels.")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise HTTPException(400, what)
    prompt = asked.get("prompt") or ""
    if not isinstance(prompt, str) or len(prompt) > 1000:
        raise HTTPException(400, "The inpaint prompt is text of up to 1000 characters.")
    chosen = asked.get("chosen")
    if chosen is not None and not (isinstance(chosen, dict) and all(isinstance(chosen.get(k), int) and not isinstance(chosen.get(k), bool)
                                                                     and chosen[k] >= 0 for k in ("id", "k"))):
        raise HTTPException(400, "The chosen try isn't readable.")
    return {"mask": mask_from(asked.get("mask")), "prompt": prompt, "mode": mode, "strength": float(strength), "chosen": chosen,
            "file": path, "steps": int(steps), "guidance": float(guidance), "context": float(shape["context"]),
            "grow": int(shape["grow"]), "blend": int(shape["blend"])}


def fix_from(body):
    """The erase mask, or None, the clone layers a page sent: each {"mask": PNG, "dx": , "dy": }, how far right and
    down its source is as a share of the picture's width and height, and the inpaint, as inpaint_from() reads it, or
    None. At least one of them."""
    mask = None if body.get("mask") is None else mask_from(body.get("mask"))
    clones = body.get("clones") or []
    if not isinstance(clones, list) or len(clones) > 50:
        raise HTTPException(400, "Clones are a list of up to 50 layers.")
    layers = []
    for layer in clones:
        offset = [layer.get(k) if isinstance(layer, dict) else None for k in ("dx", "dy")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not -1 <= v <= 1 for v in offset):
            raise HTTPException(400, "A clone's source is a share of the picture from -1 to 1 across and down.")
        layers.append((mask_from(layer.get("mask")), float(offset[0]), float(offset[1])))
    inpainting = inpaint_from(body.get("inpaint"))
    if mask is None and not layers and inpainting is None:
        raise HTTPException(400, "Paint something to erase, inpaint or clone.")
    return mask, layers, inpainting


@app.post("/api/erase")
def erase_mask(body: dict, request: Request):
    """What was painted over a paused render: an erase mask, white where to erase, and clone layers, as fix_from()
    reads them. The render fixes them as it carries on, after the step it paused at. Send it before /api/resume.
    `fresh`, 0 to 1, is the share of fresh noise the areas get; without it, the renderer picks one by how far through
    the render it is."""
    network.no_network_paths(request, body)
    mask, clones, inpainting = fix_from(body)
    fresh = body.get("fresh")
    if fresh is not None:
        if isinstance(fresh, bool) or not isinstance(fresh, (int, float)) or not 0 <= fresh <= 1:
            raise HTTPException(400, "Fresh noise is a share from 0 to 1.")
        fresh = float(fresh)
    with lock:
        if not jobs.current or not jobs.current.running() or jobs.current.kind != "render":
            raise HTTPException(409, "No render is running.")
        refuse_erase(jobs.current)
        n = 0
        while any(os.path.exists(os.path.join(jobs.current.job_dir, f)) for f in (f"erase-{n}.png", f"clone-{n}-0.png", f"inpaint-{n}.png")):
            n += 1
        path = None
        if mask is not None:
            path = os.path.join(jobs.current.job_dir, f"erase-{n}.png")
            mask.save(path)
        clones = jobs.save_clones(clones, jobs.current.job_dir, f"clone-{n}")
        if inpainting:
            inpainting["mask"].save(os.path.join(jobs.current.job_dir, f"inpaint-{n}.png"))
            inpainting["mask"] = os.path.join(jobs.current.job_dir, f"inpaint-{n}.png")
        atomic.write_json(os.path.join(jobs.current.job_dir, "erase.json"), {"mask": path, "clones": clones, "inpaint": inpainting, "n": n, "fresh": fresh})
    return {"ok": True}


@app.post("/api/inpaint-tries")
def inpaint_tries(body: dict, request: Request):
    """Paints `count` inpaint tries, 1 to 4, while a render is paused for Fix, for the page to choose from. `more`
    adds them to the tries already painted for the same ask; otherwise they start over. Each shows in the status's
    tries as it's ready, and Apply sends the chosen one with the fix."""
    network.no_network_paths(request, body)
    inpainting = inpaint_from(body.get("inpaint"))
    count = body.get("count")
    if inpainting is None or isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4:
        raise HTTPException(400, "Ask for 1 to 4 tries of an inpaint.")
    with lock:
        if not jobs.current or not jobs.current.running() or jobs.current.kind != "render" or not jobs.current.holding:
            raise HTTPException(409, "Tries are painted while a render is paused for Fix.")
        refuse_erase(jobs.current)
        tries = jobs.current.tries
        if not body.get("more"):
            tries.update(id=tries["id"] + 1, made=[], asked=0)
        tries["asked"], tries["error"] = len(tries["made"]) + count, None
        inpainting["mask"].save(os.path.join(jobs.current.job_dir, f"tries-{tries['id']}.png"))
        ask = {"id": tries["id"], "count": count, "inpaint": {**inpainting, "mask": os.path.join(jobs.current.job_dir, f"tries-{tries['id']}.png")}}
        atomic.write_json(os.path.join(jobs.current.job_dir, "tries.json"), ask)
        return {"id": tries["id"]}


@app.get("/api/try/{job_id}/{ask}/{k}")
def inpaint_try(job_id: str, ask: int, k: int):
    """One inpaint try's picture, the size of the render or the start image."""
    for ext in ("jpg", "png"):  # a render's tries, or the start image's, which become start images
        path = os.path.join(places.job_folder(job_id), f"try-{ask}-{k}.{ext}")
        if os.path.isfile(path):
            return FileResponse(path, headers={"Cache-Control": "no-store"})
    raise HTTPException(404, "No such try.")
