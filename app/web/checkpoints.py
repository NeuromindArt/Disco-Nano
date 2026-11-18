"""Checkpoints: the models Handoff and Inpaint can use, checked before anything loads them, and the model folders
they're found in."""
import os
import threading

from fastapi import HTTPException, Request

import budget
import finder
import flowfiles
import inpaint
import models
import paths
from web.base import LOCAL, app
from web import network, pickers, places


@app.post("/api/pick-checkpoint")
def pick_checkpoint(body: dict, request: Request):
    """Opens Windows' file picker for the handoff model. The checkpoint is read where it is, never copied, and
    its folder joins the checkpoint folders so the list finds its neighbours too."""
    network.local_only(request)
    picked = pickers.pick_file(body.get("start", ""), f"Choose {MODEL_KINDS}", "Models|*.safetensors;*.gguf")
    if picked["path"]:
        checked_model(picked["path"], "checkpoint")
        places.add_checkpoint_dir(os.path.dirname(picked["path"]))
    return picked


MODEL_FILES = (".safetensors", ".gguf")
MODEL_KINDS = "an SD 1.5, SDXL, Z-Image, Flux.1 or Flux.2 Klein model"
MODEL_MISSING = {  # what's wrong with a model chosen for each use that isn't a model file there
    "handoff": "The handoff model must be a .safetensors or .gguf file that exists. Choose it again in Handoff.",
    "inpaint": "The Inpaint model isn't there any more: {path}",
    "checkpoint": "Choose a .safetensors or .gguf model.",
}


def checked_model(path, use):
    """A model file the page chose, checked before anything loads it: for "handoff", for "inpaint", or as a
    "checkpoint" for either, as Browse adds one. Returns what budget.describe_checkpoint() makes of it, and raises
    an HTTPException that says what's wrong otherwise."""
    path = str(path or "")
    if not path.lower().endswith(MODEL_FILES) or not os.path.isfile(path):
        raise HTTPException(400, MODEL_MISSING[use].format(path=path))
    about = budget.describe_checkpoint(path)
    if not about:
        what = {"handoff": "Handoff", "inpaint": "Inpaint"}.get(use, "Nano")
        raise HTTPException(400, f"{what} takes {MODEL_KINDS}, and this is another kind of model.")
    if use == "handoff" and about.get("inpaint_only"):
        raise HTTPException(400, "That model is made for inpainting. Use it as the Inpaint model in Settings, and choose another for Handoff.")
    missing = flowfiles.missing_parts(path, about["kind"]) if use != "checkpoint" and about["kind"] in flowfiles.KINDS else []
    if missing:
        raise HTTPException(400, f"{flowfiles.KINDS[about['kind']]} also needs its {' and '.join(missing)}. Nano looks for "
                                 "them in the models folder the model is in, subfolders included, as ComfyUI keeps them.")
    return about


@app.post("/api/add-checkpoint")
def add_checkpoint(body: dict, request: Request):
    """A checkpoint chosen in the page's own browser. Its folder joins the checkpoint folders, as with Browse."""
    network.no_network_paths(request, body.get("path", ""))
    path = os.path.abspath(str(body.get("path", "")))
    checked_model(path, "checkpoint")
    places.add_checkpoint_dir(os.path.dirname(path))
    return {"path": path}


_search_lock = threading.Lock()  # one search at a time
_upgrade = {"thread": None}  # the one search for a PC updated from a version that searched on every visit


def update_from_searching():
    """Versions before 1.1.1 searched the drives every time the Handoff list was asked for. On the first start after
    an update, that search runs once more, and what it finds is added to the checkpoint folders, where it can be
    seen and removed. A new install asks on the welcome screen instead."""
    if LOCAL.get("checkpoint_search_done") or not LOCAL.get("welcome_done"):
        return

    def run():
        with _search_lock:
            for folder in finder.find_other_apps(places.UPSCALE_DIRS):
                places.add_checkpoint_dir(folder)
        paths.save_local_settings({"checkpoint_search_done": True})

    _upgrade["thread"] = threading.Thread(target=run, daemon=True)
    _upgrade["thread"].start()


@app.get("/api/find-checkpoint-folders")
def find_checkpoint_folders():
    """Checkpoint folders of ComfyUI and the Stable Diffusion web UIs on this PC that aren't in the list yet, and the
    models Nano would download that those apps already have, which it then uses where they are. It runs only when
    asked: by the welcome screen, and by Find in other apps in Settings. The page adds the folders it wants."""
    with _search_lock:
        found = finder.find_other_apps(places.UPSCALE_DIRS)
        copies = models.find_copies(finder.model_roots([*found, *places.CHECKPOINT_DIRS], places.UPSCALE_DIRS))
    known = {os.path.normcase(os.path.normpath(d)) for d in places.CHECKPOINT_DIRS}
    return {"found": [d for d in found if os.path.normcase(os.path.normpath(d)) not in known],
            "models": [models.LABELS.get(n, n) for n in copies]}


def model_folders():
    """The checkpoint folders, and beside a checkpoints folder, the unet and diffusion_models folders where ComfyUI
    keeps Z-Image and Flux. Drives that are unplugged for now are skipped."""
    folders = []
    for top in list(places.CHECKPOINT_DIRS):
        beside = [os.path.join(os.path.dirname(os.path.normpath(top)), n) for n in ("unet", "diffusion_models")] \
            if os.path.basename(os.path.normpath(top)).lower() == "checkpoints" else []
        for folder in [top, *beside]:
            if os.path.isdir(folder) and os.path.normcase(folder) not in map(os.path.normcase, folders):
                folders.append(folder)
    return folders


@app.get("/api/checkpoints")
def checkpoints():
    """Models for the Handoff list: SD 1.5, SDXL, Z-Image, Flux.1 and Flux.2 Klein, in the model folders, subfolders included.
    Like the upscalers, they are read where they are, so a browser on another PC picks from the files on this one.
    A file found twice, such as in a backup, shows once. Models Handoff can't use, such as SD 3 or Flux.2, are left
    out. Each one's group is its subfolder."""
    if _upgrade["thread"] is not None:  # the first list after an update waits for its one search
        _upgrade["thread"].join(timeout=15)
    found, seen = [], set()
    for top in model_folders():
        for folder, subfolders, names in os.walk(top):
            subfolders[:] = sorted(s for s in subfolders if not s.startswith(".")) if folder.count(os.sep) - top.count(os.sep) < 3 else []
            for n in sorted(names, key=str.lower):
                if not n.lower().endswith(MODEL_FILES):
                    continue
                path = os.path.join(folder, n)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                name = os.path.splitext(os.path.relpath(path, top))[0].replace(os.sep, "/")
                if (name.lower(), size) in seen:
                    continue
                seen.add((name.lower(), size))
                about = budget.describe_checkpoint(path)
                if about:
                    group, _, short = name.rpartition("/")
                    found.append({"path": path, "name": short, "group": group, "sdxl": about["kind"] == "sdxl", **about,
                                  "inpaint": inpaint.defaults(about)})
    found.sort(key=lambda f: (f["group"].lower(), f["name"].lower()))
    return {"checkpoints": found, "folders": places.CHECKPOINT_DIRS}
