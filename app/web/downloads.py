"""Models to download, fetched one at a time, and the welcome screen that offers them on the first visit."""
import os
import re
import shutil
import threading
from collections import deque

from fastapi import HTTPException, Request

import models
import paths
from web.base import app, lock
from web import network


downloads = {"running": False, "current": None, "percent": None, "waiting": deque(), "errors": {}}


def run_downloads():
    """Fetches models one at a time, on a thread of its own so renders carry on."""
    def report(message):
        found = re.search(r"\((\d+)%\)", message)
        downloads["percent"] = int(found.group(1)) if found else None

    while True:
        with lock:
            if not downloads["waiting"]:
                downloads["running"] = False
                downloads["current"] = None
                return
            name = downloads["current"] = downloads["waiting"].popleft()
            downloads["percent"] = 0
        try:
            models.ensure(name, report)
        except Exception:
            downloads["errors"][name] = "The download failed. Check your connection and try again."


@app.get("/api/models")
def model_list():
    def kind(name):
        if name in (models.PARTNER, models.DEPTH, models.POSE, models.PERSON, models.FACES, models.RIFE, models.LAMA, *models.ROBUST):
            return "partner"
        if name == models.FLUX_FILL:
            return "inpaint"
        return "upscaler" if name == models.UPSCALER else "helper" if name == "secondary" else "diffusion"
    return {"models": [{"name": n, "label": models.LABELS.get(n, n), "kind": kind(n), "about": models.ABOUT[n],
                        "size_gb": models.FILES[n]["size_gb"], "cuts": models.CUT_SCHEDULES.get(models.cut_style(n)),
                        "downloaded": os.path.exists(models.path_of(n)),
                        "found_at": models.path_of(n) if models.path_of(n) != models.own_path(n) else None}
                       for n in models.FILES]}


def downloads_json():
    return {"current": downloads["current"], "percent": downloads["percent"], "waiting": list(downloads["waiting"]),
            "errors": dict(downloads["errors"])}


@app.post("/api/downloads")
def start_downloads(body: dict):
    with lock:
        busy = {downloads["current"], *downloads["waiting"]}
        names = [n for n in dict.fromkeys(body.get("names", []))
                 if n in models.FILES and n not in busy and not os.path.exists(models.path_of(n))]
        waiting = sum(models.FILES[n]["size_gb"] for n in names + list(busy - {None}))
        free = shutil.disk_usage(paths.HOME).free / 2**30
        if waiting + 1 > free:
            raise HTTPException(400, f"That needs {waiting:.1f}GB and {free:.1f}GB is free where models are kept.")
        for n in names:
            downloads["errors"].pop(n, None)
        downloads["waiting"].extend(names)
        if names and not downloads["running"]:
            downloads["running"] = True
            threading.Thread(target=run_downloads, daemon=True).start()
        return downloads_json()


@app.get("/api/first-run")
def first_run(request: Request):
    """The welcome screen sets this PC's folders, so a browser on another PC never sees it."""
    return {"show": network.is_local(request) and not paths.local_settings().get("welcome_done")}


@app.post("/api/first-run")
def first_run_done(request: Request):
    network.local_only(request)
    paths.save_local_settings({"welcome_done": True, "checkpoint_search_done": True})
    return {"show": False}
