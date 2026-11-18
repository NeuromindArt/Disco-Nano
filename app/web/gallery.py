"""Images: the gallery of finished images in their folders, moving them, their settings, thumbnails and upscales."""
import json
import os
import re
import shutil

from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image

import atomic
import paths
from web.base import app, lock
from web import jobs, places


GALLERY_PER_FOLDER = 200
RUN = re.compile(r"(.*\(\d+\))_")  # "Disco(3)_0.png" belongs to run "Disco(3)"


def top_folder(path):
    """The gallery folder an image is in: the first folder under the output folder, or "" for none."""
    parts = os.path.relpath(path, places.OUT_DIR).split(os.sep)
    return parts[0] if len(parts) > 1 else ""


def gallery_folders():
    try:
        return sorted(e.name for e in os.scandir(places.OUT_DIR) if e.is_dir() and e.name.lower() != "partials" and not e.name.endswith("_frames"))
    except OSError:
        return []


@app.get("/api/gallery")
def gallery():
    items = scan_gallery()
    shown, counts = [], {}
    for item in items:  # newest first, so each folder keeps its newest
        counts[item["folder"]] = counts.get(item["folder"], 0) + 1
        if counts[item["folder"]] <= GALLERY_PER_FOLDER:
            shown.append(item)
    return {"items": shown, "folders": gallery_folders(), "counts": counts}


def clean_folder_name(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", str(name)).strip().strip(".")
    if not name or name.lower() in ("partials", "con", "prn", "aux", "nul") or name.endswith("_frames") or len(name) > 60:
        raise HTTPException(400, "Use a short folder name with letters, numbers and spaces.")
    return name


@app.post("/api/gallery/folder")
def new_gallery_folder(body: dict):
    """A new folder in the gallery. It is a real folder in the output folder, so renders can go there too."""
    name = clean_folder_name(body.get("name", ""))
    path = os.path.join(places.OUT_DIR, name)
    if os.path.exists(path):
        raise HTTPException(400, f"There is already a folder named {name}.")
    os.makedirs(path)
    return {"folder": name}


@app.post("/api/gallery/move")
def move_to_folder(body: dict):
    """Moves an image into another gallery folder, with what belongs to it: its upscales and handoff picture,
    and for a clip its videos and frames. The run's settings go along, or are copied if other images of the run
    stay behind. If the name is taken there, the run gets the next free number."""
    source = places.safe_output_path(str(body.get("rel", "")))
    folder = str(body.get("folder", ""))
    target = os.path.join(places.OUT_DIR, clean_folder_name(folder)) if folder else places.OUT_DIR
    if not os.path.isdir(target):
        raise HTTPException(404, "That folder no longer exists.")
    here, name = os.path.split(source)
    if os.path.normcase(os.path.abspath(here)) == os.path.normcase(os.path.abspath(target)):
        return {"moved": 0}
    with lock:
        if jobs.current and jobs.current.running() and jobs.current.kind == "render":
            raise HTTPException(409, "Wait for the render to finish, then move images.")
        match = RUN.match(name)
        run = match.group(1) if match else None
        stem = name[:-4]
        moving = [n for n in os.listdir(here) if n.startswith(stem) and (n == name or n[len(stem)] in "_.")]
        clip = run and any(os.path.exists(os.path.join(here, f"{run}.{k}")) for k in ("mp4", "webp", "gif"))
        if clip:
            moving += [n for n in os.listdir(here) if n.startswith(run + ".") or n == run + "_frames"]
        settings_file = f"{run}_settings.txt" if run else None
        others = run and any(n.startswith(run + "_") and n.endswith(".png") and n not in moving for n in os.listdir(here))
        new_run = run
        if run:  # a free run number in the target, so nothing there is overwritten
            base, number = re.match(r"(.*)\((\d+)\)$", run).groups()
            taken = set(os.listdir(target))
            number = int(number)
            while any(n.startswith(f"{base}({number})_") or n.startswith(f"{base}({number}).") for n in taken):
                number += 1
            new_run = f"{base}({number})"
        renamed = lambda n: new_run + n[len(run):] if run and n.startswith(run) else n
        for n in dict.fromkeys(moving):
            if os.path.exists(os.path.join(target, renamed(n))):
                raise HTTPException(400, f"{renamed(n)} is already in that folder.")
        for n in dict.fromkeys(moving):
            shutil.move(os.path.join(here, n), os.path.join(target, renamed(n)))
        if settings_file and os.path.exists(os.path.join(here, settings_file)):
            (shutil.copy2 if others else shutil.move)(os.path.join(here, settings_file), os.path.join(target, renamed(settings_file)))
        moved_to = os.path.join(target, renamed(name))
    return {"moved": len(moving), "url": places.image_url(moved_to)}


def scan_gallery():
    items = []
    for folder, dirs, files in os.walk(places.OUT_DIR):
        # A clip's frames are thousands of PNGs, and the gallery shows its last frame from the folder above.
        dirs[:] = [d for d in dirs if not d.endswith("_frames") and d.lower() != "partials"]
        for name in files:
            if re.search(r"(\(\d+\)_\d+|_x\d+)\.png$", name):
                path = os.path.join(folder, name)
                item = {"url": places.image_url(path), "thumb": "/thumbs/" + places.image_url(path)[len("/images/"):], "name": name,
                        "batch": os.path.basename(folder), "folder": top_folder(path),
                        "rel": os.path.relpath(path, places.OUT_DIR).replace(os.sep, "/"), "mtime": os.path.getmtime(path)}
                run = re.match(r"(.*\(\d+\))_0\.png$", name)
                for kind in ("mp4", "webp", "gif") if run else ():  # an animation's last frame stands for it
                    clip = os.path.join(folder, f"{run.group(1)}.{kind}")
                    if os.path.isfile(clip):
                        item["video"], item["video_kind"] = places.image_url(clip), kind
                        break
                items.append(item)
    items.sort(key=lambda i: i["mtime"], reverse=True)
    return items


@app.get("/api/image-settings")
def image_settings(rel: str):
    """The settings a past image was made with, from the file saved beside it."""
    path = places.safe_output_path(rel)
    match = RUN.match(os.path.basename(path))
    if not match:
        raise HTTPException(404, "No settings saved for this image.")
    txt = os.path.join(os.path.dirname(path), match.group(1) + "_settings.txt")
    if not os.path.isfile(txt):
        raise HTTPException(404, "No settings saved for this image.")
    with open(txt, encoding="utf-8", errors="replace") as f:
        settings = json.load(f)
    if settings.get("init_image"):
        settings["init_size"] = places.image_size(settings["init_image"])
    if not settings.pop("range_scale_works", False):
        settings.pop("range_scale", None)  # it did nothing when this image was made, so reusing it mustn't turn it on
    return settings


@app.get("/api/upscalers")
def upscalers():
    names = {}
    for d in places.UPSCALE_DIRS + [places.BUILT_IN_UPSCALERS]:
        if os.path.isdir(d):
            for n in os.listdir(d):
                if n.endswith((".pth", ".safetensors")) and n not in names:
                    names[n] = os.path.join(d, n)
    return {"models": sorted(names)}


@app.post("/api/upscale")
def upscale(body: dict):
    path = places.safe_output_path(str(body.get("rel", "")))
    model = next((os.path.join(d, str(body.get("model", ""))) for d in places.UPSCALE_DIRS + [places.BUILT_IN_UPSCALERS]
                  if os.path.isfile(os.path.join(d, os.path.basename(str(body.get("model", "")))))), None)
    if model is None or os.path.basename(model) != str(body.get("model")):
        raise HTTPException(400, "Upscaler not found.")
    with lock:
        busy = jobs.why_busy()
        if busy:
            raise HTTPException(409, busy)
        job_id, job_dir = places.new_job_dir("upscale")
        jobs.spawn("upscale", os.path.basename(path), "upscale.py", ["--input", path, "--model", model], job_id, job_dir)
        return {"id": job_id}


@app.get("/images/{rel:path}")
def images(rel: str):
    return FileResponse(places.safe_output_path(rel))  # not a static mount, because the folder can change


THUMBS = os.path.join(paths.HOME, "cache", "thumbs")
THUMB_SIZE = 320


@app.get("/thumbs/{rel:path}")
def thumbs(rel: str):
    """A small JPEG of a finished image, made once and kept. The gallery would otherwise load every image
    at full size, and an upscaled one can be 4096 pixels wide."""
    path = places.safe_output_path(rel)
    stamp = os.stat(path)
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)[-150:] + f"_{int(stamp.st_mtime)}_{stamp.st_size}.jpg"
    thumb = os.path.join(THUMBS, key)
    if not os.path.isfile(thumb):
        os.makedirs(THUMBS, exist_ok=True)
        with Image.open(path) as image:
            image.draft("RGB", (THUMB_SIZE * 2, THUMB_SIZE * 2))
            small = image.convert("RGB")
            small.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        # Another request may have made it meanwhile, and be sending it, so this one can't replace it: that one is used.
        if not atomic.save_image(small, thumb, quality=86) and not os.path.isfile(thumb):
            raise HTTPException(503, "The thumbnail is busy. Try again.")
    return FileResponse(thumb, headers={"Cache-Control": "max-age=31536000, immutable"})
