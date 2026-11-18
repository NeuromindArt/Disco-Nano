"""Files the page sends to use in a render: start, ghost and style images, videos and image prompts, uploaded so
a browser on another PC works the same, and previews of them."""
import os
import re
import shutil
import time

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from web.base import app
from web import network, places


@app.post("/api/init-image")
async def init_image(request: Request, name: str):
    """Start, ghost and style images are uploaded, so they work the same from a browser on another PC."""
    safe = places.safe_name(name)
    if not safe.lower().endswith(places.IMAGE_KINDS):
        raise HTTPException(400, "Use a PNG, JPG or WEBP image.")
    path = os.path.join(places.INIT_DIR, safe)
    await places.receive(request, path)
    if places.image_size(path) is None:
        os.remove(path)
        raise HTTPException(400, f"{name} isn't an image Nano can read.")
    return {"path": path, "name": safe, "size": places.image_size(path), "url": places.input_url(path)}


@app.post("/api/upload-video")
async def upload_video(request: Request, name: str):
    """A video for Video Input, kept with the start images."""
    safe = places.safe_name(name)
    if not safe.lower().endswith(places.VIDEO_KINDS):
        raise HTTPException(400, "Use an MP4, MOV, WEBM, MKV, AVI or GIF video.")
    path = os.path.join(places.INIT_DIR, "videos", safe)
    await places.receive(request, path)
    return {"path": path, "name": safe, "url": places.input_url(path)}


@app.post("/api/prompt-images")
async def prompt_image(request: Request, name: str, group: str = ""):
    """One image of a set of image prompts. The page sends each file of a drop to the same new folder."""
    safe = places.safe_name(name)
    if not safe.lower().endswith(places.IMAGE_KINDS):
        raise HTTPException(400, f"{name} isn't a PNG, JPG or WEBP image.")
    group = re.sub(r"[^A-Za-z0-9_-]+", "", group)[:40] or time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(places.INIT_DIR, "prompt_images", group, safe)
    await places.receive(request, path)
    if places.image_size(path) is None:
        os.remove(path)
        raise HTTPException(400, f"{name} isn't an image Nano can read.")
    return {"folder": os.path.dirname(path), "group": group}


@app.get("/api/prompt-images")
def prompt_image_list(folder: str, request: Request):
    """The images in an image prompt folder, with previews when they were uploaded here."""
    network.no_network_paths(request, folder)
    if not os.path.isdir(folder):
        return {"images": [], "missing": True}
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(places.IMAGE_KINDS))
    return {"images": [{"name": n, "url": places.input_url(os.path.join(folder, n))} for n in names], "missing": False}


@app.get("/api/input-preview")
def input_preview(path: str, request: Request):
    """A preview URL and the size of an image the settings name, such as a ghost image."""
    network.no_network_paths(request, path)
    found = os.path.isfile(path)
    return {"url": places.input_url(path) if found else None, "size": places.image_size(path) if found else None, "missing": not found}


@app.get("/inputs/{rel:path}")
def inputs(rel: str):
    path = os.path.realpath(os.path.join(places.INIT_DIR, rel))
    if not places.inside(path, places.INIT_DIR) or not os.path.isfile(path):
        raise HTTPException(404, "File not found.")
    return FileResponse(path)


@app.post("/api/init-from-output")
def init_from_output(body: dict):
    path = places.safe_output_path(str(body.get("rel", "")))
    name = places.safe_name(path)
    target = os.path.join(places.INIT_DIR, name)
    shutil.copyfile(path, target)
    return {"path": target, "name": name, "size": places.image_size(target), "url": places.input_url(target)}
