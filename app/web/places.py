"""Where Nano keeps files: the images, start images and job folders, and the model folders set in Settings. Also
the checks that keep a path the page sends inside them, and the URLs the page reads the files at."""
import os
import re
import shutil
import tempfile
import time

from fastapi import HTTPException
from PIL import Image, ImageOps

import atomic
import paths
from web.base import LOCAL, lock


OUT_DIR = LOCAL.get("output_dir") or paths.OUTPUT
INIT_DIR = paths.INIT_IMAGES
JOBS_DIR = paths.JOBS
BUILT_IN_UPSCALERS = os.path.join(paths.MODELS, "upscale_models")
UPSCALE_DIRS = LOCAL.get("upscale_dirs", [])
BUILT_IN_CHECKPOINTS = os.path.join(paths.MODELS, "checkpoints")
CHECKPOINT_DIRS = LOCAL.get("checkpoint_dirs") or [BUILT_IN_CHECKPOINTS]
KEEP_JOBS = 20
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except OSError:
    print(f"Can't reach {OUT_DIR}, so images will go to {paths.OUTPUT} for now.", flush=True)
    OUT_DIR = paths.OUTPUT
for d in (OUT_DIR, INIT_DIR, JOBS_DIR, BUILT_IN_UPSCALERS, BUILT_IN_CHECKPOINTS):
    os.makedirs(d, exist_ok=True)


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.splitdrive(path)[0].lower() == os.path.splitdrive(root)[0].lower() and os.path.commonpath([path, root]) == root


def image_url(path):
    return "/images/" + os.path.relpath(path, OUT_DIR).replace(os.sep, "/")


def safe_output_path(rel):
    path = os.path.realpath(os.path.join(OUT_DIR, rel))
    if not inside(path, OUT_DIR):
        raise HTTPException(400, "That file is outside the output folder.")
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found.")
    return path


def new_job_dir(prefix):
    job_id = time.strftime("%Y%m%d-%H%M%S-") + prefix
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    jobs = sorted(os.listdir(JOBS_DIR))
    for old in jobs[:-KEEP_JOBS]:
        shutil.rmtree(os.path.join(JOBS_DIR, old), ignore_errors=True)
    return job_id, job_dir


def job_folder(job_id):
    """A job's folder, from an id the page sent."""
    if not re.fullmatch(r"[0-9A-Za-z-]+", job_id):
        raise HTTPException(400, "Bad job id.")
    return os.path.join(JOBS_DIR, job_id)


def writable_folder(path):
    path = os.path.abspath(path)
    try:
        os.makedirs(path, exist_ok=True)
        tempfile.TemporaryFile(dir=path).close()
    except OSError:
        raise HTTPException(400, f"Can't save to {path}.")
    return path


def use_folders(output_dir, upscale_dirs, checkpoint_dirs):
    """Sets the folders, as Settings changes them, and keeps them for the next start. Called holding the lock."""
    global OUT_DIR, UPSCALE_DIRS, CHECKPOINT_DIRS
    OUT_DIR, UPSCALE_DIRS, CHECKPOINT_DIRS = output_dir, upscale_dirs, checkpoint_dirs
    paths.save_local_settings({"output_dir": output_dir, "upscale_dirs": upscale_dirs, "checkpoint_dirs": checkpoint_dirs})


def add_checkpoint_dir(folder):
    """Adds a folder to the checkpoint folders. Under the lock, as Settings' own change of the folders is, so one
    never undoes the other."""
    global CHECKPOINT_DIRS
    if not folder or not os.path.isdir(folder):
        return
    with lock:
        if folder not in CHECKPOINT_DIRS:
            CHECKPOINT_DIRS = CHECKPOINT_DIRS + [folder]
            paths.save_local_settings({"checkpoint_dirs": CHECKPOINT_DIRS})


def image_size(path):
    try:
        with Image.open(path) as image:
            return list(ImageOps.exif_transpose(image).size)
    except OSError:
        return None


IMAGE_KINDS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_KINDS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".gif")


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(str(name))).lstrip(".") or "file"


async def receive(request, path):
    """Streams an upload to disk, so a long video never sits in memory whole."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = atomic.temp_of(path)
    try:
        with open(part, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
    except BaseException:  # the browser went away partway
        os.remove(part)
        raise
    if not atomic.replace(part, path):
        raise HTTPException(409, f"{os.path.basename(path)} is open elsewhere. Close it, or rename the file, and drop it again.")


def input_url(path):
    """A URL for a file under the inputs folder, for the page to preview. Files elsewhere get none."""
    if not inside(path, INIT_DIR):
        return None
    return "/inputs/" + os.path.relpath(os.path.realpath(path), os.path.realpath(INIT_DIR)).replace(os.sep, "/")
