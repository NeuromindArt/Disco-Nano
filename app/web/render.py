"""Rendering: the settings the page sends, checked, and starting, stopping, pausing and following a render."""
import os
import re
import shutil
import socket

from fastapi import HTTPException, Request
from fastapi.responses import Response

import atomic
import keyframes
import models
import paths
import rules
from renderer import signature as renderer_signature
from settings import DEFAULTS, FIELDS, TERMS, handoff_parts, uses_partner
from web.base import app, lock
from web import checkpoints, downloads, fix, jobs, network, places, system


KNOWN_FIELDS = {f["name"] for f in FIELDS} | {"width_height", "init_image", "regions"}


@app.get("/api/schema")
def schema():
    return {"fields": FIELDS, "terms": TERMS}


@app.get("/api/status")
def status(request: Request):
    with lock:
        task = jobs.task_json(jobs.current)
        fetching = downloads.downloads_json()
        tried = jobs.image_tries.json() if jobs.image_tries else None
    system_free = shutil.disk_usage(os.environ.get("SystemDrive", "C:") + os.sep).free
    idle_gpu, idle_ram = system.idle_memory(jobs.current)
    return {"gpu": system.gpu_info(), "ram": system.ram_info(), "other_gpu_app": system.other_gpu_app(), "disk": system.disk_info(),
            "idle": {"gpu": idle_gpu, "ram": idle_ram}, "local": network.is_local(request), "machine": socket.gethostname(),
            "version": paths.VERSION, "system_free_gb": round(system_free / 2**30, 1), "task": task, "downloads": fetching, "image_tries": tried}


def settings_from(body):
    """The settings a page sent, with only the ones Nano knows."""
    settings = body.get("settings", {})
    if not isinstance(settings, dict):
        raise HTTPException(400, "The settings aren't readable.")
    return {k: v for k, v in settings.items() if k in KNOWN_FIELDS}


@app.post("/api/plan")
def render_plan(body: dict, request: Request):
    """The memory plan for these settings, and the settings the others rule out, for the page to show."""
    network.no_network_paths(request, body)
    settings = settings_from(body)
    with lock:
        result = jobs.memory_plan(settings)
        idle = not (jobs.current and jobs.current.running())
    blocked = rules.blocked(settings)
    return {**result, "idle": idle, "blocked": blocked, "conflicts": rules.conflicts(settings, ruled=blocked),
            "clashing": rules.clashing(settings, ruled=blocked)}


@app.post("/api/render")
def render(body: dict, request: Request):
    network.no_network_paths(request, body)
    prompts = body.get("prompts")
    prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()] if isinstance(prompts, list) else []
    if not prompts:
        raise HTTPException(400, "Add a prompt first.")
    with lock:
        busy = jobs.why_busy()
        if busy:
            raise HTTPException(409, busy)
        settings = settings_from(body)
        settings["regions"] = clean_regions(settings.get("regions"))
        if not settings["regions"]:
            del settings["regions"]
        clash = rules.conflicts(settings)
        if clash:
            raise HTTPException(400, clash[0])
        needed = [settings.get("diffusion_model", models.MAIN)] + ["secondary"] * bool(settings.get("use_secondary_model"))
        animation = settings.get("animation_mode", "None")
        needed += [models.PARTNER] * uses_partner(settings)
        needed += [models.DEPTH] * bool(settings.get("depth_target") or animation == "3D")
        if animation != "None":
            try:  # a typo in a keyframe should stop here, not after the models load
                keyframes.motion({**DEFAULTS, **settings}, max(int(settings.get("max_frames", 1)), 1))
                keyframes.skips({**DEFAULTS, **settings}, max(int(settings.get("max_frames", 1)), 1))
                keyframes.prompt_sets(prompts)
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                raise HTTPException(400, str(e)) from None
            if animation == "Video Input" and not os.path.isfile(str(settings.get("video_init_path", ""))):
                raise HTTPException(400, "Choose a video file for Video Input.")
        if any(handoff_parts(settings)):
            checkpoints.checked_model(settings.get("handoff_file"), "handoff")
        for name in needed:
            if name in models.FILES and not os.path.exists(models.path_of(name)):
                raise HTTPException(400, f"{name} isn't downloaded yet. Get it from Models in Settings.")
        if body.get("resume_clip"):
            if animation == "None":
                raise HTTPException(400, "Set Animation to carry on a clip.")
            settings["resume_clip"] = places.safe_output_path(str(body["resume_clip"]))
        batch = re.sub(r"[^A-Za-z0-9 _-]+", "", str(body.get("batch_name", "Disco"))).strip() or "Disco"
        settings["batch_name"] = batch
        job_id, job_dir = places.new_job_dir("render")
        requested = body.get("options", {})
        # Offload is chosen here, from the same plan the page draws, unless the request sets options.offload.
        offload = requested.get("offload")
        if offload is None:
            offload = bool(jobs.memory_plan(settings).get("offload"))
        options = {"offload": bool(offload),
                   "init_fit": requested.get("init_fit") if requested.get("init_fit") in ("fit", "fill") else "crop",
                   "output_dir": places.OUT_DIR}
        atomic.write_json(os.path.join(job_dir, "job.json"), {"settings": settings, "prompts": prompts, "options": options}, indent=1)
        jobs.render_in_worker(prompts[0][:90], job_id, job_dir, renderer_signature(settings),
                         bool(settings.get("keep_models_loaded", DEFAULTS["keep_models_loaded"])))
    return {"id": job_id}


@app.post("/api/stop")
def stop():
    with lock:
        if not jobs.current or not jobs.current.running():
            return {"ok": False}
        # A worker holding models from an earlier render ends cleanly at its next step, a handoff model's included,
        # so those models stay loaded. A first render that's still loading has nothing to keep, so it just ends.
        kept = jobs.current.worker is not None and jobs.current.worker.jobs > 1
        asked = os.path.exists(os.path.join(jobs.current.job_dir, "stop"))
        # A second Stop ends it now, for a step that is taking too long, such as a prompt encoded on the CPU. Its
        # models load again next time.
        if jobs.current.kind == "render" and (jobs.current.sampling_started or kept) and not asked:
            open(os.path.join(jobs.current.job_dir, "stop"), "w").close()
            jobs.current.message = "Stopping after this step. Stop again to end it now."
        else:
            jobs.current.proc.terminate()  # a first render still loading, or a second Stop: the next render starts a new worker
            jobs.current.state, jobs.current.message = "stopped", jobs.FRIENDLY["stopped"]
    return {"ok": True}


def clean_regions(value):
    """Regional prompts as the page sends them, checked: boxes in fractions of the frame, each with a prompt."""
    if not isinstance(value, dict):
        return None
    boxes = []
    listed = value.get("boxes")
    for box in listed[:12] if isinstance(listed, list) else []:
        if not isinstance(box, dict):
            continue
        try:
            x, y, w, h = (min(max(float(box[k]), 0.0), 1.0) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            continue
        text = str(box.get("text", "")).strip()[:400]
        if text and w > 0.01 and h > 0.01:
            boxes.append({"x": x, "y": y, "w": min(w, 1 - x), "h": min(h, 1 - y), "text": text})
    try:
        strength = min(max(float(value.get("strength", 0.75)), 0.0), 1.0)
    except (TypeError, ValueError):
        strength = 0.75
    return {"strength": strength, "boxes": boxes} if boxes else None


@app.post("/api/pause")
def pause(why: str = ""):
    """Holds the render after its next step, so regions can be drawn on the preview, or something erased."""
    with lock:
        if not jobs.current or not jobs.current.running() or jobs.current.kind != "render":
            return {"ok": False}
        if why == "erase":
            fix.refuse_erase(jobs.current)
        if not jobs.current.holding:
            jobs.current.pause_for = "erase" if why == "erase" else "regions"
        open(os.path.join(jobs.current.job_dir, "pause"), "w").close()
        if not jobs.current.holding:
            jobs.current.message = "Pausing after this step"
    return {"ok": True}


@app.post("/api/resume")
def resume(body: dict):
    """Carries on a paused render, with the regions the page sends."""
    with lock:
        if not jobs.current or jobs.current.kind != "render":
            return {"ok": False}
        regions = clean_regions(body.get("regions"))
        if "regions" in body:  # without them, it carries on with the regions it had
            atomic.write_json(os.path.join(jobs.current.job_dir, "regions.json"), regions)
        try:
            os.remove(os.path.join(jobs.current.job_dir, "pause"))
        except FileNotFoundError:
            pass
        if jobs.current.running() and jobs.current.state in ("paused", "rendering"):
            # at once, so the page doesn't read the pause it just ended as a new one
            jobs.current.state, jobs.current.message, jobs.current.holding = "rendering", "Rendering", False
        jobs.current.pause_for = ""
    return {"ok": True}


@app.get("/api/preview/{job_id}")
def preview(job_id: str):
    path = os.path.join(places.job_folder(job_id), "preview.jpg")
    # Read whole, in one go: the renderer replaces this file at every step, and a file sent in pieces could change
    # partway, which breaks the picture and blanks the page's preview for a moment.
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        raise HTTPException(404, "No preview yet.") from None
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/curve")
def curve(text: str, frames: int = 60, shape: str = "Linear"):
    """Every frame's value for one keyframe setting, for the curve editor."""
    try:
        return {"values": keyframes.series(text, min(max(frames, 1), 10000), shape)}
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return {"error": str(e)}
