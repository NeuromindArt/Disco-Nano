"""Local web app for Disco Diffusion Nano. Binds to 127.0.0.1 only.

Never imports torch, so the idle server holds no VRAM. Renders and upscales each run in
their own process, so their memory is fully released when they end and an out-of-memory
failure cannot take the app down.
"""
import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import deque
from urllib.parse import urlparse

print("Loading the web server", flush=True)
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

print("Checking folders", flush=True)
import budget
import keyframes
import models
import paths
import rules
from settings import DEFAULTS, FIELDS, TERMS, uses_partner


APP_DIR = paths.APP_DIR
ROOT = paths.HOME
PYTHON = paths.PYTHON
LOCAL = paths.local_settings()
OUT_DIR = LOCAL.get("output_dir") or paths.OUTPUT
INIT_DIR = paths.INIT_IMAGES
JOBS_DIR = paths.JOBS
BUILT_IN_UPSCALERS = os.path.join(paths.MODELS, "upscale_models")
UPSCALE_DIRS = LOCAL.get("upscale_dirs", [])
BUILT_IN_CHECKPOINTS = os.path.join(paths.MODELS, "checkpoints")
CHECKPOINT_DIRS = LOCAL.get("checkpoint_dirs") or [BUILT_IN_CHECKPOINTS]
OTHER_GPU_PORTS = LOCAL.get("other_gpu_ports", [8188, 7860])  # ComfyUI and Stable Diffusion web UIs
HOST, PORT = "127.0.0.1", 7861
URL = f"http://{HOST}:{PORT}"
KEEP_JOBS = 20
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except OSError:
    print(f"Can't reach {OUT_DIR}, so images will go to {paths.OUTPUT} for now.", flush=True)
    OUT_DIR = paths.OUTPUT
for d in (OUT_DIR, INIT_DIR, JOBS_DIR, BUILT_IN_UPSCALERS, BUILT_IN_CHECKPOINTS):
    os.makedirs(d, exist_ok=True)

KNOWN_FIELDS = {f["name"] for f in FIELDS} | {"width_height", "init_image"}
FRIENDLY = {
    "oom": "The GPU ran out of memory. Close other GPU apps, or lower the size or use fewer CLIP models.",
    "error": "The render failed. Details are in the log below.",
    "stopped": "Stopped.",
}

app = FastAPI(title="Disco Diffusion Nano")
lock = threading.Lock()


class Task:
    """The one render or upscale allowed to run at a time."""

    def __init__(self, kind, job_id, job_dir, proc, label):
        self.kind, self.job_id, self.job_dir, self.proc, self.label = kind, job_id, job_dir, proc, label
        self.state, self.message = "starting", "Starting"
        self.step = self.total = self.batch = 0
        self.n_batches = 1
        self.step_times = deque(maxlen=12)
        self.started = time.time()
        self.sampling_started = None
        self.paused = False  # a status message came between steps
        self.seed = None
        self.images = []
        self.preview_rev = 0
        self.peak_vram = None
        self.error = None
        self.log = deque(maxlen=60)
        self.finished_at = None


current = None
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


def pump(task):
    """Follows a worker's stdout, turning @@EVT lines into task state."""
    log_path = os.path.join(task.job_dir, "run.log")
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        for raw in task.proc.stdout:
            line = raw.rstrip("\r\n")
            log.write(line + "\n")
            log.flush()
            marker = line.find("@@EVT ")
            if marker < 0:
                if line.strip():
                    task.log.append(line[-300:])
                continue
            try:
                handle_event(task, json.loads(line[marker + 6:]))
            except ValueError:
                continue
    code = task.proc.wait()
    with lock:
        if task.state not in ("done", "stopped", "oom", "error"):
            task.state = "error"  # the worker died without reporting why
            task.message = FRIENDLY["error"]
            task.error = f"Worker exited unexpectedly (code {code})."
        task.finished_at = time.time()


def handle_event(task, e):
    kind = e["type"]
    with lock:
        if kind == "status":
            task.state, task.message = "loading", e["message"]
            task.paused = True  # set by a status between steps, such as a handoff, and cleared at the next step
        elif kind == "sampling":
            task.state, task.message = "rendering", "Upscaling" if task.kind == "upscale" else "Rendering"
            task.total, task.n_batches, task.seed = e["total_steps"], e["n_batches"], e.get("seed")
            task.sampling_started = e["t"]
        elif kind == "step":
            if task.paused and task.sampling_started:
                task.state, task.message, task.paused = "rendering", "Upscaling" if task.kind == "upscale" else "Rendering", False
            task.step, task.total, task.batch = e["step"], e["total"], e["batch"]
            task.step_times.append(e["t"])
        elif kind == "preview":
            task.preview_rev += 1
        elif kind == "image":
            task.images.append(e["path"])
        elif kind == "finished":
            task.state = e["outcome"]
            task.message = FRIENDLY.get(e["outcome"], "Done")
            task.error = e.get("error")
            task.peak_vram = e.get("peak_vram_gb")


def seconds_per_step(task):
    t = task.step_times
    return (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else None


def task_json(task):
    if task is None:
        return None
    spp = seconds_per_step(task)
    eta = None
    if spp and task.total:
        left = max(task.total - task.step, 0) + (task.n_batches - task.batch - 1) * task.total
        eta = round(left * spp)
    return {
        "kind": task.kind, "id": task.job_id, "label": task.label, "state": task.state, "message": task.message,
        "step": task.step, "total": task.total, "batch": task.batch, "n_batches": task.n_batches,
        "seconds_per_step": round(spp, 2) if spp else None, "eta_s": eta, "seed": task.seed,
        "elapsed_s": round((task.finished_at or time.time()) - task.started), "preview_rev": task.preview_rev,
        "images": [image_url(p) for p in task.images], "peak_vram_gb": task.peak_vram, "error": task.error,
        "log": list(task.log)[-14:] if task.state in ("error", "oom") else [],
        "active": task.proc.poll() is None,
    }


_gpu_seen = {"at": 0.0, "info": None}


def gpu_info():
    """Calls nvidia-smi at most once every 5 seconds. Each call asks the driver for a report,
    and polling once a second during a render makes the screen hitch more often."""
    if time.time() - _gpu_seen["at"] < 5:
        return _gpu_seen["info"]
    _gpu_seen["info"] = read_gpu()
    _gpu_seen["at"] = time.time()
    return _gpu_seen["info"]


def read_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
        used, total, util = [float(x) for x in out.split(",")]
        return {"used_gb": round(used / 1024, 2), "total_gb": round(total / 1024, 2), "util": util}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ram_info():
    """System RAM, from Windows. Cheap enough to read on every status call."""
    if os.name != "nt":
        return None
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = Status(dwLength=ctypes.sizeof(Status))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    total = status.ullTotalPhys / 2**30
    return {"used_gb": round(total - status.ullAvailPhys / 2**30, 1), "total_gb": round(total, 1)}


# GPU and RAM use from before the running render started. The memory plan stacks a render on top of
# these, so while one runs, the plan is drawn against what was in use without it.
_idle = {"gpu": None, "ram": None}


def idle_memory():
    if not (current and current.proc.poll() is None):
        _idle["gpu"], _idle["ram"] = gpu_info() or _idle["gpu"], ram_info() or _idle["ram"]
    return _idle["gpu"], _idle["ram"]


def memory_plan(settings):
    gpu, ram = idle_memory()
    if not gpu or not ram:
        return {}
    return budget.plan(settings, gpu, ram)


def disk_info():
    """The drive that holds the images, which is what the status bar shows, and how much of it is Nano's."""
    try:
        usage = shutil.disk_usage(os.path.splitdrive(OUT_DIR)[0] + os.sep)
    except OSError:
        return None  # the drive was unplugged while the app was open
    measure_disco()
    return {"used_gb": round(usage.used / 2**30, 1), "total_gb": round(usage.total / 2**30, 1),
            "free_gb": round(usage.free / 2**30, 1), "folder": OUT_DIR, "disco_gb": _disco_size["gb"]}


_disco_size = {"at": 0.0, "gb": None, "busy": False}


def folder_bytes(top):
    total = 0
    for folder, _, names in os.walk(top):
        for n in names:
            try:
                total += os.lstat(os.path.join(folder, n)).st_size
            except OSError:
                pass
    return total


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.splitdrive(path)[0].lower() == os.path.splitdrive(root)[0].lower() and os.path.commonpath([path, root]) == root


def measure_disco():
    """Nano's files on the images' drive: its home folder (environment, models, inputs) and the images.
    Walking every file takes a few seconds, so it runs on a thread at most every five minutes."""
    if _disco_size["busy"] or time.time() - _disco_size["at"] < 300:
        return

    def walk():
        try:
            drive = os.path.splitdrive(os.path.realpath(OUT_DIR))[0].lower()
            tops = [ROOT] + ([] if inside(OUT_DIR, ROOT) else [OUT_DIR])
            same = [t for t in tops if os.path.splitdrive(os.path.realpath(t))[0].lower() == drive]
            _disco_size["gb"] = round(sum(folder_bytes(t) for t in same) / 2**30, 1)
        except OSError:
            pass  # the next pass tries again
        finally:
            _disco_size["at"], _disco_size["busy"] = time.time(), False

    _disco_size["busy"] = True
    threading.Thread(target=walk, daemon=True).start()


_other_seen = {"at": 0.0, "open": False}


def other_gpu_app():
    """Rechecked at most every 10 seconds, because a closed port on Windows takes the whole timeout to answer."""
    if time.time() - _other_seen["at"] > 10:
        _other_seen["open"] = any(port_open(p) for p in OTHER_GPU_PORTS)
        _other_seen["at"] = time.time()
    return _other_seen["open"]


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.15)
        return s.connect_ex(("127.0.0.1", port)) == 0


def image_url(path):
    return "/images/" + os.path.relpath(path, OUT_DIR).replace(os.sep, "/")


def safe_output_path(rel):
    path = os.path.realpath(os.path.join(OUT_DIR, rel))
    root = os.path.realpath(OUT_DIR)
    if os.path.splitdrive(path)[0].lower() != os.path.splitdrive(root)[0].lower() or os.path.commonpath([path, root]) != root:
        raise HTTPException(400, "That file is outside the output folder.")
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found.")
    return path


def spawn(kind, label, cmd, job_id, job_dir):
    global current
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", DISCO_NANO_HOME=ROOT)
    # Below normal priority: a render still gets every idle cycle, but Windows serves the desktop first,
    # so the PC stays usable while it runs.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", cwd=ROOT, env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    current = Task(kind, job_id, job_dir, proc, label)
    threading.Thread(target=pump, args=(current,), daemon=True).start()


def new_job_dir(prefix):
    job_id = time.strftime("%Y%m%d-%H%M%S-") + prefix
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    jobs = sorted(os.listdir(JOBS_DIR))
    for old in jobs[:-KEEP_JOBS]:
        shutil.rmtree(os.path.join(JOBS_DIR, old), ignore_errors=True)
    return job_id, job_dir


# ---- rendering for other PCs on the network ----
# Off until turned on in All settings. Then Nano also listens on this PC's network addresses, and a browser
# on another PC opens http://<address>:7861. It enters the code shown here once, and a cookie remembers it.
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
KEY_COOKIE = "nano_key"
CODE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I to misread
_network = {"servers": [], "addresses": [], "error": None}
_wrong = {}  # address: wrong codes, to slow down guessing


def share_code():
    code = LOCAL.get("share_code")
    if not code:
        import secrets
        code = "".join(secrets.choice(CODE_LETTERS) for _ in range(8))
        LOCAL["share_code"] = code
        paths.save_local_settings({"share_code": code})
    return code


def lan_addresses():
    """This PC's IPv4 addresses on its networks, leaving out loopback and the self-assigned 169.254 ones."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:  # the address used to reach the network, in case the name lookup missed it
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # a documentation address: nothing is sent
            found.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(a for a in found if not a.startswith(("127.", "169.254.", "0.")))


def start_sharing():
    """Listens on every network address too, beside 127.0.0.1. Windows may ask once to let Python through."""
    stop_sharing()
    _network["error"] = None
    for address in lan_addresses():
        server = uvicorn.Server(uvicorn.Config(app, host=address, port=PORT, log_level="warning"))
        server.install_signal_handlers = lambda: None
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        _network["servers"].append(server)
    deadline = time.time() + 3
    while time.time() < deadline and not all(s.started for s in _network["servers"]):
        time.sleep(0.05)
    _network["addresses"] = [s.config.host for s in _network["servers"] if s.started]
    if not _network["addresses"]:
        _network["error"] = "Nano couldn't listen on this PC's network. Check that it's connected."


def stop_sharing():
    for server in _network["servers"]:
        server.should_exit = True
    _network["servers"], _network["addresses"] = [], []


def network_json():
    on = bool(LOCAL.get("share_on_network"))
    return {"on": on, "addresses": _network["addresses"] if on else [], "port": PORT, "code": share_code() if on else None,
            "name": socket.gethostname(), "error": _network["error"] if on else None}


def from_network(request):
    """True for a request that came in over the network rather than from this PC's own browser."""
    return not is_local(request)


@app.middleware("http")
async def only_this_app(request: Request, call_next):
    """Stops other websites in the same browser from driving this server, and asks a browser on another PC for
    the code before it can do anything."""
    host = request.headers.get("host", "").rsplit(":", 1)[0].lower()
    origin = request.headers.get("origin")
    if from_network(request):
        if not LOCAL.get("share_on_network"):
            return JSONResponse({"detail": "Sharing is off on this PC."}, 403)
        known = {a.lower() for a in _network["addresses"]} | {socket.gethostname().lower()}
        if host not in known:
            return JSONResponse({"detail": "Forbidden host."}, 403)
        if origin and urlparse(origin).hostname not in known:
            return JSONResponse({"detail": "Forbidden origin."}, 403)
        if request.cookies.get(KEY_COOKIE) != share_code() and request.url.path not in ("/api/connect", "/connect.html", "/style.css",
                                                                                         "/icon-64.png", "/icon.ico") \
                and not request.url.path.startswith("/fonts/"):
            if request.url.path in ("/", "/index.html"):
                return FileResponse(os.path.join(APP_DIR, "static", "connect.html"), headers={"Cache-Control": "no-cache"})
            return JSONResponse({"detail": "Enter the code from the PC that renders first."}, 401)
    else:
        if host not in LOOPBACK:
            return JSONResponse({"detail": "Forbidden host."}, 403)
        if origin and urlparse(origin).hostname not in LOOPBACK:
            return JSONResponse({"detail": "Forbidden origin."}, 403)
    response = await call_next(request)
    if not request.url.path.startswith(("/images/", "/thumbs/", "/inputs/")):
        response.headers["Cache-Control"] = "no-cache"  # so an updated app shows on the next load
    return response


@app.post("/api/connect")
async def connect(request: Request):
    """A browser on another PC gives the code once, and keeps it in a cookie."""
    body = await request.json()
    who = request.client.host if request.client else "?"
    if _wrong.get(who, 0) >= 20:
        raise HTTPException(429, "Too many wrong codes. Restart Nano on the PC that renders to try again.")
    code = re.sub(r"[^A-Z0-9]", "", str(body.get("code", "")).upper())
    if code != share_code():
        _wrong[who] = _wrong.get(who, 0) + 1
        time.sleep(0.5)
        raise HTTPException(403, "That code doesn't match. It's in All settings under Network on the PC that renders.")
    _wrong.pop(who, None)
    response = JSONResponse({"ok": True})
    response.set_cookie(KEY_COOKIE, code, max_age=60 * 60 * 24 * 365, httponly=True, samesite="strict")
    return response


@app.get("/api/network")
def network():
    return network_json()


@app.post("/api/network")
async def set_network(request: Request):
    """Turns sharing on or off, or makes a new code, which signs out every other PC."""
    local_only(request)
    body = await request.json()
    if body.get("new_code"):
        LOCAL.pop("share_code", None)
        paths.save_local_settings({"share_code": None})
        share_code()
    if "on" in body:
        on = bool(body["on"])
        LOCAL["share_on_network"] = on
        paths.save_local_settings({"share_on_network": on})
        await asyncio.to_thread(start_sharing if on else stop_sharing)
    return network_json()


@app.get("/api/schema")
def schema():
    return {"fields": FIELDS, "terms": TERMS}


@app.get("/api/models")
def model_list():
    def kind(name):
        if name in (models.PARTNER, models.DEPTH, models.RIFE):
            return "partner"
        return "upscaler" if name == models.UPSCALER else "helper" if name == "secondary" else "diffusion"
    return {"models": [{"name": n, "label": models.LABELS.get(n, n), "kind": kind(n), "about": models.ABOUT[n],
                        "size_gb": models.FILES[n]["size_gb"],
                        "downloaded": os.path.exists(models.path_of(n))} for n in models.FILES]}


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
    return {"show": is_local(request) and not paths.local_settings().get("welcome_done")}


@app.post("/api/first-run")
def first_run_done(request: Request):
    local_only(request)
    paths.save_local_settings({"welcome_done": True})
    return {"show": False}


@app.get("/api/status")
def status(request: Request):
    with lock:
        task = task_json(current)
        fetching = downloads_json()
    system_free = shutil.disk_usage(os.environ.get("SystemDrive", "C:") + os.sep).free
    idle_gpu, idle_ram = idle_memory()
    return {"gpu": gpu_info(), "ram": ram_info(), "other_gpu_app": other_gpu_app(), "disk": disk_info(),
            "idle": {"gpu": idle_gpu, "ram": idle_ram}, "local": is_local(request), "machine": socket.gethostname(),
            "version": paths.VERSION, "system_free_gb": round(system_free / 2**30, 1), "task": task, "downloads": fetching}


@app.post("/api/plan")
def render_plan(body: dict):
    """The memory plan for these settings, and the settings the others rule out, for the page to show."""
    settings = {k: v for k, v in body.get("settings", {}).items() if k in KNOWN_FIELDS}
    with lock:
        result = memory_plan(settings)
    blocked = rules.blocked(settings)
    return {**result, "blocked": blocked, "conflicts": rules.conflicts(settings, ruled=blocked),
            "clashing": rules.clashing(settings, ruled=blocked)}


@app.post("/api/render")
async def render(request: Request):
    body = await request.json()
    prompts = [p.strip() for p in body.get("prompts", []) if isinstance(p, str) and p.strip()]
    if not prompts:
        raise HTTPException(400, "Add a prompt first.")
    with lock:
        if current and current.proc.poll() is None:
            raise HTTPException(409, "Something is already running.")
        settings = {k: v for k, v in body.get("settings", {}).items() if k in KNOWN_FIELDS}
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
        handoff_file = str(settings.get("handoff_file") or "")
        if handoff_file and float(settings.get("handoff_share", DEFAULTS["handoff_share"]) or 0) > 0:
            if not handoff_file.lower().endswith(".safetensors") or not os.path.isfile(handoff_file):
                raise HTTPException(400, "The handoff model must be a .safetensors checkpoint that exists. Choose it again in Handoff.")
        for name in needed:
            if name in models.FILES and not os.path.exists(models.path_of(name)):
                raise HTTPException(400, f"{name} isn't downloaded yet. Get it from Models in All settings.")
        if body.get("resume_clip"):
            if settings.get("animation_mode", "None") == "None":
                raise HTTPException(400, "Set Animation to carry on a clip.")
            settings["resume_clip"] = safe_output_path(str(body["resume_clip"]))
        batch = re.sub(r"[^A-Za-z0-9 _-]+", "", str(body.get("batch_name", "Disco"))).strip() or "Disco"
        settings["batch_name"] = batch
        job_id, job_dir = new_job_dir("render")
        requested = body.get("options", {})
        # Offload is chosen here, from the same plan the page draws, unless a script asks for it outright.
        offload = requested.get("offload")
        if offload is None:
            offload = bool(memory_plan(settings).get("offload"))
        options = {"offload": bool(offload),
                   "init_fit": requested.get("init_fit") if requested.get("init_fit") in ("fit", "fill") else "crop",
                   "output_dir": OUT_DIR}
        with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as f:
            json.dump({"settings": settings, "prompts": prompts, "options": options}, f, indent=1)
        spawn("render", prompts[0][:90], [PYTHON, "-B", os.path.join(APP_DIR, "renderer.py"), "--job", job_dir],
              job_id, job_dir)
    return {"id": job_id}


@app.post("/api/stop")
def stop():
    with lock:
        if not current or current.proc.poll() is not None:
            return {"ok": False}
        if current.kind == "render" and current.sampling_started:
            open(os.path.join(current.job_dir, "stop"), "w").close()  # ends cleanly at the next step
            current.message = "Stopping after this step"
        else:
            current.proc.terminate()  # still loading, nothing to preserve
            current.state, current.message = "stopped", FRIENDLY["stopped"]
    return {"ok": True}


@app.get("/api/preview/{job_id}")
def preview(job_id: str):
    if not re.fullmatch(r"[0-9A-Za-z-]+", job_id):
        raise HTTPException(400, "Bad job id.")
    path = os.path.join(JOBS_DIR, job_id, "preview.png")
    if not os.path.isfile(path):
        raise HTTPException(404, "No preview yet.")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


GALLERY_PER_FOLDER = 200
RUN = re.compile(r"(.*\(\d+\))_")  # "Disco(3)_0.png" belongs to run "Disco(3)"


def top_folder(path):
    """The gallery folder an image is in: the first folder under the output folder, or "" for none."""
    parts = os.path.relpath(path, OUT_DIR).split(os.sep)
    return parts[0] if len(parts) > 1 else ""


def gallery_folders():
    try:
        return sorted(e.name for e in os.scandir(OUT_DIR) if e.is_dir() and e.name.lower() not in ("partials",) and not e.name.endswith("_frames"))
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
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        raise HTTPException(400, f"There is already a folder named {name}.")
    os.makedirs(path)
    return {"folder": name}


@app.post("/api/gallery/move")
def move_to_folder(body: dict):
    """Moves an image into another gallery folder, with what belongs to it: its upscales and handoff picture,
    and for a clip its videos and frames. The run's settings go along, or are copied if other images of the run
    stay behind. If the name is taken there, the run gets the next free number."""
    source = safe_output_path(str(body.get("rel", "")))
    folder = str(body.get("folder", ""))
    target = os.path.join(OUT_DIR, clean_folder_name(folder)) if folder else OUT_DIR
    if not os.path.isdir(target):
        raise HTTPException(404, "That folder no longer exists.")
    here, name = os.path.split(source)
    if os.path.normcase(os.path.abspath(here)) == os.path.normcase(os.path.abspath(target)):
        return {"moved": 0}
    with lock:
        if current and current.proc.poll() is None and current.kind == "render":
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
    return {"moved": len(moving), "url": image_url(moved_to)}


def scan_gallery():
    items = []
    for folder, dirs, files in os.walk(OUT_DIR):
        # A clip's frames are thousands of PNGs, and the gallery shows its last frame from the folder above.
        dirs[:] = [d for d in dirs if not d.endswith("_frames") and d.lower() != "partials"]
        for name in files:
            if re.search(r"(\(\d+\)_\d+|_x\d+)\.png$", name):
                path = os.path.join(folder, name)
                item = {"url": image_url(path), "thumb": "/thumbs/" + image_url(path)[len("/images/"):], "name": name,
                        "batch": os.path.basename(folder), "folder": top_folder(path),
                        "rel": os.path.relpath(path, OUT_DIR).replace(os.sep, "/"), "mtime": os.path.getmtime(path)}
                run = re.match(r"(.*\(\d+\))_0\.png$", name)
                for kind in ("mp4", "webp", "gif") if run else ():  # an animation's last frame stands for it
                    clip = os.path.join(folder, f"{run.group(1)}.{kind}")
                    if os.path.isfile(clip):
                        item["video"], item["video_kind"] = image_url(clip), kind
                        break
                items.append(item)
    items.sort(key=lambda i: i["mtime"], reverse=True)
    return items


@app.get("/api/image-settings")
def image_settings(rel: str):
    """The settings a past image was made with, from the file saved beside it."""
    path = safe_output_path(rel)
    match = re.match(r"(.*\(\d+\))_", os.path.basename(path))
    if not match:
        raise HTTPException(404, "No settings saved for this image.")
    txt = os.path.join(os.path.dirname(path), match.group(1) + "_settings.txt")
    if not os.path.isfile(txt):
        raise HTTPException(404, "No settings saved for this image.")
    with open(txt, encoding="utf-8", errors="replace") as f:
        settings = json.load(f)
    if settings.get("init_image"):
        settings["init_size"] = image_size(settings["init_image"])
    if not settings.pop("range_scale_works", False):
        settings.pop("range_scale", None)  # it did nothing when this image was made, so reusing it mustn't turn it on
    return settings


@app.get("/api/folders")
def folders():
    return {"output_dir": OUT_DIR, "upscale_dirs": UPSCALE_DIRS, "checkpoint_dirs": CHECKPOINT_DIRS}


def writable_folder(path):
    path = os.path.abspath(path)
    try:
        os.makedirs(path, exist_ok=True)
        tempfile.TemporaryFile(dir=path).close()
    except OSError:
        raise HTTPException(400, f"Can't save to {path}.")
    return path


@app.post("/api/folders")
async def set_folders(request: Request):
    global OUT_DIR, UPSCALE_DIRS, CHECKPOINT_DIRS, current
    local_only(request)
    body = await request.json()
    with lock:
        if current and current.proc.poll() is None:
            raise HTTPException(409, "Wait for the current job to finish.")
        wanted = str(body.get("output_dir", OUT_DIR)).strip() or paths.OUTPUT
        if not os.path.isabs(wanted):
            raise HTTPException(400, "Use a full folder path, such as D:\\Art\\Disco.")
        output_dir = writable_folder(wanted)
        dirs = [d.strip() for d in body.get("upscale_dirs", UPSCALE_DIRS) if isinstance(d, str) and d.strip()]
        for d in dirs:
            if d not in UPSCALE_DIRS and not os.path.isdir(d):  # a known drive may just be unplugged
                raise HTTPException(400, f"Folder not found: {d}")
        checkpoint_dirs = [d.strip() for d in body.get("checkpoint_dirs", CHECKPOINT_DIRS) if isinstance(d, str) and d.strip()]
        for d in checkpoint_dirs:
            if d not in CHECKPOINT_DIRS and not os.path.isdir(d):
                raise HTTPException(400, f"Folder not found: {d}")
        if output_dir != OUT_DIR:
            current = None  # its images were in the old folder
            _disco_size["at"] = 0  # measure the new drive
        OUT_DIR, UPSCALE_DIRS, CHECKPOINT_DIRS = output_dir, dirs, checkpoint_dirs
        paths.save_local_settings({"output_dir": output_dir, "upscale_dirs": dirs, "checkpoint_dirs": checkpoint_dirs})
    return folders()


# Windows only lets the app in use bring a window to the front, and that app is the browser, so a picker
# opened from here would land behind it with no taskbar button. The owner window below is shown, topmost,
# on the taskbar and pulled to the front, and a dialog it owns stays above it.
PICK_OWNER = """
Add-Type -AssemblyName System.Windows.Forms
Add-Type -Namespace Nano -Name Win -MemberDefinition @'
[DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
[DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, IntPtr p);
[DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
[DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool attach);
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
[DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
[DllImport("user32.dll")] public static extern void keybd_event(byte k, byte s, uint f, UIntPtr e);
'@
$owner = New-Object System.Windows.Forms.Form -Property @{
  Text = 'Disco Diffusion Nano'; TopMost = $true; ShowInTaskbar = $true; FormBorderStyle = 'None'
  StartPosition = 'CenterScreen'; Width = 1; Height = 1; Opacity = 0
}
if ($env:PICK_ICON -and (Test-Path $env:PICK_ICON)) { $owner.Icon = New-Object System.Drawing.Icon $env:PICK_ICON }
$owner.Show()
$front = [Nano.Win]::GetForegroundWindow()
$theirs = [Nano.Win]::GetWindowThreadProcessId($front, [IntPtr]::Zero)
$mine = [Nano.Win]::GetCurrentThreadId()
[void][Nano.Win]::AttachThreadInput($mine, $theirs, $true)
[Nano.Win]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero)  # Alt down lifts Windows' lock on the foreground
[void][Nano.Win]::SetForegroundWindow($owner.Handle)
[void][Nano.Win]::BringWindowToTop($owner.Handle)
[Nano.Win]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)  # Alt up lands here, so the browser's menu stays shut
[void][Nano.Win]::AttachThreadInput($mine, $theirs, $false)
$owner.Activate()
[Console]::OutputEncoding = [Text.Encoding]::UTF8
"""

PICK_FOLDER = PICK_OWNER + """
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = $env:PICK_TITLE
$dialog.SelectedPath = $env:PICK_START
if ($dialog.ShowDialog($owner) -eq 'OK') { $dialog.SelectedPath }
$owner.Close()
"""


def run_picker(script, **values):
    env = dict(os.environ, PICK_ICON=os.path.join(APP_DIR, "static", "icon.ico"), **values)
    result = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or result.stderr.strip():
        print(f"The file picker failed: {result.stderr.strip()[-600:]}", flush=True)
    return {"path": result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None}


def local_only(request: Request):
    """A Windows dialog opens on the PC that runs Nano, so only a browser on that PC may open one."""
    if not is_local(request):
        raise HTTPException(403, "Choose this on the PC that renders. From here, drop files onto the page instead.")


@app.post("/api/pick-folder")
def pick_folder(body: dict, request: Request):
    """Opens Windows' folder picker. A browser page can't read a real path from its own."""
    local_only(request)
    return run_picker(PICK_FOLDER, PICK_TITLE=str(body.get("title", "Choose a folder")), PICK_START=str(body.get("start", "")))


PICK_FILE = PICK_OWNER + """
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = $env:PICK_TITLE
$dialog.Filter = $env:PICK_FILTER
if ($env:PICK_START) { $dialog.InitialDirectory = $env:PICK_START }
if ($dialog.ShowDialog($owner) -eq 'OK') { $dialog.FileName }
$owner.Close()
"""


def pick_file(start, title, kinds):
    start = os.path.dirname(str(start))
    return run_picker(PICK_FILE, PICK_START=start if os.path.isdir(start) else "", PICK_TITLE=title, PICK_FILTER=kinds)


@app.post("/api/pick-checkpoint")
def pick_checkpoint(body: dict, request: Request):
    """Opens Windows' file picker for the handoff model. The checkpoint is read where it is, never copied, and
    its folder joins the checkpoint folders so the list finds its neighbours too."""
    local_only(request)
    picked = pick_file(body.get("start", ""), "Choose a Stable Diffusion 1.5 or SDXL checkpoint", "Checkpoints|*.safetensors")
    if picked["path"]:
        add_checkpoint_dir(os.path.dirname(picked["path"]))
    return picked


def add_checkpoint_dir(folder):
    global CHECKPOINT_DIRS
    if folder and os.path.isdir(folder) and folder not in CHECKPOINT_DIRS:
        CHECKPOINT_DIRS = CHECKPOINT_DIRS + [folder]
        paths.save_local_settings({"checkpoint_dirs": CHECKPOINT_DIRS})


@app.get("/api/checkpoints")
def checkpoints():
    """Stable Diffusion checkpoints in the checkpoint folders, for the Handoff list. Like the upscalers, they are
    read where they are, so a browser on another PC picks from the files on this one."""
    found = []
    for d in CHECKPOINT_DIRS:
        try:
            names = sorted(os.listdir(d), key=str.lower)
        except OSError:
            continue  # a drive that is unplugged for now
        for n in names:
            path = os.path.join(d, n)
            if n.lower().endswith(".safetensors") and os.path.isfile(path) and path not in [f["path"] for f in found]:
                found.append({"path": path, "name": n[:-12], "sdxl": budget._is_sdxl(path)})
    return {"checkpoints": found, "folders": CHECKPOINT_DIRS}


@app.get("/api/curve")
def curve(text: str, frames: int = 60, shape: str = "Linear"):
    """Every frame's value for one keyframe setting, for the curve editor."""
    try:
        return {"values": keyframes.series(text, min(max(frames, 1), 10000), shape)}
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return {"error": str(e)}


def is_local(request):
    return bool(request.client) and request.client.host in ("127.0.0.1", "::1")


@app.get("/api/upscalers")
def upscalers():
    names = {}
    for d in UPSCALE_DIRS + [BUILT_IN_UPSCALERS]:
        if os.path.isdir(d):
            for n in os.listdir(d):
                if n.endswith((".pth", ".safetensors")) and n not in names:
                    names[n] = os.path.join(d, n)
    return {"models": sorted(names)}


@app.post("/api/upscale")
async def upscale(request: Request):
    body = await request.json()
    path = safe_output_path(str(body.get("rel", "")))
    model = next((os.path.join(d, str(body.get("model", ""))) for d in UPSCALE_DIRS + [BUILT_IN_UPSCALERS]
                  if os.path.isfile(os.path.join(d, os.path.basename(str(body.get("model", "")))))), None)
    if model is None or os.path.basename(model) != str(body.get("model")):
        raise HTTPException(400, "Upscaler not found.")
    with lock:
        if current and current.proc.poll() is not None or current is None:
            job_id, job_dir = new_job_dir("upscale")
            spawn("upscale", os.path.basename(path),
                  [PYTHON, "-B", os.path.join(APP_DIR, "upscale.py"), "--input", path, "--model", model], job_id, job_dir)
            return {"id": job_id}
    raise HTTPException(409, "Something is already running.")


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
    part = path + ".part"
    with open(part, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    os.replace(part, path)


@app.post("/api/init-image")
async def init_image(request: Request, name: str):
    """Start, ghost and style images are uploaded, so they work the same from a browser on another PC."""
    safe = safe_name(name)
    if not safe.lower().endswith(IMAGE_KINDS):
        raise HTTPException(400, "Use a PNG, JPG or WEBP image.")
    path = os.path.join(INIT_DIR, safe)
    await receive(request, path)
    if image_size(path) is None:
        os.remove(path)
        raise HTTPException(400, f"{name} isn't an image Nano can read.")
    return {"path": path, "name": safe, "size": image_size(path), "url": input_url(path)}


@app.post("/api/upload-video")
async def upload_video(request: Request, name: str):
    """A video for Video Input, kept with the start images."""
    safe = safe_name(name)
    if not safe.lower().endswith(VIDEO_KINDS):
        raise HTTPException(400, "Use an MP4, MOV, WEBM, MKV, AVI or GIF video.")
    path = os.path.join(INIT_DIR, "videos", safe)
    await receive(request, path)
    return {"path": path, "name": safe, "url": input_url(path)}


@app.post("/api/prompt-images")
async def prompt_image(request: Request, name: str, group: str = ""):
    """One image of a set of image prompts. The page sends each file of a drop to the same new folder."""
    safe = safe_name(name)
    if not safe.lower().endswith(IMAGE_KINDS):
        raise HTTPException(400, f"{name} isn't a PNG, JPG or WEBP image.")
    group = re.sub(r"[^A-Za-z0-9_-]+", "", group)[:40] or time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(INIT_DIR, "prompt_images", group, safe)
    await receive(request, path)
    if image_size(path) is None:
        os.remove(path)
        raise HTTPException(400, f"{name} isn't an image Nano can read.")
    return {"folder": os.path.dirname(path), "group": group}


@app.get("/api/prompt-images")
def prompt_image_list(folder: str):
    """The images in an image prompt folder, with previews when they were uploaded here."""
    if not os.path.isdir(folder):
        return {"images": [], "missing": True}
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_KINDS))
    return {"images": [{"name": n, "url": input_url(os.path.join(folder, n))} for n in names], "missing": False}


def input_url(path):
    """A URL for a file under the inputs folder, for the page to preview. Files elsewhere get none."""
    if not inside(path, INIT_DIR):
        return None
    return "/inputs/" + os.path.relpath(os.path.realpath(path), os.path.realpath(INIT_DIR)).replace(os.sep, "/")


@app.get("/api/input-preview")
def input_preview(path: str):
    """A preview URL and the size of an image the settings name, such as a ghost image."""
    found = os.path.isfile(path)
    return {"url": input_url(path) if found else None, "size": image_size(path) if found else None, "missing": not found}


@app.get("/inputs/{rel:path}")
def inputs(rel: str):
    path = os.path.realpath(os.path.join(INIT_DIR, rel))
    if not inside(path, INIT_DIR) or not os.path.isfile(path):
        raise HTTPException(404, "File not found.")
    return FileResponse(path)


@app.post("/api/init-from-output")
async def init_from_output(request: Request):
    body = await request.json()
    path = safe_output_path(str(body.get("rel", "")))
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(path))
    target = os.path.join(INIT_DIR, name)
    shutil.copyfile(path, target)
    return {"path": target, "name": name, "size": image_size(target), "url": input_url(target)}


# Opens a folder in Explorer and brings its window to the front, for the same reason as the pickers.
OPEN_FOLDER = """
Add-Type -Namespace Nano -Name Win -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
[DllImport("user32.dll")] public static extern void keybd_event(byte k, byte s, uint f, UIntPtr e);
'@
$shell = New-Object -ComObject Shell.Application
$want = (Resolve-Path -LiteralPath $env:OPEN_PATH).Path.TrimEnd('\\')
$shell.Open($want)
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Milliseconds 100
  foreach ($w in $shell.Windows()) {
    try { $at = $w.Document.Folder.Self.Path.TrimEnd('\\') } catch { continue }
    if ($at -eq $want) {
      $h = [IntPtr]$w.HWND
      [Nano.Win]::ShowWindow($h, 9) | Out-Null
      [Nano.Win]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero)
      [Nano.Win]::SetForegroundWindow($h) | Out-Null
      [Nano.Win]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)
      exit
    }
  }
}
"""


@app.post("/api/open-output")
def open_output(request: Request):
    local_only(request)
    env = dict(os.environ, OPEN_PATH=OUT_DIR)
    subprocess.Popen(["powershell", "-NoProfile", "-STA", "-Command", OPEN_FOLDER], env=env,
                     creationflags=subprocess.CREATE_NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"ok": True}


@app.get("/images/{rel:path}")
def images(rel: str):
    return FileResponse(safe_output_path(rel))  # not a static mount, because the folder can change


THUMBS = os.path.join(ROOT, "cache", "thumbs")
THUMB_SIZE = 320


@app.get("/thumbs/{rel:path}")
def thumbs(rel: str):
    """A small JPEG of a finished image, made once and kept. The gallery would otherwise load every image
    at full size, and an upscaled one can be 4096 pixels wide."""
    path = safe_output_path(rel)
    stamp = os.stat(path)
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)[-150:] + f"_{int(stamp.st_mtime)}_{stamp.st_size}.jpg"
    thumb = os.path.join(THUMBS, key)
    if not os.path.isfile(thumb):
        os.makedirs(THUMBS, exist_ok=True)
        with Image.open(path) as image:
            image.draft("RGB", (THUMB_SIZE * 2, THUMB_SIZE * 2))
            small = image.convert("RGB")
            small.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        part = thumb + ".part"
        small.save(part, "JPEG", quality=86)
        os.replace(part, thumb)
    return FileResponse(thumb, headers={"Cache-Control": "max-age=31536000, immutable"})


app.mount("/", StaticFiles(directory=os.path.join(APP_DIR, "static"), html=True), name="ui")


def open_when_ready():
    while not port_open(PORT):
        time.sleep(0.1)
    print(f"Ready at {URL}. Close this window to shut the app down.", flush=True)
    if LOCAL.get("share_on_network"):
        start_sharing()
        for address in _network["addresses"]:
            print(f"Other PCs on the network can render here at http://{address}:{PORT} with the code in All settings.", flush=True)
    if not os.environ.get("NANO_NO_BROWSER"):
        webbrowser.open(URL)


if __name__ == "__main__":
    if not os.path.exists(PYTHON):
        sys.exit(f"Environment not found at {PYTHON}")
    if port_open(PORT):
        print(f"Disco Diffusion Nano is already running at {URL}", flush=True)
        webbrowser.open(URL)
        sys.exit(0)
    print("Starting the server", flush=True)
    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
