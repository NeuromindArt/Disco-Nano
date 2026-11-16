"""The jobs that use the GPU, one at a time: renders, in the render worker that stays open between them, upscales,
each in a process of its own, and the start image's inpaint tries."""
import json
import os
import time
from collections import deque

import atomic
import budget
import procs
from web.base import lock
from web import places, system


FRIENDLY = {
    "oom": "The GPU ran out of memory. Close other GPU apps, or lower the size or use fewer CLIP models.",
    "error": "The render failed. Details are in the log below.",
    "stopped": "Stopped.",
}


class Task:
    """The one render or upscale allowed to run at a time. An upscale has a process of its own. A render runs in
    the render worker, which stays open after it."""

    def __init__(self, kind, job_id, job_dir, proc, label, worker=None):
        self.kind, self.job_id, self.job_dir, self.proc, self.label = kind, job_id, job_dir, proc, label
        self.worker, self.ended, self.keep_models = worker, False, True
        self.state, self.message = "starting", "Starting"
        self.step = self.total = self.batch = 0
        self.at = None  # how far through the whole schedule the last step was, from 0 to 1
        self.n_batches = 1
        self.step_times = deque(maxlen=12)
        self.started = time.time()
        self.sampling_started = None
        self.paused = False  # a status message came between steps
        self.holding = False  # paused by the page, for regions to be drawn or something erased
        self.pause_for = ""  # "regions" or "erase", so a page opened during the pause shows the right editor
        self.tries = {"id": 0, "made": [], "asked": 0, "error": None}  # Fix's inpaint tries during this pause
        self.seed = None
        self.images = []
        self.preview_rev = 0
        self.peak_vram = None
        self.error = None
        self.log = deque(maxlen=60)
        self.finished_at = None

    def running(self):
        if self.worker is not None:
            return not self.ended and self.proc.poll() is None
        return self.proc.poll() is None


class Worker:
    """The render process, kept open between renders so the models it loaded wait in RAM for the next one.
    It reads job folders on stdin. Closing the app closes stdin, which ends it and frees its memory. A render
    that needs what this process can't change, such as Exact repeat, starts a new one."""

    def __init__(self, sig):
        self.sig, self.task, self.log = sig, None, None
        self.ending = False  # closing, or about to exit after a failed render: it takes no more jobs
        self.pid = None  # the Python process itself, which a venv's launcher starts as a child of proc
        self.jobs = 0  # renders handed to it, so a stop knows whether models from an earlier one wait in it
        self.child = procs.Child("renderer.py", "--serve", stdin=True, on_line=self.read, on_exit=self.exited,
                                 env={"NANO_EXACT": "1" if sig["exact_repeat"] else "0", "NANO_LARGE": "1" if sig["large"] else "0"})
        self.proc = self.child.proc

    def alive(self):
        return self.child.alive() and not self.ending

    def run(self, task):
        # the log first: the reader writes to it as soon as it sees a task
        self.log = open(os.path.join(task.job_dir, "run.log"), "a", encoding="utf-8", errors="replace")
        self.task = task
        self.jobs += 1
        self.child.write(task.job_dir)

    def read(self, line, event):
        if event and event.get("type") == "ready":
            self.pid = event.get("pid") if isinstance(event.get("pid"), int) else None
        task, log = self.task, self.log
        if task is None:
            return  # between renders, such as its ready line
        log.write(line + "\n")
        log.flush()
        follow(task, line, event)
        if event and event.get("type") == "finished":
            self.done(task)

    def exited(self, code):
        task = self.task
        if task is not None and not task.ended:
            ended_unexpectedly(task, code)
            self.done(task)

    def done(self, task):
        with lock:
            task.ended = True
            task.finished_at = time.time()
            system.forget_gpu_reading()
        self.task = None
        if self.log:
            self.log.close()
            self.log = None
        if task.state in ("oom", "error"):
            self.ending = True  # renderer.py stops serving after a failed render, so this process is on its way out
        if not task.keep_models:
            self.close()

    def close(self):
        """Ends the worker. Closing stdin lets it finish cleanly; one still busy loading is ended outright, and one
        paused ends at once, as it sees its stdin close."""
        self.ending = True
        self.child.close(grace=0 if self.task is not None else 10)


_worker = None


def render_in_worker(label, job_id, job_dir, sig, keep_models):
    """Hands a render to the worker, starting a new one when there is none or it can't take this render."""
    global current, _worker
    if _worker is not None and (not _worker.alive() or _worker.sig != sig):
        _worker.close()
        _worker = None
    starting_job()
    if _worker is None:
        _worker = Worker(sig)
    current = Task("render", job_id, job_dir, _worker.proc, label, worker=_worker)
    current.keep_models = keep_models
    _worker.run(current)


def close_worker():
    if _worker is not None:
        _worker.close()


current = None


def follow(task, line, event):
    """A line of a job's output: an event changes the task, and the rest is kept for the log the page shows when it
    fails."""
    if event is not None:
        handle_event(task, event)
    elif line.strip():
        task.log.append(line[-300:])


def ended_unexpectedly(task, code):
    """A job's process ended. One that never said how it ended failed, and says so."""
    with lock:
        if task.state not in ("done", "stopped", "oom", "error"):
            task.state = "error"
            task.message = FRIENDLY["error"]
            task.error = f"Worker exited unexpectedly (code {code})."


def handle_event(task, e):
    kind = e.get("type")
    with lock:
        if kind == "status":
            if task.holding:  # an inpaint try painting during a pause: the render is still paused
                task.message = e["message"]
            else:
                task.state, task.message = "loading", e["message"]
                task.paused = True  # set by a status between steps, such as a handoff, and cleared at the next step
        elif kind == "sampling":
            task.state, task.message = "rendering", "Upscaling" if task.kind == "upscale" else "Rendering"
            task.total, task.n_batches, task.seed = e["total_steps"], e["n_batches"], e.get("seed")
            task.sampling_started = e["t"]
        elif kind == "step":
            if task.paused and task.sampling_started:
                task.state, task.message, task.paused = "rendering", "Upscaling" if task.kind == "upscale" else "Rendering", False
            task.step, task.total, task.batch, task.at = e["step"], e["total"], e["batch"], e.get("at")
            task.step_times.append(e["t"])
        elif kind == "paused":
            task.state, task.holding = "paused", True
            task.message = "Paused. Paint over what to fix." if task.pause_for == "erase" else "Paused. Draw regions, then Continue."
        elif kind == "tried":
            if e.get("id") == task.tries["id"]:
                if e.get("error"):
                    task.tries["error"], task.tries["asked"] = e["error"], len(task.tries["made"])
                else:
                    task.tries["made"].append(e["k"])
            if task.holding and task.tries["asked"] <= len(task.tries["made"]):
                task.message = "Paused. Paint over what to fix."  # not the last try's step count
        elif kind == "resumed":
            task.state, task.message, task.holding = "rendering", "Rendering", False
            task.tries.update(made=[], asked=0, error=None)  # a pause's tries are for its own picture
            task.step_times.clear()  # the wait isn't render time
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
    if spp and task.total and not task.holding:
        left = max(task.total - task.step, 0) + (task.n_batches - task.batch - 1) * task.total
        eta = round(left * spp)
    return {
        "kind": task.kind, "id": task.job_id, "label": task.label, "state": task.state, "message": task.message,
        "step": task.step, "total": task.total, "batch": task.batch, "n_batches": task.n_batches, "at": task.at,
        "seconds_per_step": round(spp, 2) if spp else None, "eta_s": eta, "seed": task.seed,
        "elapsed_s": round((task.finished_at or time.time()) - task.started), "preview_rev": task.preview_rev,
        "images": [places.image_url(p) for p in task.images], "peak_vram_gb": task.peak_vram, "error": task.error,
        "log": list(task.log)[-14:] if task.state in ("error", "oom") else [],
        "pausing": task.kind == "render" and os.path.exists(os.path.join(task.job_dir, "pause")),
        "pause_for": task.pause_for if task.kind == "render" else "",
        "tries": {"id": task.tries["id"], "made": [f"/api/try/{task.job_id}/{task.tries['id']}/{k}" for k in task.tries["made"]],
                  "painting": task.tries["asked"] > len(task.tries["made"]), "error": task.tries["error"]} if task.kind == "render" else None,
        "active": task.running(),
    }


def worker_held():
    """The VRAM and RAM the waiting render worker holds: its CUDA context and the models it kept. The next render
    either reuses it or replaces it, so none of it belongs to other apps, and the plan already counts a context."""
    worker = _worker
    if worker is None or not worker.alive() or worker.task is not None:
        return 0.0, 0.0
    return budget.CONTEXT_GB, system.process_ram_gb(worker.pid or worker.proc.pid)


def memory_plan(settings):
    gpu, ram = system.idle_memory(current)
    if not gpu or not ram:
        return {}
    vram_held, ram_held = worker_held()
    gpu = {**gpu, "used_gb": max(gpu["used_gb"] - vram_held, 0.0)}
    ram = {**ram, "used_gb": max(ram["used_gb"] - ram_held, 0.0)}
    return budget.plan(settings, gpu, ram)


def spawn(kind, label, script, args, job_id, job_dir):
    """Starts a job in a process of its own, such as an upscale, as the one running task."""
    global current
    starting_job()
    task = Task(kind, job_id, job_dir, None, label)

    def ended(code):
        ended_unexpectedly(task, code)
        with lock:
            task.finished_at = time.time()
            system.forget_gpu_reading()

    task.proc = procs.Child(script, *args, log_path=os.path.join(job_dir, "run.log"), on_line=lambda line, event: follow(task, line, event),
                            on_exit=ended).proc
    current = task


def starting_job():
    """Before a render or upscale starts: the start image's Inpaint model leaves the GPU and RAM to it, and what the
    PC uses without it is read, so what the job adds is measured from there."""
    close_tries()
    system.take_idle_reading()


def job_settings(task):
    try:
        with open(os.path.join(task.job_dir, "job.json"), encoding="utf-8") as f:
            return json.load(f)["settings"]
    except (OSError, ValueError, KeyError):
        return {}


_fixing = 0  # fixes of the start image on the GPU now, which a render or upscale waits for


def fixing_on_gpu():
    """Starts a fix of the start image on the GPU, unless a job is running there. Returns whether it did: if so, a
    render or upscale waits until fixed_on_gpu() says it's done. Called holding the lock."""
    global _fixing
    if current and current.running():
        return False
    _fixing += 1
    return True


def fixed_on_gpu():
    """A fix of the start image on the GPU ended. Called holding the lock."""
    global _fixing
    _fixing -= 1
    system.forget_gpu_reading()


def why_busy():
    """Why a render or upscale can't start now, or None. Called holding the lock."""
    if current and current.running():
        return "Something is already running."
    if _fixing:
        return "The start image is being fixed. Start once it's done."
    return None


class Tries:
    """The start image's inpaint tries, to choose from as during a render. erase.py --serve paints them, in a process
    that keeps the Inpaint model loaded while the editor is open, so More tries and the next fix don't wait for it to
    load again. It ends when the editor closes, a render starts or the app closes, and by itself when left idle."""
    asks = 0  # ask ids go on counting across processes, so a page never takes a new process's tries for an old one's

    def __init__(self):
        self.job_id, self.folder = places.new_job_dir("tries")
        self.id, self.made, self.asked, self.error, self.message, self.last = 0, [], 0, None, "", None
        self.child = procs.Child("erase.py", "--serve", self.folder, stdin=True, log_path=os.path.join(self.folder, "run.log"),
                                 on_line=self.read, on_exit=self.exited)

    def alive(self):
        return self.child.alive()

    def read(self, line, e):
        if e is None:
            return
        with lock:
            if e.get("type") == "status":
                self.message = str(e.get("message") or "")
            elif e.get("type") == "tried" and e.get("id") == self.id:
                if e.get("error"):
                    self.error, self.asked = e["error"], len(self.made)
                else:
                    self.made.append(e["k"])

    def exited(self, code):
        with lock:
            if self.asked > len(self.made):  # it ended in the middle of a try
                self.error, self.asked = f"Inpainting stopped unexpectedly (code {code}).", len(self.made)

    def ask(self, path, mask, clones, inpainting, count, more):
        """Asks for `count` more tries of the fix, or starts over unless `more`. Returns the ask's id."""
        if not more or not self.last or self.last["image"] != path:
            Tries.asks += 1
            self.id = Tries.asks
            self.made, self.asked = [], 0
            stem = os.path.join(self.folder, str(self.id))
            if mask is not None:
                mask.save(stem + "-erase.png")
            inpainting["mask"].save(stem + "-inpaint.png")
            self.last = {"id": self.id, "image": path, "mask": stem + "-erase.png" if mask is not None else None,
                         "clones": save_clones(clones, self.folder, f"{self.id}-clone"),
                         "inpaint": {**{k: v for k, v in inpainting.items() if k != "chosen"}, "mask": stem + "-inpaint.png"}}
        self.asked += count
        self.error = None
        atomic.write_json(os.path.join(self.folder, "ask.json"), {**self.last, "upto": self.asked})
        return self.id

    def json(self):
        return {"id": self.id, "made": [f"/api/try/{self.job_id}/{self.id}/{k}" for k in self.made],
                "painting": self.asked > len(self.made), "error": self.error, "message": self.message}

    def close(self):
        """Ends the process, which lets the model go: between a try's steps, or outright when it's still loading."""
        self.child.close()


image_tries = None  # the start image's tries process, while the editor over the start image is open


def tries_process():
    """The start image's tries process, started when there is none or it ended. Called holding the lock."""
    global image_tries
    if image_tries is None or not image_tries.alive():
        image_tries = Tries()
    return image_tries


def close_tries():
    global image_tries
    if image_tries is not None:
        image_tries.close()
        image_tries = None


def forget_task():
    """The images moved to another folder, so the last job's image URLs point nowhere. Called holding the lock."""
    global current
    current = None


def save_clones(layers, folder, stem):
    """Saves the clone layers' masks in the folder, and returns them as the renderer reads them."""
    listed = []
    for k, (mask, dx, dy) in enumerate(layers):
        path = os.path.join(folder, f"{stem}-{k}.png")
        mask.save(path)
        listed.append({"mask": path, "dx": dx, "dy": dy})
    return listed
