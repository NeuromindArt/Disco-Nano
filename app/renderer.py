"""Runs one render in its own process and reports progress as @@EVT lines on stdout.

Usage: python renderer.py --job <job_dir>      (job_dir/job.json holds the settings)
       python renderer.py --serve               (stays open; renders each job folder written to stdin)
A file named `stop` in the job folder ends the run at the next step. A file named `pause` holds it after the next
step until the file goes; regions.json, written before it goes, holds the regions to carry on with.
"""
import argparse
import json
import os
import sys
import time
import traceback

import paths


def emit(kind, **data):
    data["type"] = kind
    data["t"] = round(time.time(), 2)
    print("\n@@EVT " + json.dumps(data), flush=True)  # leading newline: a library may have left a part line


class Hooks:
    """How disco.render reports back."""

    def __init__(self, job_dir, progress=False, regions=None):
        self.job_dir = job_dir
        self.stop_file = os.path.join(job_dir, "stop")
        self.pause_file = os.path.join(job_dir, "pause")
        # Regional prompts: boxes with prompts of their own. A pause can change them, which counts up the revision.
        self.regions, self.regions_rev, self.regions_read = regions, 0, None
        self.started = False
        # With Save progress video, every preview is also kept as a frame until its image is saved.
        self.progress = os.path.join(job_dir, "progress") if progress else None
        self.frames = []

    def status(self, message):
        emit("status", message=message)

    def stopped(self):
        return os.path.exists(self.stop_file)

    def step(self, j, total, batch, n_batches, seed):
        if not self.started:
            self.started = True
            emit("sampling", total_steps=total, n_batches=n_batches, seed=seed)
        emit("step", step=j, total=total, batch=batch, n_batches=n_batches)
        if os.path.exists(self.stop_file):
            raise KeyboardInterrupt
        if os.path.exists(self.pause_file):
            self.hold()
        self.read_regions()

    def pausing(self):
        return os.path.exists(self.pause_file)

    def hold(self):
        """Waits while the pause file is there, so regions can be drawn on the preview."""
        emit("paused")
        while os.path.exists(self.pause_file):
            if os.path.exists(self.stop_file):
                raise KeyboardInterrupt
            time.sleep(0.2)
        emit("resumed")

    def read_regions(self):
        """Takes up regions.json when the page has written a new one, which it does as a pause ends."""
        path = os.path.join(self.job_dir, "regions.json")
        try:
            changed = os.path.getmtime(path)
            if changed == self.regions_read:
                return
            self.regions_read = changed
            with open(path, encoding="utf-8") as f:
                regions = json.load(f)
        except (OSError, ValueError):
            return  # none yet, or one being written: carry on with the regions it had
        if regions != self.regions:
            self.regions, self.regions_rev = regions, self.regions_rev + 1

    def preview(self, image):
        image = image.convert("RGB")
        if self.progress:
            os.makedirs(self.progress, exist_ok=True)
            frame = os.path.join(self.progress, f"{len(self.frames):05d}.jpg")
            image.save(frame, quality=92)
            self.frames.append(frame)
        # JPEG, because a preview is written every step: a large PNG takes several times longer to write.
        temp = os.path.join(self.job_dir, "preview.tmp.jpg")
        image.save(temp, quality=90)
        for _ in range(3):
            try:
                os.replace(temp, os.path.join(self.job_dir, "preview.jpg"))
                break
            except PermissionError:  # the server is sending the last preview, and Windows won't replace an open file
                time.sleep(0.05)
        else:
            return  # this preview is skipped, and the next one will get through
        emit("preview")

    def image(self, path):
        if self.frames:
            self.status("Saving the progress video")
            try:
                import animate
                animate.save_progress_video(self.frames + [path], os.path.splitext(path)[0] + "_progress.mp4")
            except Exception as e:  # the image itself is saved, so a failed video is only reported
                print(f"The progress video could not be saved: {e!r}", flush=True)
            for frame in self.frames:
                try:
                    os.remove(frame)
                except OSError:
                    pass
            self.frames = []
        emit("image", path=path)


def signature(settings):
    """What a worker can't change once torch has started. A render with a different one needs a new worker."""
    width, height = settings.get("width_height", (0, 0))
    return {"exact_repeat": bool(settings.get("exact_repeat")), "large": width * height >= 500_000}


def set_up_process(sig):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("TORCH_HOME", os.path.join(paths.HOME, "cache", "torch"))
    if sig["exact_repeat"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # must be set before torch starts
    if sig["large"]:
        # At 1280x768 a 9MB batch of CLIP cuts could land inside a freed 720MB block and pin all of it through
        # the backward pass, leaving no room for the next 720MB block. This stops big free blocks being split
        # for small tensors. It costs about 5% speed, so smaller renders, which have room to spare, skip it.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    for name in paths.CODE_REPOS:
        sys.path.insert(0, os.path.join(paths.HOME, name))


def run_job(job_dir):
    """One render, reported as events. Returns how it ended."""
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    options, settings = job.get("options", {}), job["settings"]
    still = settings.get("animation_mode", "None") == "None"  # a clip is its own video
    hooks = Hooks(job_dir, progress=still and bool(settings.get("save_progress_video")), regions=settings.get("regions"))
    outcome, detail = "done", {}
    cpu_start = time.process_time()
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        threads = int(os.environ.get("DISCO_NANO_THREADS") or settings.get("cpu_threads", 4))
        if threads > 0:  # more threads mostly spin while they wait for the GPU, which slows the rest of the PC
            torch.set_num_threads(threads)
        if settings.get("exact_repeat"):
            torch.use_deterministic_algorithms(True)
        import memory
        cap = memory.limit_vram()
        hooks.status(f"Loading models ({cap:.1f}GB VRAM available)")
        import disco
        memory.half_precision_norm()
        memory.lean_unet()
        # The secondary model skips the big UNet's backward pass, which is all Offload helps with. Offload would
        # still move CLIP off and back every step, and at 1280x768 that left 2GB of VRAM in unusable pieces.
        offload = bool(options.get("offload")) and not settings.get("use_secondary_model")
        memory.offload_activations(offload)
        disco.render(settings, job["prompts"], hooks, offload, options.get("init_fit", "crop"),
                     options.get("output_dir") or paths.OUTPUT)
    except KeyboardInterrupt:
        outcome = "stopped"
    except BaseException as e:  # out of memory included; every failure must reach the UI as an event
        is_oom = type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()
        outcome = "oom" if is_oom else "error"
        detail = {"error": repr(e)[:600]}
        print(traceback.format_exc()[-1800:], flush=True)
    detail["cpu_seconds"] = round(time.process_time() - cpu_start, 1)
    try:
        detail["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        detail["peak_tensors_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)  # reserved can fill to the cap
    except Exception:
        pass
    peak_ram = peak_ram_bytes()
    if peak_ram:
        detail["peak_ram_gb"] = round(peak_ram / 2**30, 2)
    emit("finished", outcome=outcome, **detail)
    return outcome


def serve():
    """A worker that stays open: it reads job folders from stdin, one per line, and renders each in turn. Models
    it loaded wait in RAM for the next render. It ends when the app closes its stdin, and after a render that
    failed, so the next one starts from a clean process."""
    set_up_process({"exact_repeat": os.environ.get("NANO_EXACT") == "1", "large": os.environ.get("NANO_LARGE") == "1"})
    emit("ready", pid=os.getpid())  # a venv's python.exe is a launcher, so the server can't see this process's own id
    for line in sys.stdin:
        job_dir = line.strip()
        if not job_dir:
            continue
        if run_job(os.path.abspath(job_dir)) in ("oom", "error"):
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", help="render this one job folder and exit")
    parser.add_argument("--serve", action="store_true", help="stay open and render job folders read from stdin")
    args = parser.parse_args()
    if args.serve:
        serve()
        return
    job_dir = os.path.abspath(args.job)
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
        set_up_process(signature(json.load(f)["settings"]))
    outcome = run_job(job_dir)
    sys.exit(0 if outcome in ("done", "stopped") else 1)


def peak_ram_bytes():
    """The most memory this process committed at once, which is what it asked of RAM and the page file."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters(cb=ctypes.sizeof(Counters))
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE  # a pseudo handle, -1, which must reach the call as 64 bits
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not info(current(), ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakPagefileUsage


if __name__ == "__main__":
    main()
