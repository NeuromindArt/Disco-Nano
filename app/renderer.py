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

import atomic
import paths
from procs import emit, failure, stdin_closed


class Hooks:
    """How disco.render reports back."""

    def __init__(self, job_dir, progress=False, regions=None):
        self.job_dir = job_dir
        self.stop_file = os.path.join(job_dir, "stop")
        self.pause_file = os.path.join(job_dir, "pause")
        # Regional prompts: boxes with prompts of their own. A pause can change them, which counts up the revision.
        self.regions, self.regions_rev, self.regions_read = regions, 0, None
        self.started = False
        self.while_paused = None  # called over and over while a pause holds, for Fix's inpaint tries
        # With Save progress video, every preview is also kept as a frame until its image is saved.
        self.progress = os.path.join(job_dir, "progress") if progress else None
        self.frames = []

    def status(self, message):
        emit("status", message=message)

    def stopped(self):
        return os.path.exists(self.stop_file)

    def tried(self, **data):
        """An inpaint try, painted during a pause, for the page to show."""
        emit("tried", **data)

    def step(self, j, total, batch, n_batches, seed, can_hold=True, at=None):
        """Step j, counted from 0, of `total`. The page shows the steps done, from 1 to `total`. Tiles report each
        tile's step but hold a pause only between whole steps, where the preview and an erase are. `at` is how far
        through the whole schedule the step is, from 0 to 1, which a start image or Tiles makes differ from j/total."""
        if not self.started:
            self.started = True
            emit("sampling", total_steps=total, n_batches=n_batches, seed=seed)
        emit("step", step=j + 1, total=total, batch=batch, n_batches=n_batches, **({} if at is None else {"at": round(at, 3)}))
        if os.path.exists(self.stop_file):
            raise KeyboardInterrupt
        if can_hold and os.path.exists(self.pause_file):
            self.hold()
        self.read_regions()

    def pausing(self):
        return os.path.exists(self.pause_file)

    def hold(self):
        """Waits while the pause file is there, so regions can be drawn on the preview. A server that closes meanwhile
        ends the render, so a paused one never holds the GPU with no app left to carry it on."""
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            # VRAM the render keeps for its next step goes back while it waits: with the card nearly full, the
            # browser's own drawing on the GPU stalls, and the brush lags. An erase's extra steps add to what is kept.
            torch.cuda.empty_cache()
        emit("paused")
        while os.path.exists(self.pause_file):
            if os.path.exists(self.stop_file) or stdin_closed():
                raise KeyboardInterrupt
            if self.while_paused:
                self.while_paused()
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
        # JPEG, because a preview is written every step: a large PNG takes several times longer to write. One the server
        # is still sending can't be replaced; this one is then skipped, and the next gets through.
        if atomic.save_image(image, os.path.join(self.job_dir, "preview.jpg"), tries=3, quality=90):
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
        # DISCO_NANO_THREADS overrides the setting, for scripts that time renders at several thread counts
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
        outcome, detail = failure(e)
        print(traceback.format_exc()[-1800:], flush=True)
    # disco.render puts its models away in a finally, but after a Stop that runs while the stop is still on its way
    # out, holding the render's last tensors. They are let go only now, so the GPU is given back here as well, or
    # each stopped render would leave the next one with less room.
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
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
