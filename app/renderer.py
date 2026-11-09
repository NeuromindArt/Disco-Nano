"""Runs one render in its own process and reports progress as @@EVT lines on stdout.

Usage: python renderer.py --job <job_dir>      (job_dir/job.json holds the settings)
A file named `stop` in the job folder ends the run at the next step.
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

    def __init__(self, job_dir):
        self.job_dir = job_dir
        self.stop_file = os.path.join(job_dir, "stop")
        self.started = False

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

    def preview(self, image):
        temp = os.path.join(self.job_dir, "preview.tmp.png")
        image.save(temp)
        for _ in range(3):
            try:
                os.replace(temp, os.path.join(self.job_dir, "preview.png"))
                break
            except PermissionError:  # the server is sending the last preview, and Windows won't replace an open file
                time.sleep(0.05)
        else:
            return  # this preview is skipped, and the next one will get through
        emit("preview")

    def image(self, path):
        emit("image", path=path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    job_dir = os.path.abspath(parser.parse_args().job)
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    options = job.get("options", {})
    exact_repeat = bool(job["settings"].get("exact_repeat"))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("TORCH_HOME", os.path.join(paths.HOME, "cache", "torch"))
    if exact_repeat:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # must be set before torch starts
    width, height = job["settings"].get("width_height", (0, 0))
    if width * height >= 500_000:
        # At 1280x768 a 9MB batch of CLIP cuts could land inside a freed 720MB block and pin all of it through
        # the backward pass, leaving no room for the next 720MB block. This stops big free blocks being split
        # for small tensors. It costs about 5% speed, so smaller renders, which have room to spare, skip it.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    for name in paths.CODE_REPOS:
        sys.path.insert(0, os.path.join(paths.HOME, name))

    hooks = Hooks(job_dir)
    outcome, detail = "done", {}
    try:
        import torch
        threads = int(os.environ.get("DISCO_NANO_THREADS") or job["settings"].get("cpu_threads", 4))
        if threads > 0:  # more threads mostly spin while they wait for the GPU, which slows the rest of the PC
            torch.set_num_threads(threads)
        if exact_repeat:
            torch.use_deterministic_algorithms(True)
        import memory
        cap = memory.limit_vram()
        hooks.status(f"Loading models ({cap:.1f}GB VRAM available)")
        import disco
        memory.half_precision_norm()
        memory.lean_unet()
        # The secondary model skips the big UNet's backward pass, which is all Offload helps with. Offload would
        # still move CLIP off and back every step, and at 1280x768 that left 2GB of VRAM in unusable pieces.
        offload = bool(options.get("offload")) and not job["settings"].get("use_secondary_model")
        if offload:
            memory.offload_activations()
        disco.render(job["settings"], job["prompts"], hooks, offload, options.get("init_fit", "crop"),
                     options.get("output_dir") or paths.OUTPUT)
    except KeyboardInterrupt:
        outcome = "stopped"
    except BaseException as e:  # out of memory included; every failure must reach the UI as an event
        is_oom = type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()
        outcome = "oom" if is_oom else "error"
        detail = {"error": repr(e)[:600]}
        print(traceback.format_exc()[-1800:], flush=True)
    detail["cpu_seconds"] = round(time.process_time(), 1)
    try:
        detail["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        detail["peak_tensors_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)  # reserved can fill to the cap
    except Exception:
        pass
    peak_ram = peak_ram_bytes()
    if peak_ram:
        detail["peak_ram_gb"] = round(peak_ram / 2**30, 2)
    emit("finished", outcome=outcome, **detail)
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
