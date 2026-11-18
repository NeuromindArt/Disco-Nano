"""Readings of the PC: GPU and RAM in use, before and during a job, the images' drive, and other GPU apps that
are open."""
import os
import shutil
import socket
import subprocess
import threading
import time

import paths
from web.base import LOCAL
from web import places


OTHER_GPU_PORTS = LOCAL.get("other_gpu_ports", [8188, 7860])  # ComfyUI and Stable Diffusion web UIs
_gpu_seen = {"at": 0.0, "info": None}


def forget_gpu_reading():
    """A job just ended or let go of the GPU, and the last reading still counts its memory: the next one is read anew."""
    _gpu_seen["at"] = 0.0


def gpu_info(since=0.0):
    """Calls nvidia-smi at most once every 5 seconds. Each call asks the driver for a report,
    and polling once a second during a render makes the screen hitch more often. A reading taken
    before `since` is read again. A failed reading is retried at the next call."""
    if _gpu_seen["info"] and time.time() - _gpu_seen["at"] < 5 and _gpu_seen["at"] >= since:
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


def idle_memory(task):
    """The GPU and RAM in use apart from `task`, the last job: read now while it isn't running, and kept from before it
    started while it is."""
    if not (task and task.running()):
        # Windows takes a moment to hand back a finished worker's memory, so only later readings count.
        settled = (task.finished_at or time.time()) + 1.5 if task else 0.0
        gpu = gpu_info(since=settled)
        if gpu and _gpu_seen["at"] >= settled:
            _idle["gpu"] = gpu
        _idle["ram"] = ram_info() or _idle["ram"]
    return _idle["gpu"], _idle["ram"]


def take_idle_reading():
    """A fresh reading just before a job starts, so what it adds is measured from what was really in use."""
    gpu = read_gpu()
    if gpu:
        _gpu_seen["info"], _gpu_seen["at"] = gpu, time.time()
        _idle["gpu"] = gpu
    _idle["ram"] = ram_info() or _idle["ram"]


def process_ram_gb(pid):
    """The RAM a process has in use, from Windows, or 0 when it can't be read."""
    if os.name != "nt":
        return 0.0
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                 "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                 "PagefileUsage", "PeakPagefileUsage")]
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)  # query limited information, read memory
    if not handle:
        return 0.0
    try:
        counters = Counters(cb=ctypes.sizeof(Counters))
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0.0
        return counters.WorkingSetSize / 2**30
    finally:
        kernel.CloseHandle(handle)


def disk_info():
    """The drive that holds the images, which is what the status bar shows, and how much of it is Nano's."""
    try:
        usage = shutil.disk_usage(os.path.splitdrive(places.OUT_DIR)[0] + os.sep)
    except OSError:
        return None  # the drive was unplugged while the app was open
    measure_disco()
    return {"used_gb": round(usage.used / 2**30, 1), "total_gb": round(usage.total / 2**30, 1),
            "free_gb": round(usage.free / 2**30, 1), "folder": places.OUT_DIR, "disco_gb": _disco_size["gb"]}


_disco_size = {"at": 0.0, "gb": None, "busy": False}


def measure_again():
    """The images moved to another folder, maybe on another drive: the next status measures it anew."""
    _disco_size["at"] = 0


def folder_bytes(top):
    total = 0
    for folder, _, names in os.walk(top):
        for n in names:
            try:
                total += os.lstat(os.path.join(folder, n)).st_size
            except OSError:
                pass
    return total


def measure_disco():
    """Nano's files on the images' drive: its home folder (environment, models, inputs) and the images.
    Walking every file takes a few seconds, so it runs on a thread at most every five minutes."""
    if _disco_size["busy"] or time.time() - _disco_size["at"] < 300:
        return

    def walk():
        try:
            drive = os.path.splitdrive(os.path.realpath(places.OUT_DIR))[0].lower()
            tops = [paths.HOME] + ([] if places.inside(places.OUT_DIR, paths.HOME) else [places.OUT_DIR])
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
