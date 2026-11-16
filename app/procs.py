"""Nano's own processes: renders, upscales and fixes each run in a Python process of their own, so their memory is
all given back when they end, and a crash can't take the app down.

Both ends of how they talk are here. A process reports on stdout as @@EVT lines, each one JSON event, and the server
ends a long-lived one by closing its stdin.

The process's side: emit() and stdin_closed().
The server's side: Child, which starts a process and follows what it reports, and run_once() for a short one.
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback

import paths

EVENT = "@@EVT "


# ---- the process's side ----
def emit(kind, **data):
    """Reports an event to the server."""
    data["type"] = kind
    data["t"] = round(time.time(), 2)
    print("\n" + EVENT + json.dumps(data), flush=True)  # leading newline: a library may have left a part line


def failure(e):
    """How a process that failed with `e` ended, for its finished event: "oom" when the GPU ran out of memory, otherwise
    "error", with the error to show."""
    oom = type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()
    return ("oom" if oom else "error"), {"error": repr(e)[:600]}


def stdin_closed():
    """Whether the server closed this process's stdin, or ended, seen without reading it: on Windows, a read waiting
    on a pipe holds up everything else that touches it, such as starting a process, which loading a model does.
    A process started from a console, not by the server, has no pipe to close, and never sees it closed."""
    if os.name == "nt":
        import ctypes
        import msvcrt
        kernel = ctypes.windll.kernel32
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        if kernel.GetFileType(handle) != 3:  # FILE_TYPE_PIPE
            return False
        waiting = ctypes.c_ulong()
        return not kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(waiting), None)
    import select
    import stat
    if not stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode):
        return False
    return bool(select.select([sys.stdin], [], [], 0)[0]) and not sys.stdin.read(1)


# ---- the server's side ----
def event_in(line):
    """The event an output line carries, or None. It may follow other output on the same line, such as a progress bar
    a library left unfinished."""
    at = line.find(EVENT)
    if at < 0:
        return None
    try:
        event = json.loads(line[at + len(EVENT):])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def environment(**extra):
    return dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", DISCO_NANO_HOME=paths.HOME, **extra)


# Below normal priority: a render still gets every idle cycle, but Windows serves the desktop first, so the PC stays
# usable while it runs. No window: the app's console is the only one.
FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0


def command(script, *args):
    return [paths.PYTHON, "-B", os.path.join(paths.APP_DIR, script), *map(str, args)]


class Child:
    """A process of Nano's, with its output followed on a thread: each line goes to `log_path`, when there is one,
    and to `on_line(line, event)`, with the event it carries or None. `on_exit(code)` is called once it has ended.
    With `stdin`, the server can write to it and end it by closing it."""

    def __init__(self, script, *args, on_line, on_exit=None, log_path=None, stdin=False, env=None):
        self.on_line, self.on_exit, self.log_path = on_line, on_exit, log_path
        self.proc = subprocess.Popen(command(script, *args), stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", cwd=paths.HOME, env=environment(**(env or {})), creationflags=FLAGS)
        threading.Thread(target=self._read, daemon=True).start()

    def alive(self):
        return self.proc.poll() is None

    def write(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def close(self, grace=10.0):
        """Closes its stdin, which it ends on at its next check. One that hasn't ended `grace` seconds later, such as
        one still loading a model, is ended outright. A venv's launcher takes the Python it started with it."""
        try:
            self.proc.stdin.close()
        except (OSError, AttributeError):
            pass
        if grace <= 0:
            self.terminate()
            return
        timer = threading.Timer(grace, self.terminate)
        timer.daemon = True  # the app can close without waiting for it
        timer.start()

    def terminate(self):
        if self.alive():
            self.proc.terminate()

    def _read(self):
        log = open(self.log_path, "a", encoding="utf-8", errors="replace") if self.log_path else None
        try:
            for raw in self.proc.stdout:
                line = raw.rstrip("\r\n")
                if log:
                    log.write(line + "\n")
                    log.flush()
                try:
                    self.on_line(line, event_in(line))
                except Exception:  # one bad line mustn't stop the reading, or the process would stall on a full pipe
                    traceback.print_exc()
        finally:
            if log:
                log.close()
        code = self.proc.wait()
        if self.on_exit:
            self.on_exit(code)


def run_once(script, *args, timeout):
    """Runs a short process to its end and returns it, with its output. Raises subprocess.TimeoutExpired."""
    return subprocess.run(command(script, *args), capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=paths.HOME, env=environment(), timeout=timeout, creationflags=FLAGS)


def last_line(run):
    """The last line a finished process printed, which is where a failure says why."""
    lines = [line for line in (run.stderr or run.stdout or "").splitlines() if line.strip()]
    return lines[-1][-300:] if lines else ""
