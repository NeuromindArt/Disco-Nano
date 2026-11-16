"""Local web app for Disco Diffusion Nano. Binds to 127.0.0.1 only, and to this PC's network addresses too while
sharing is on.

Never imports torch, so the idle server holds no VRAM. Renders and upscales each run in their own process, so their
memory is fully released when they end and an out-of-memory failure cannot take the app down.

Each part of the server is a module of the web package, which adds its own routes to the app: render, fix, gallery,
inputs, pickers, checkpoints, downloads and network. This file starts them.
"""
import atexit
import os
import sys
import threading
import time
import webbrowser

print("Loading the web server", flush=True)
import uvicorn
from fastapi.staticfiles import StaticFiles

print("Checking folders", flush=True)
from web.base import APP_DIR, HOST, PORT, PYTHON, URL, app
from web import checkpoints, downloads, fix, gallery, inputs, jobs, network, pickers, render, system  # noqa: F401  (each adds its routes)

app.mount("/", StaticFiles(directory=os.path.join(APP_DIR, "static"), html=True), name="ui")  # after every route


def open_when_ready():
    while not system.port_open(PORT):
        time.sleep(0.1)
    print(f"Ready at {URL}. Close this window to shut the app down.", flush=True)
    network.share_at_start()
    if not os.environ.get("NANO_NO_BROWSER"):
        webbrowser.open(URL)


if __name__ == "__main__":
    if not os.path.exists(PYTHON):
        sys.exit(f"Environment not found at {PYTHON}")
    if system.port_open(PORT):
        print(f"Disco Diffusion Nano is already running at {URL}", flush=True)
        webbrowser.open(URL)
        sys.exit(0)
    print("Starting the server", flush=True)
    checkpoints.update_from_searching()
    threading.Thread(target=open_when_ready, daemon=True).start()
    atexit.register(jobs.close_worker)  # the worker also ends by itself when this process's end closes its stdin
    atexit.register(jobs.close_tries)  # and so does the start image's tries process
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
