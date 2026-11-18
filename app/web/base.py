"""What every part of the server shares: the FastAPI app they add their routes to, the lock that guards what
they share, this PC's own settings and the address Nano listens on."""
import os
import threading

from fastapi import FastAPI

import paths


APP_DIR = paths.APP_DIR
PYTHON = paths.PYTHON
LOCAL = paths.local_settings()  # this PC's own settings: its folders, sharing, and what the welcome screen did

HOST, PORT = "127.0.0.1", int(os.environ.get("DISCO_NANO_PORT") or 7861)  # another port runs a second copy
URL = f"http://{HOST}:{PORT}"

app = FastAPI(title="Disco Diffusion Nano")
lock = threading.Lock()  # guards the running job, the downloads and the folders, which requests share across threads
