"""This PC or another: what only a browser on the PC that renders may do, and rendering for other PCs on the
network, with the code they give once."""
import asyncio
import os
import re
import socket
import threading
import time
from urllib.parse import urlparse

import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

import paths
from web.base import APP_DIR, LOCAL, PORT, app


# Off until turned on in Settings. Then Nano also listens on this PC's network addresses, and a browser
# on another PC opens http://<address>:<port>. It enters the code shown here once, and a cookie remembers it.
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


def share_at_start():
    """Shares this PC again when Nano starts, if sharing was left on."""
    if LOCAL.get("share_on_network"):
        start_sharing()
        for address in _network["addresses"]:
            print(f"Other PCs on the network can render here at http://{address}:{PORT} with the code in Settings.", flush=True)


def stop_sharing():
    for server in _network["servers"]:
        server.should_exit = True
    _network["servers"], _network["addresses"] = [], []


def network_json():
    on = bool(LOCAL.get("share_on_network"))
    return {"on": on, "addresses": _network["addresses"] if on else [], "port": PORT, "code": share_code() if on else None,
            "name": socket.gethostname(), "error": _network["error"] if on else None}


@app.middleware("http")
async def only_this_app(request: Request, call_next):
    """Stops other websites in the same browser from driving this server, and asks a browser on another PC for
    the code before it can do anything."""
    host = request.headers.get("host", "").rsplit(":", 1)[0].lower()
    origin = request.headers.get("origin")
    if not is_local(request):
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
async def connect(body: dict, request: Request):
    """A browser on another PC gives the code once, and keeps it in a cookie."""
    who = request.client.host if request.client else "?"
    if _wrong.get(who, 0) >= 20:
        raise HTTPException(429, "Too many wrong codes. Restart Nano on the PC that renders to try again.")
    code = re.sub(r"[^A-Z0-9]", "", str(body.get("code", "")).upper())
    if code != share_code():
        _wrong[who] = _wrong.get(who, 0) + 1
        await asyncio.sleep(0.5)  # slows guessing without holding up the server
        raise HTTPException(403, "That code doesn't match. It's in Settings, under Share this PC, on the PC that renders.")
    _wrong.pop(who, None)
    response = JSONResponse({"ok": True})
    response.set_cookie(KEY_COOKIE, code, max_age=60 * 60 * 24 * 365, httponly=True, samesite="strict")
    return response


@app.get("/api/network")
def network():
    return network_json()


@app.post("/api/network")
def set_network(body: dict, request: Request):
    """Turns sharing on or off, or makes a new code, which signs out every other PC."""
    local_only(request)
    if body.get("new_code"):
        LOCAL.pop("share_code", None)
        paths.save_local_settings({"share_code": None})
        share_code()
    if "on" in body:
        on = bool(body["on"])
        LOCAL["share_on_network"] = on
        paths.save_local_settings({"share_on_network": on})
        (start_sharing if on else stop_sharing)()
    return network_json()


def local_only(request: Request):
    """A Windows dialog opens on the PC that runs Nano, so only a browser on that PC may open one."""
    if not is_local(request):
        raise HTTPException(403, "Choose this on the PC that renders. From here, drop files onto the page instead.")


def no_network_paths(request: Request, *chosen):
    """A browser on another PC can choose files and folders on this one, but not network paths, a shared folder on
    another PC: opening one makes Windows connect to that PC and offer this PC's login. `chosen` can be whole request
    bodies: every string in them is checked, so a path in a setting is never missed. Call it before anything opens
    one, even to see that it's there."""
    if not is_local(request) and any(s.strip().replace("/", "\\").startswith("\\\\") for s in strings_in(chosen)):
        raise HTTPException(403, "Choose network files and folders on the PC that renders.")


def strings_in(value):
    """Every string in a JSON value, however deep."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from strings_in(v)


def is_local(request):
    return bool(request.client) and request.client.host in ("127.0.0.1", "::1")
