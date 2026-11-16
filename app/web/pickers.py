"""Folders and files chosen on the PC that renders: the folders in Settings, Windows' own pickers, the page's own
browser for another PC, and opening the images folder."""
import os
import socket
import subprocess

from fastapi import HTTPException, Request

import paths
from web.base import APP_DIR, app, lock
from web import jobs, network, places, system


@app.get("/api/folders")
def folders():
    return {"output_dir": places.OUT_DIR, "upscale_dirs": places.UPSCALE_DIRS, "checkpoint_dirs": places.CHECKPOINT_DIRS}


@app.post("/api/folders")
def set_folders(body: dict, request: Request):
    """Model folders can be set from a browser on another PC too, since they only say where to read. Where
    images are saved is set on the PC that renders."""
    with lock:
        if jobs.current and jobs.current.running():
            raise HTTPException(409, "Wait for the current job to finish.")
        wanted = str(body.get("output_dir", places.OUT_DIR)).strip() or paths.OUTPUT
        if not os.path.isabs(wanted):
            raise HTTPException(400, "Use a full folder path, such as D:\\Art\\Disco.")
        if not network.is_local(request) and os.path.normcase(os.path.abspath(wanted)) != os.path.normcase(places.OUT_DIR):
            raise HTTPException(403, "Choose where images are saved on the PC that renders.")
        output_dir = places.writable_folder(wanted)
        dirs = [d.strip() for d in body.get("upscale_dirs", places.UPSCALE_DIRS) if isinstance(d, str) and d.strip()]
        checkpoint_dirs = [d.strip() for d in body.get("checkpoint_dirs", places.CHECKPOINT_DIRS) if isinstance(d, str) and d.strip()]
        network.no_network_paths(request, [d for d in dirs + checkpoint_dirs if d not in places.UPSCALE_DIRS + places.CHECKPOINT_DIRS])
        for d in dirs:
            if d not in places.UPSCALE_DIRS and not os.path.isdir(d):  # a known drive may just be unplugged
                raise HTTPException(400, f"Folder not found: {d}")
        for d in checkpoint_dirs:
            if d not in places.CHECKPOINT_DIRS and not os.path.isdir(d):
                raise HTTPException(400, f"Folder not found: {d}")
        if output_dir != places.OUT_DIR:
            jobs.forget_task()  # its images were in the old folder
            system.measure_again()  # the new folder's drive
        places.use_folders(output_dir, dirs, checkpoint_dirs)
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


@app.post("/api/pick-folder")
def pick_folder(body: dict, request: Request):
    """Opens Windows' folder picker. A browser page can't read a real path from its own."""
    network.local_only(request)
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


HIDDEN_FOLDERS = {"$recycle.bin", "system volume information", "recovery", "config.msi"}
# The files each kind of choice shows. A folder choice shows its models too, so you can tell the right folder.
BROWSE_KINDS = {"checkpoint": (".safetensors", ".gguf"), "checkpoint-folder": (".safetensors", ".gguf"),
                "upscaler": (".pth", ".safetensors"), "folder": ()}


@app.get("/api/browse")
def browse(request: Request, path: str = "", kind: str = "folder"):
    """Folders, and the model files a kind wants, on the PC that renders. A browser on another PC can't open
    Windows' file dialogs here, so the page draws its own from this."""
    network.no_network_paths(request, path)
    wanted = BROWSE_KINDS.get(kind, ())
    machine = socket.gethostname()
    if not path:
        drives = [f"{letter}:\\" for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.isdir(f"{letter}:\\")]
        return {"path": "", "parent": None, "folders": [{"name": d, "path": d} for d in drives], "files": [], "machine": machine}
    path = os.path.abspath(path)
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
    except OSError:
        raise HTTPException(400, f"Can't open {path}.") from None
    folders, files = [], []
    for entry in entries:
        try:
            if entry.is_dir():
                if not entry.name.startswith((".", "$")) and entry.name.lower() not in HIDDEN_FOLDERS:
                    folders.append({"name": entry.name, "path": entry.path})
            elif wanted and entry.name.lower().endswith(wanted):
                files.append({"name": entry.name, "path": entry.path, "gb": round(entry.stat().st_size / 2**30, 1)})
        except OSError:
            continue  # a folder Windows won't let us look into
    parent = os.path.dirname(path)
    return {"path": path, "parent": "" if parent == path else parent, "folders": folders, "files": files, "machine": machine}


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
    network.local_only(request)
    env = dict(os.environ, OPEN_PATH=places.OUT_DIR)
    subprocess.Popen(["powershell", "-NoProfile", "-STA", "-Command", OPEN_FOLDER], env=env,
                     creationflags=subprocess.CREATE_NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"ok": True}
