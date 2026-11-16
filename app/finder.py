"""Finds what other apps on this PC already have: the folders ComfyUI and the Stable Diffusion web UIs keep their
checkpoints in, and copies of models Nano would otherwise download. It needs no torch, so setup can use it before
anything is installed beyond the standard library."""
import os

# Install folders of other Stable Diffusion apps, and where each keeps its checkpoints.
OTHER_APPS = ("comfyui", "stable-diffusion-webui", "sd-webui", "sd.webui", "webui", "forge", "automatic", "invokeai", "swarmui")
APP_CHECKPOINTS = (("models", "checkpoints"), ("models", "Stable-diffusion"), ("Models", "Stable-Diffusion"))
SKIP_FOLDERS = {"windows", "program files", "program files (x86)", "programdata", "appdata", "node_modules", "$recycle.bin",
                "system volume information", "steam", "steamapps", "recovery"}


def find_other_apps(upscale_dirs=()):
    """Where ComfyUI and the Stable Diffusion web UIs keep their checkpoints: beside the upscaler folders, and in any
    install folder up to three folders deep on each drive. It reads folder names only, skips system folders, and
    stops after 50,000 of them."""
    found = []

    def add(*parts):
        path = os.path.join(*parts)
        if os.path.isdir(path) and path.lower() not in [f.lower() for f in found]:
            found.append(path)

    try:
        for d in upscale_dirs:  # an upscale_models folder usually sits beside the checkpoints
            add(os.path.dirname(os.path.normpath(d)), "checkpoints")
            add(os.path.dirname(os.path.normpath(d)), "Stable-diffusion")
        looked = 0
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            level = [f"{letter}:\\"]
            if not os.path.isdir(level[0]):
                continue
            for _ in range(3):
                deeper = []
                for folder in level:
                    try:
                        entries = list(os.scandir(folder))
                    except OSError:
                        continue
                    for entry in entries:
                        looked += 1
                        name = entry.name.lower()
                        try:
                            if not entry.is_dir() or name.startswith((".", "$")) or name in SKIP_FOLDERS:
                                continue
                        except OSError:
                            continue
                        if name.startswith(OTHER_APPS):
                            for parts in APP_CHECKPOINTS:
                                add(entry.path, *parts)
                        deeper.append(entry.path)
                level = deeper if looked < 50000 else []
    except OSError:
        pass
    return found


def model_roots(checkpoint_dirs=(), upscale_dirs=()):
    """Folders where other apps keep models: the models folder above each checkpoints and upscaler folder, and
    PyTorch's download cache, where torchvision's models land."""
    roots = []
    for folder in [*checkpoint_dirs, *upscale_dirs]:
        parent = os.path.dirname(os.path.normpath(folder))
        if os.path.basename(parent).lower() == "models":
            roots.append(parent)
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints"))
    unique = []
    for root in roots:
        if os.path.isdir(root) and os.path.normcase(root) not in map(os.path.normcase, unique):
            unique.append(root)
    return unique


def files_named(roots, names, depth=4, limit=200000):
    """Every file under the roots, up to `depth` folders down, whose lowercase name is in `names`."""
    found, looked = [], 0
    for root in roots:
        level = [root]
        for _ in range(depth + 1):
            deeper = []
            for folder in level:
                try:
                    entries = list(os.scandir(folder))
                except OSError:
                    continue
                for entry in entries:
                    looked += 1
                    try:
                        if entry.is_dir():
                            if not entry.name.startswith("."):
                                deeper.append(entry.path)
                        elif entry.name.lower() in names:
                            found.append(entry.path)
                    except OSError:
                        continue
            level = deeper if looked < limit else []
    return found
