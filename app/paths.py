"""Where Nano keeps its environment, models and images.

Everything large lives in one home folder. It defaults to `data` beside this project. To keep it
somewhere else, set DISCO_NANO_HOME or put the folder's path in a file named home.txt in the
project folder.

Optional machine settings go in local.json inside the home folder. The app also saves some of its
own choices there, such as output_dir and upscale_dirs. For example:
    {"output_dir": "D:/Pictures/Disco",
     "upscale_dirs": ["D:/Models/upscale_models"],
     "vit_l14_weights": ["D:/Models/clip/ViT-L-14.safetensors"],
     "other_gpu_ports": [8188, 7860]}
"""
import json
import os

VERSION = "1.1.0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)


def find_home():
    home = os.environ.get("DISCO_NANO_HOME")
    marker = os.path.join(PROJECT_DIR, "home.txt")
    if not home and os.path.isfile(marker):
        with open(marker, encoding="utf-8") as f:
            home = f.read().strip()
    return os.path.abspath(home or os.path.join(PROJECT_DIR, "data"))


HOME = find_home()
PYTHON = os.path.join(HOME, ".venv", "Scripts", "python.exe")
MODELS = os.path.join(HOME, "models")
OUTPUT = os.path.join(HOME, "images_out")
INIT_IMAGES = os.path.join(HOME, "init_images")
JOBS = os.path.join(HOME, "jobs")
# setup.bat downloads each into the home folder, at the commit Nano was tested with.
CODE_REPOS = {
    "guided-diffusion": ("kostarion/guided-diffusion", "ced66cb5b4dc58580701cc1519f5efd3b0436b27"),
    "CLIP": ("openai/CLIP", "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"),
    "ResizeRight": ("assafshocher/ResizeRight", "510d4d5b67dccf4efdee9f311ed42609a71f17c5"),
}


def local_settings():
    path = os.path.join(HOME, "local.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_local_settings(changes):
    settings = {**local_settings(), **changes}
    path = os.path.join(HOME, "local.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    os.replace(path + ".tmp", path)
