"""Downloads the code Nano builds on, guided-diffusion, CLIP and ResizeRight, during setup.

Each comes from GitHub as a zip at the commit Nano was tested with, so setup needs no Git. A folder
that is already there, from an earlier setup or a Git clone, is kept as it is.
"""
import os
import shutil
import sys
import urllib.request
import zipfile

import paths


def fetch(name, repo, commit):
    target = os.path.join(paths.HOME, name)
    if os.path.isdir(target) and os.listdir(target):
        return
    print(f"Downloading {name}", flush=True)
    url = f"https://codeload.github.com/{repo}/zip/{commit}"
    work = os.path.join(paths.HOME, "cache", "code-download")
    shutil.rmtree(work, ignore_errors=True)  # left over if an earlier setup was interrupted
    os.makedirs(work)
    try:
        archive = os.path.join(work, "code.zip")
        with urllib.request.urlopen(url, timeout=60) as response, open(archive, "wb") as f:
            shutil.copyfileobj(response, f)
        with zipfile.ZipFile(archive) as z:
            z.extractall(work)
        inner = [d for d in os.listdir(work) if os.path.isdir(os.path.join(work, d))]
        if len(inner) != 1:
            raise RuntimeError(f"Unexpected contents in the {name} download")
        if os.path.isdir(target):
            os.rmdir(target)
        shutil.move(os.path.join(work, inner[0]), target)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    os.makedirs(paths.HOME, exist_ok=True)
    try:
        for name, (repo, commit) in paths.CODE_REPOS.items():
            fetch(name, repo, commit)
    except Exception as error:
        print(f"Could not download the code: {error}", flush=True)
        sys.exit(1)
