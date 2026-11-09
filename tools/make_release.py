"""Builds the zip for a GitHub release: dist/Disco-Diffusion-Nano-<version>.zip.

    python tools/make_release.py [ref]

It packs the files committed at ref, HEAD by default, inside a "Disco Diffusion Nano" folder.
.gitattributes leaves out the README pictures, this folder and the GitHub files, and gives the
.bat files Windows line endings. Commit first: changes that aren't committed are not included.
The version comes from VERSION in app/paths.py.
"""
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    with open(os.path.join(ROOT, "app", "paths.py"), encoding="utf-8") as f:
        version = re.search(r'^VERSION = "([^"]+)"', f.read(), re.M).group(1)
    if ref == "HEAD" and git("status", "--porcelain", "--untracked-files=no").strip():
        print("Note: there are uncommitted changes, and the zip leaves them out.")
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", f"Disco-Diffusion-Nano-{version}.zip")
    git("archive", "--format=zip", "--prefix=Disco Diffusion Nano/", "-o", out, ref)

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        bat = z.read("Disco Diffusion Nano/setup.bat")
    problems = [p for p in ("setup.bat", "Start Disco Nano.bat", "app/server.py", "LICENSE")
                if f"Disco Diffusion Nano/{p}" not in names]
    if b"\n" in bat.replace(b"\r\n", b""):
        problems.append("setup.bat has Unix line endings")
    if any("/docs/" in n or "/tools/" in n for n in names):
        problems.append("docs or tools were packed")
    if problems:
        os.remove(out)
        sys.exit("Release zip not made: " + "; ".join(problems))
    size = os.path.getsize(out) / 2**20
    print(f"{out}\n{len(names)} entries, {size:.1f}MB, version {version} from {ref}")


if __name__ == "__main__":
    main()
