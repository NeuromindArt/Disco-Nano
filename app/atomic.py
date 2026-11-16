"""Files written whole: each is written beside its place and then moved into it, so a process reading it never sees
one half written. The server and the processes it starts talk through files like this, such as a render's
regions.json and preview.jpg.

On Windows a file that is open, such as a preview being sent to the page, can't be replaced, so the move is tried a
few times before it gives up.
"""
import itertools
import json
import os
import time

TRIES, WAIT = 5, 0.05


def replace(temp, path, tries=TRIES):
    """Moves `temp` into `path`. Returns False when `path` stayed open the whole time; `temp` is then removed."""
    for k in range(tries):
        try:
            os.replace(temp, path)
            return True
        except PermissionError:
            if k == tries - 1:
                break
            time.sleep(WAIT)
    try:
        os.remove(temp)
    except OSError:
        pass
    return False


_written = itertools.count()


def temp_of(path):
    """Where `path` is written before it moves into place: a name of its own each time, so two writers, in one
    process or two, never share one. It keeps the extension, which says what kind of image to save."""
    stem, ext = os.path.splitext(path)
    return f"{stem}.{os.getpid()}-{next(_written)}.tmp{ext}"


def write_json(path, data, indent=None):
    temp = temp_of(path)
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    if not replace(temp, path):
        raise PermissionError(f"{path} is open elsewhere and couldn't be replaced.")


def save_image(image, path, tries=TRIES, **options):
    """Saves a PIL image. Returns False when `path` stayed open, which a preview can skip: the next one gets through."""
    temp = temp_of(path)
    image.save(temp, **options)
    return replace(temp, path, tries)
