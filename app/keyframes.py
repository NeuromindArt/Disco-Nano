"""Keyframes and prompt sets for animation. Kept apart from torch, so the web server can check them."""
import ast
import math
import re

MOTION = ["angle", "zoom", "translation_x", "translation_y", "translation_z", "rotation_3d_x", "rotation_3d_y", "rotation_3d_z"]


# Keyframes ---------------------------------------------------------------------------------------

FUNCTIONS = {name: getattr(math, name) for name in ("sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "exp", "log",
                                                     "floor", "ceil", "fabs")}
FUNCTIONS.update(abs=abs, min=min, max=max, round=round)
NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}
ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Add, ast.Sub,
           ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd)


def formula(text):
    """Turns one keyframe value, such as 1.02 or 1+0.01*sin(t/10), into a function of the frame t."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        raise ValueError(f"Could not read {text!r} as a number or formula.") from None
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ValueError(f"{text!r} can only use numbers, t, + - * / ** %, and functions such as sin.")
        if isinstance(node, ast.Name) and node.id != "t" and node.id not in NAMES and node.id not in FUNCTIONS:
            raise ValueError(f"{text!r} uses {node.id!r}. Formulas can use t, pi and functions such as sin.")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS):
            raise ValueError(f"{text!r} calls a function formulas can't use.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"{text!r} holds something that isn't a number.")
    code = compile(tree, "<keyframe>", "eval")
    return lambda t: float(eval(code, {"__builtins__": {}}, {**NAMES, **FUNCTIONS, "t": t}))  # checked above


def split_top(text):
    """Splits at commas that aren't inside brackets, so sin(a, b) stays whole."""
    parts, depth, current = [], 0, ""
    for char in text:
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return parts + [current]


def keyframes(text):
    """Reads Disco's keyframe form, 0:(1.0), 60:(1.02), into a sorted list of (frame, formula)."""
    points = {}
    for part in split_top(str(text)):
        if not part.strip():
            continue
        frame, colon, value = part.partition(":")
        if not colon or not frame.strip().isdigit():
            raise ValueError(f"Could not read {part.strip()!r}. Write keyframes as 0:(1.0), 60:(1.02).")
        value = value.strip()
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1]
        points[int(frame)] = formula(value)
    if not points:
        raise ValueError(f"{text!r} has no keyframes. Write them as 0:(1.0), 60:(1.02).")
    return sorted(points.items())


def series(text, frames, shape="Linear"):
    """The value at every frame. Formulas are worked out at each frame and then blended between keyframes."""
    points = keyframes(text)
    values = []
    for t in range(frames):
        after = next((i for i, (at, _) in enumerate(points) if at > t), len(points))
        if after == 0:
            values.append(points[0][1](t))
            continue
        if after == len(points):
            values.append(points[-1][1](t))
            continue
        (a, fa), (b, fb) = points[after - 1], points[after]
        s = (t - a) / (b - a)
        if shape == "Cubic":
            # Catmull-Rom through the neighbors, so curves pass through every key without corners
            before = points[after - 2][1](t) if after >= 2 else fa(t)
            beyond = points[after + 1][1](t) if after + 1 < len(points) else fb(t)
            p0, p1, p2, p3 = before, fa(t), fb(t), beyond
            values.append(0.5 * (2 * p1 + (p2 - p0) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s ** 2
                                 + (3 * p1 - p0 - 3 * p2 + p3) * s ** 3))
        else:
            values.append(fa(t) + (fb(t) - fa(t)) * s)
    return values


def motion(cfg, frames):
    """Every camera value for every frame, as {name: [values]}."""
    out = {}
    for name in MOTION:
        try:
            out[name] = series(cfg[name], frames, cfg["interp_spline"])
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from None
    return out


def skips(cfg, frames):
    """Frame skip % for every frame: its keyframes if it has them, or the one steady value."""
    if not str(cfg.get("frame_skip_keys", "")).strip():
        return [float(cfg["frames_skip_steps"])] * frames
    try:
        return [min(max(v, 0.0), 99.0) for v in series(cfg["frame_skip_keys"], frames, cfg["interp_spline"])]
    except ValueError as e:
        raise ValueError(f"frame_skip_keys: {e}") from None


# Prompts over time -------------------------------------------------------------------------------

PROMPT_START = re.compile(r"^\s*(\d+)\s*:\s*(.*)$")


def prompt_sets(prompts):
    """Splits prompt lines into sets by frame. A line starting 120: begins the set used from frame 120."""
    sets, at = {}, 0
    for line in prompts:
        match = PROMPT_START.match(line)
        if match:
            at, line = int(match.group(1)), match.group(2)
        if line.strip():
            sets.setdefault(at, []).append(line)
    if 0 not in sets:
        raise ValueError("The first prompt must start at frame 0.")
    return sorted(sets.items())
