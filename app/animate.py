"""Animation: each frame starts from the last one, moved by the camera, and Disco repaints part of it. film() paints
a render's clip.

Written for Nano. The setting names follow Disco Diffusion's, so its guides still apply, but the code
is new: the 3D camera reads depth with Depth Anything V2 Small instead of MiDaS and AdaBins.
"""
import gc
import math
import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

import keyframes
import tiles
from imaging import size_start_image
from keyframes import motion
from loaders import device, load_lpips

TRANSLATION_SCALE = 1 / 200  # Disco's 3D units, so its keyframe values move about as far here
NEAR, FAR = 2.0, 20.0  # the depth range a scene is spread over, in the same units


# Prompts over time ----------------------------------------------------------------------------------

def mix(before, after, share):
    """Prompt stats that count `before` by 1 - share and `after` by share, for a fade between sets."""
    out = []
    for a, b in zip(before, after):
        out.append({"model": a["model"], "embeds": torch.cat([a["embeds"], b["embeds"]]),
                    "weights": torch.cat([a["weights"] * (1 - share), b["weights"] * share])})
        if "roles" in a and "roles" in b:  # Prompt routing's kind for each line, so it keeps routing during a fade
            out[-1]["roles"] = torch.cat([a["roles"], b["roles"]])
    return out


def stats_at(all_stats, frame, fade):
    """The prompt stats for one frame. Each set fades in over `fade` frames from the set before it."""
    current = max(i for i, (at, _) in enumerate(all_stats) if at <= frame)
    at, stats = all_stats[current]
    if current and fade and frame - at < fade:
        return mix(all_stats[current - 1][1], stats, (frame - at + 1) / (fade + 1))
    return stats


# Camera ------------------------------------------------------------------------------------------

def pixel_grid(height, width, device):
    """Pixel centers, measured from the middle of the frame, in pixels."""
    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5 - height / 2
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5 - width / 2
    return torch.meshgrid(ys, xs, indexing="ij")


def sample(image, gx, gy, cfg):
    """Reads image at grid positions given in pixels from the middle, as one warp."""
    height, width = image.shape[-2:]
    grid = torch.stack([gx * 2 / width, gy * 2 / height], -1)[None]
    mode = "bicubic" if cfg["sampling_mode"] == "bicubic" else "bilinear"
    moved = F.grid_sample(image.float(), grid, mode=mode, padding_mode=cfg["padding_mode"], align_corners=False)
    return moved.clamp(-1, 1)


def warp_2d(image, cfg, zoom, angle, tx, ty):
    """Zoom above 1 moves in, angle turns the picture counterclockwise, and translation moves the camera, in pixels."""
    height, width = image.shape[-2:]
    ys, xs = pixel_grid(height, width, image.device)
    turn = math.radians(angle)
    c, s = math.cos(turn), math.sin(turn)
    gx = (c * xs - s * ys) / zoom + tx
    gy = (s * xs + c * ys) / zoom + ty
    return sample(image, gx, gy, cfg)


def rotation(rx, ry, rz):
    """A rotation matrix from angles in degrees around x, y and z."""
    ax, ay, az = (math.radians(v) for v in (rx, ry, rz))
    x = torch.tensor([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    y = torch.tensor([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    z = torch.tensor([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return z @ y @ x


def warp_3d(image, distance, cfg, move, turn):
    """Moves a camera through the scene. Each new pixel looks up where its point was in the last frame.

    The point's distance is read at the new pixel, which is close enough for the small steps of one frame.
    move is (x, y, z) in Disco's units; z above 0 flies forward. turn is degrees around x, y and z.
    """
    height, width = image.shape[-2:]
    ys, xs = pixel_grid(height, width, image.device)
    focal = (height / 2) / math.tan(math.radians(cfg["fov"]) / 2)
    z = distance
    points = torch.stack([xs * z / focal, ys * z / focal, z], -1)
    spin = rotation(*turn).to(image.device)
    shift = torch.tensor(move, device=image.device, dtype=torch.float32) * TRANSLATION_SCALE
    old = points @ spin.T + shift
    depth = old[..., 2].clamp(min=1e-3)
    gx, gy = focal * old[..., 0] / depth, focal * old[..., 1] / depth
    behind = old[..., 2] <= 1e-3
    gx, gy = gx.masked_fill(behind, 1e6), gy.masked_fill(behind, 1e6)  # off the frame, so padding fills it
    return sample(image, gx, gy, cfg)


class Camera:
    """Moves the last frame for the next one."""

    def __init__(self, cfg, frames, device, status, depth_model=None):
        """`depth_model` gives the render's one depth model, which the 3D camera shares."""
        self.cfg = cfg
        self.mode = cfg["animation_mode"]
        self.path = motion(cfg, frames)
        self.frames = frames
        self.pingpong = cfg.get("loop_mode") == "Ping-pong"
        self.meter = None
        if self.mode == "3D":  # the 3D camera needs to know how far each point is
            if depth_model is None:
                import depth
                depth_model = lambda: depth.Depth(device, status)
            self.meter = depth_model()

    def at(self, frame):
        return {name: values[frame] for name, values in self.path.items()}

    def distance(self, image):
        """How far each pixel is, from Depth Anything's relative depth, spread between NEAR and FAR."""
        near = self.meter.closeness(image.add(1).div(2))[0]
        low, high = torch.quantile(near.flatten(), torch.tensor([0.02, 0.98], device=near.device))
        share = ((near - low) / (high - low).clamp(min=1e-6)).clamp(0, 1)
        return 1 / (share * (1 / NEAR - 1 / FAR) + 1 / FAR)

    def step(self, frame):
        """The keyframe step that frame takes, and whether it goes back along it. Ping-pong spends the first half
        of the clip going out and the second coming back the same way, step by step, so the frame after the last
        is the first again. An odd clip holds still for one frame at the turn, and the step is None."""
        half = self.frames // 2
        if not self.pingpong or frame <= half:
            return frame, False
        if frame <= self.frames - half:
            return None, False
        return self.frames + 1 - frame, True

    def move(self, image, frame, back=False):
        """Moves the picture by one frame's camera step. back undoes the step, close enough for one frame."""
        step, reverse = self.step(frame)
        if step is None:
            return image
        v = self.at(step)
        sign = -1 if back != reverse else 1
        if self.mode == "2D":
            return warp_2d(image, self.cfg, v["zoom"] ** sign, sign * v["angle"], sign * v["translation_x"],
                           sign * v["translation_y"])
        return warp_3d(image, self.distance(image), self.cfg,
                       tuple(sign * v[k] for k in ("translation_x", "translation_y", "translation_z")),
                       tuple(sign * v[k] for k in ("rotation_3d_x", "rotation_3d_y", "rotation_3d_z")))


def sharpen(image, amount):
    """An unsharp mask, which wins back the softness each resample adds over many frames."""
    if not amount:
        return image
    blurred = TF.gaussian_blur(image, 5, 1.0)
    return (image + amount * (image - blurred)).clamp(-1, 1)


WHITE = (0.95047, 1.0, 1.08883)
TO_XYZ = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]


def to_lab(image):
    """-1 to 1 RGB into CIE Lab, where light and color can be matched apart."""
    rgb = image.add(1).div(2).clamp(0, 1)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = torch.einsum("ij,bjhw->bihw", torch.tensor(TO_XYZ, device=image.device), linear)
    xyz = xyz / torch.tensor(WHITE, device=image.device)[None, :, None, None]
    f = torch.where(xyz > 0.008856, xyz.clamp(min=1e-8) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def from_lab(lab):
    fy = (lab[:, 0] + 16) / 116
    f = torch.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], 1)
    xyz = torch.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787)
    xyz = xyz * torch.tensor(WHITE, device=lab.device)[None, :, None, None]
    linear = torch.einsum("ij,bjhw->bihw", torch.linalg.inv(torch.tensor(TO_XYZ, device=lab.device)), xyz).clamp(0, 1)
    rgb = torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp(min=1e-8) ** (1 / 2.4) - 0.055)
    return rgb.mul(2).sub(1).clamp(-1, 1)


def match_histograms(source, reference):
    """Gives each channel of source the reference's spread of values, keeping source's order of pixels."""
    flat, ref = source.flatten(2), reference.flatten(2)
    order = flat.argsort(-1)
    rank = order.argsort(-1)  # where each pixel falls among its channel's values
    wanted = ref.sort(-1).values
    if wanted.shape[-1] != flat.shape[-1]:
        wanted = F.interpolate(wanted, size=flat.shape[-1], mode="linear", align_corners=True)
    return wanted.gather(-1, rank).view_as(source)


def hold_colors(image, reference, strength):
    """Matches the light and color of the reference frame, channel by channel in Lab, as Deforum's
    color coherence does. This stops colors drifting or burning out over a long clip."""
    if not strength or reference is None:
        return image
    lab = to_lab(image.float())
    matched = match_histograms(lab, to_lab(reference.float()))
    return from_lab(lab + strength * (matched - lab))


def video_flow(before, after):
    """Optical flow between two video frames: for each pixel of after, how far away it was in before."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ValueError("Follow the video's motion needs OpenCV. Run setup again to add it.") from None
    gray = [cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY) for frame in (after, before)]
    finder = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return torch.from_numpy(finder.calc(gray[0], gray[1], None))


def follow_flow(image, flow, cfg):
    """Moves a frame along the video's own motion, so its paint travels with what moves."""
    flow = flow.to(image.device)
    ys, xs = pixel_grid(*image.shape[-2:], image.device)
    return sample(image, xs + flow[..., 0], ys + flow[..., 1], cfg)


# In-between frames -------------------------------------------------------------------------------

def add_between(frames, count, folder, device, status, wrap=False):
    """Writes count frames between each pair of painted ones into folder, and returns the whole list. With wrap,
    the last frame gets in-betweens into the first too, for a clip that loops."""
    import rife
    smoother = rife.Smoother(device, status)
    os.makedirs(folder, exist_ok=True)

    def load(name):
        return TF.to_tensor(Image.open(name).convert("RGB")).unsqueeze(0).mul(2).sub(1).to(device)

    out, before = [], load(frames[0])
    for n, name in enumerate(frames):
        out.append(name)
        if n + 1 == len(frames) and not wrap:
            break
        after = load(frames[(n + 1) % len(frames)])
        for k, image in enumerate(smoother.between(before, after, count)):
            path = os.path.join(folder, f"{n:05d}_{k + 1}.png")
            TF.to_pil_image(image[0].add(1).div(2).clamp(0, 1).cpu()).save(path)
            out.append(path)
        before = after
    return out


# Video in and out --------------------------------------------------------------------------------

def need_av(why):
    try:
        import av
    except ImportError:
        raise ValueError(f"{why} needs PyAV. Run setup again to add it.") from None
    return av


class VideoInput:
    """Frames of a video file, every nth one, sized to the frame."""

    def __init__(self, path, every, width, height, fit, size_image, limit, pingpong=False):
        """With pingpong, the frames run forward and then back, without repeating the two ends, so the clip comes
        back to its first frame. limit counts both ways."""
        if pingpong:
            limit = (limit + 2) // 2
        if not path or not os.path.isfile(path):
            raise ValueError("Video Input needs a video file. Choose one under Animation.")
        av = need_av("Video Input")
        self.frames = []
        with av.open(path) as box:
            stream = box.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            self.fps = float(rate) / every if rate else None
            for n, frame in enumerate(box.decode(stream)):
                if n % every == 0:
                    self.frames.append(size_image(frame.to_image().convert("RGB"), width, height, fit))
                    if len(self.frames) >= limit:
                        break
        if not self.frames:
            raise ValueError("No frames could be read from that video.")
        if pingpong and len(self.frames) > 2:
            self.frames += self.frames[-2:0:-1]


def save_video(frames, fps, path):
    """Writes PNG frames into an H.264 MP4 that plays almost anywhere."""
    av = need_av("MP4")
    with Image.open(frames[0]) as first:
        width, height = first.size
    even = (width // 2 * 2, height // 2 * 2)  # H.264 in yuv420p needs even sides
    with av.open(path, "w") as box:
        stream = box.add_stream("libx264", rate=round(fps))
        stream.width, stream.height = even
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "slow"}
        for name in frames:
            frame = av.VideoFrame.from_image(Image.open(name).convert("RGB").crop((0, 0, *even)))
            for packet in stream.encode(frame):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)


def save_progress_video(frames, path, fps=24, hold=1.5):
    """An MP4 of a picture forming, from its previews and the finished image last. Previews can differ in size,
    as tiles show a small copy while a start picture is full size, so each is scaled to the finished image's
    size. The finished image stays on screen for `hold` seconds at the end."""
    av = need_av("MP4")
    with Image.open(frames[-1]) as last:
        width, height = last.size
    scale = min(1.0, 1920 / max(width, height))  # past 1920 pixels the video only grows
    even = (max(int(width * scale) // 2 * 2, 2), max(int(height * scale) // 2 * 2, 2))
    with av.open(path, "w") as box:
        stream = box.add_stream("libx264", rate=fps)
        stream.width, stream.height = even
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        names = frames + [frames[-1]] * round(fps * hold)
        for name in names:
            with Image.open(name) as picture:
                frame = av.VideoFrame.from_image(picture.convert("RGB").resize(even, Image.LANCZOS))
            for packet in stream.encode(frame):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)


def save_animation(frames, fps, path, longest=None):
    """Writes an animated WebP or GIF, chosen by the file's extension."""
    with Image.open(frames[0]) as first:
        width, height = first.size
    # Every frame is held in RAM while encoding. A long clip at a large size is made smaller to fit in about 1.5GB;
    # the MP4 keeps the full size.
    budget = 1.5 * 2**30 / (len(frames) * width * height * 3)
    if budget < 1:
        side = int(max(width, height) * budget ** 0.5)
        longest = min(longest, side) if longest else side
    pictures = []
    for name in frames:
        picture = Image.open(name).convert("RGB")
        if longest:
            picture = ImageOps.contain(picture, (longest, longest), Image.LANCZOS)
        pictures.append(picture)
    duration = round(1000 / fps)
    if path.lower().endswith(".gif"):
        pictures = [p.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG) for p in pictures]
        pictures[0].save(path, save_all=True, append_images=pictures[1:], duration=duration, loop=0, disposal=1)
    else:
        pictures[0].save(path, save_all=True, append_images=pictures[1:], duration=duration, loop=0, quality=88, method=4)


# The clip ----------------------------------------------------------------------------------------------
def film(r):
    """Renders the frames of the render `r`'s animation, each from the last one moved by the camera or from a video
    frame, then saves the clip."""
    cfg, hooks, seed, grid, final, side_x, side_y = r.cfg, r.hooks, r.seed, r.grid, r.final, r.side_x, r.side_y
    steer, fixed_control = r.steer, r.fixed_control
    mode = cfg["animation_mode"]
    loop = cfg["loop_mode"] if cfg["loop_mode"] in ("Paint back to the start", "Ping-pong") else None
    r.write_notes(again=bool(r.resume))  # claims the run number, so a stopped animation keeps its frames
    frames_dir = r.named("_frames")
    os.makedirs(frames_dir, exist_ok=True)
    video = None
    if mode == "Video Input":
        hooks.status("Reading the video")
        video = VideoInput(cfg["video_init_path"], max(int(cfg["extract_nth_frame"]), 1), *final,
                           r.init_fit, size_start_image, int(cfg["max_frames"]), loop == "Ping-pong")
    total = min(int(cfg["max_frames"]), len(video.frames)) if video else int(cfg["max_frames"])
    # Seamless loop: the last `reach` frames are painted more and more like the frame they lead back to. With
    # ping-pong that is the frame at the same point on the way out, so it must already be painted.
    reach = min(max(int(cfg["loop_frames"]), 1), (total - 1) // 2 if loop == "Ping-pong" else total - 1) if loop else 0
    circle = None  # with ping-pong and one shared noise, the noise goes once around a circle between two draws
    r.n_batches = total
    camera = Camera(cfg, total, device, hooks.status, r.depth_model) if mode in ("2D", "3D") else None
    skips = [min(round(cfg["steps"] * share / 100), cfg["steps"] - 1) for share in keyframes.skips(cfg, total)]
    if cfg["frames_scale"] and r.lpips_model is None:
        r.lpips_model = load_lpips()

    def paint(start, skip, n, hand=True):
        """One frame. With tiles, each frame is tiled like a still."""
        # before a handoff moves where Disco starts, so the views stay spread over the whole frame. A video's frames
        # and every frame after the first are clip frames; a 2D or 3D clip's first frame is its start image.
        r.fit_views(skip, frame=bool(video) or n > 0)
        if hand:  # the last frame's Small first is not this one's
            r.pull["set"], r.pull["to"] = False, None
        if hand and r.handing:
            start, skip = r.hand_over(start, skip, n, (side_x, side_y) if grid == 1 else final, frame_seed)
            r.reseed(frame_seed)  # each frame starts alike, whatever the handoff drew
        if grid == 1:
            start, skip = r.small_start(start, skip, n)
            return r.paint(start, side_x, side_y, skip, n)
        return tiles.paint(r, None if start is None else start.float().cpu(), skip, n).to(device)

    turbo = max(int(cfg["turbo_steps"]), 1) if cfg["turbo_mode"] and camera else 1
    refresh = {at + int(cfg["prompt_fade"]) for at, _ in r.all_stats[1:]}  # a new prompt gets its own colors
    first, previous, saved = None, None, {}
    rendered_first = False  # the first frame's Render first pass already had its handoff
    between, painted_at = [], 0  # with Turbo, the frames moved on from the last repaint
    begin = 0

    def frame_at(n):
        return TF.to_tensor(Image.open(saved[n]).convert("RGB")).unsqueeze(0).mul(2).sub(1).to(device)

    if r.resume:  # pick up after the last frame on disk, as if the run had never stopped
        while os.path.isfile(os.path.join(frames_dir, f"{begin:05d}.png")):
            saved[begin] = os.path.join(frames_dir, f"{begin:05d}.png")
            begin += 1
        if not begin:
            raise ValueError("That clip has no frames to continue from.")
        if begin >= total:
            raise ValueError(f"That clip already has {begin} frames. Raise Frames to make it longer.")
        previous = frame_at(begin - 1)
        if tuple(previous.shape[-2:]) != (final[1], final[0]):
            raise ValueError(f"That clip's frames are {previous.shape[-1]}x{previous.shape[-2]}. Set the same size and Tiles "
                             "to carry it on.")
        first = frame_at(max([0] + [n for n in refresh if n < begin]))
        painted_at = begin - 1
        hooks.status(f"Carrying on from frame {begin + 1}")

    def keep(n, image):
        saved[n] = os.path.join(frames_dir, f"{n:05d}.png")
        r.to_image(image).save(saved[n])

    try:
        for n in range(begin, total):
            r.stats = stats_at(r.all_stats, n, cfg["prompt_fade"])
            # Only a video can share one noise: a moving camera drags fixed noise across the scene into streaks.
            frame_seed = seed if video and cfg["video_init_seed_continuity"] else seed + n
            if video and loop == "Ping-pong" and cfg["video_init_seed_continuity"]:
                # The same noise would paint the way back as a copy of the way out. Going round a circle, each
                # frame's noise is close to the last one's, and the frame after the last is back at the first's.
                sized = (1, 3, final[1], final[0]) if grid > 1 else (1, 3, side_y, side_x)
                if circle is None:
                    circle = [torch.randn(sized, generator=torch.Generator().manual_seed(seed + k)) for k in (0, 1)]
                turn = 2 * math.pi * n / total
                r.given["noise"] = math.cos(turn) * circle[0] + math.sin(turn) * circle[1]
            r.reseed(frame_seed)
            if steer and (fixed_control is not None or video):
                # the Control image, or else the clean video frame, never the start the last frame was mixed into
                steer.set_source(fixed_control if fixed_control is not None else TF.to_tensor(video.frames[n]).unsqueeze(0))
            if video:
                shown = TF.to_tensor(video.frames[n]).unsqueeze(0).mul(2).sub(1).to(device)
                start = shown
                if previous is None and cfg["video_blend"]:
                    # The first frame builds on a first pass of itself, as later frames build on the last
                    # one. Without this it stays close to the video while the rest of the clip doesn't.
                    hooks.status("Painting a first pass of the first frame")
                    previous = paint(shown, r.skip_steps, n)  # a full repaint from Denoise, as a still would be
                if previous is not None and cfg["video_blend"]:
                    last = previous
                    if cfg["video_flow"] and n:  # the last frame's paint travels with what moved in the video
                        last = follow_flow(previous, video_flow(video.frames[n - 1], video.frames[n]), cfg)
                    # The carried paint is repainted every frame, and each pass adds contrast and color. Held to the
                    # first frame's light and color, as camera moves are, it can't compound into a burned-out clip.
                    last = hold_colors(last, first, cfg["color_hold"])
                    start = shown + cfg["video_blend"] * (last - shown)
                skip = skips[n]
            elif n == 0:
                shown = start = r.start_tensor()
                skip = r.skip_steps  # from a start image or noise, with tiles or without
                if grid > 1 and not r.start_image and cfg["tile_start"] == "Render first":
                    hooks.status("Painting the first frame at tile size")
                    r.fit_views(r.skip_steps, frame=bool(video))
                    rough, rough_skip = r.hand_over(None, r.skip_steps, n, (side_x, side_y), frame_seed)
                    r.reseed(frame_seed)
                    rough = r.to_image(r.denoise(rough, side_x, side_y, rough_skip, n)).resize(final, Image.LANCZOS)
                    shown = start = TF.to_tensor(rough).unsqueeze(0).mul(2).sub(1).to(device)
                    skip, rendered_first = r.tile_skip, True
            else:
                moved = sharpen(camera.move(previous, n), cfg["frame_sharpen"])
                shown = start = hold_colors(moved, first, cfg["color_hold"])
                skip = skips[n]
            if reach and n >= total - reach:
                back_to = (n - (total - reach) + 1) / (reach + 1)
                target = frame_at(total - n if loop == "Ping-pong" else 0)
                hooks.status(f"Painting back to the start, {round(100 * back_to)}% of the way")
                start = start + back_to * (target - start)
            r.init_scale = 0 if start is None else cfg["frames_scale"] if (n or video) else cfg["init_scale"]
            if steer and fixed_control is None and not video:  # a camera move follows the frame it starts from
                steer.set_source(None if shown is None else shown.add(1).div(2))
            r.given["ghost"] = shown.add(1).div(2).float() if r.frame_ghost and shown is not None else None
            if turbo > 1 and n % turbo and n < total - 1:
                between.append((n, start))  # between repaints the camera alone moves the picture
                previous = start
                hooks.preview(r.to_image(start))
                continue
            gc.collect()
            torch.cuda.empty_cache()
            r.reseed(frame_seed)  # passes before this one drew noise too, and every frame must start alike
            image = paint(start, skip, n, hand=not rendered_first)
            rendered_first = False
            back = image
            for m, ahead in reversed(between):  # fade each one toward the new repaint, moved back to it
                back = camera.move(back, m + 1, back=True)
                keep(m, ahead + (m - painted_at) / (n - painted_at) * (back - ahead))
            between, painted_at = [], n
            previous = image
            first = image if first is None or n in refresh else first
            keep(n, image)
    finally:  # a stopped animation still gets a video of the frames it has
        for m, ahead in between:
            keep(m, ahead)
        saved = [saved[n] for n in sorted(saved)]
        if saved:
            fps = max(float(cfg["fps"]), 1)
            extra = int(cfg["in_between"])
            if extra > 0 and len(saved) > 1:  # the clip keeps its length, at a higher frame rate
                saved = add_between(saved, extra, os.path.join(frames_dir, "between"), device, hooks.status,
                                    wrap=bool(loop) and len(saved) == total)
                fps *= extra + 1
            hooks.status("Saving the video")
            base = r.named("")
            if cfg["save_mp4"] and len(saved) > 1:
                save_video(saved, fps, base + ".mp4")
            if cfg["save_webp"] and len(saved) > 1:
                save_animation(saved, fps, base + ".webp")
            if cfg["save_gif"] and len(saved) > 1:
                save_animation(saved, fps, base + ".gif", longest=512)
            hooks.image(r.save(r.to_image(previous), 0))
