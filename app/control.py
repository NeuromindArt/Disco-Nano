"""Controls: ControlNet's idea, done through guidance. Disco's model has nowhere to plug a ControlNet in, so each
control measures how far the render's current estimate is from a control picture, and pushes it closer at every
step, beside the prompt's push.

Edges hold the control picture's outlines, Depth its near and far, Pose the skeletons of the people in it, Person
their real silhouettes, clothes and broad colors, Face each face's features, and Face shape each face's outline and
near and far, measured close up, which holds which way a head is turned. Face detail gives every face close-ups
of its own, found in the control picture or, with none, in Disco's own estimate as the render goes. Person works on the picture between steps, as a ghost does, since
an outline alone can be met with a few painted strokes and a filled shape can't. The
control picture is read once, while it is clean, so nothing needs to see Disco's blurry early estimate except the
measures that are cheap and steady on it. Edges and Pose only ever ask for more edge where the control has one,
never less, so plain areas stay free for the prompt to paint.
"""
import contextlib
import math

import torch
import torch.nn.functional as F

import lab
import models
from imaging import blur
from keyframes import schedule_at

LUMA = (0.299, 0.587, 0.114)
STRUCTURE_SIDE = 256  # edges are measured at about this size, so outlines count and fine texture doesn't
# COCO's 17 keypoints, joined into limbs: head, arms, torso sides and legs.
LIMBS = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (5, 6), (11, 12)]
TORSO = (5, 6, 12, 11)
PERSON_SCORE = 0.7
POINT_SCORE = 2.0  # Keypoint R-CNN's own score for a joint; below this it is a guess
FACE_SIDE = 192  # Face shape measures each face at this size, whatever its size in the frame


def wanted(cfg):
    return any(float(cfg.get(name) or 0) > 0 for name in ("edge_control", "depth_control", "pose_control", "person_control",
                                                           "face_control", "face_shape", "face_detail"))


def edges(image):
    """How strong an outline each spot of an image of 0 to 1 has, at about STRUCTURE_SIDE pixels on the short side."""
    gray = (image.float() * torch.tensor(LUMA, device=image.device)[None, :, None, None]).sum(1, keepdim=True)
    shrink = max(min(gray.shape[-2:]) // STRUCTURE_SIDE, 1)
    if shrink > 1:
        gray = F.avg_pool2d(gray, shrink)
    gray = blur(gray, 1.0)
    sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=gray.device, dtype=gray.dtype)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, sobel.view(1, 1, 3, 3))
    gy = F.conv2d(padded, sobel.t().reshape(1, 1, 3, 3))
    return (gx.square() + gy.square() + 1e-8).sqrt()


def one_sided(target, render):
    """How much outline the render is missing where the target has one. A render outline a pixel away still counts."""
    near = F.max_pool2d(render, 3, stride=1, padding=1)
    return (target * F.relu(target - near)).sum() / target.sum().clamp(min=1e-6)


class Detector:
    """A network that finds something in a clean picture, kept in RAM between renders. It comes to the GPU only while
    it looks, since it looks once per control picture."""

    def __init__(self, net, name, status, prefix=""):
        weights = torch.load(models.ensure(name, status), map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix(prefix): v for k, v in weights.items()})
        self.net = net.eval().requires_grad_(False)

    def to(self, device):
        self.net.to(device)
        return self

    @contextlib.contextmanager
    def looking(self, device):
        """The network on `device` while it looks, and back in RAM after."""
        self.net.to(device)
        try:
            yield self.net
        finally:
            self.net.to("cpu")


class Pose(Detector):
    """Keypoint R-CNN from torchvision, which finds people's joints."""

    def __init__(self, status):
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        super().__init__(keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_keypoints=17), models.POSE, status)

    @torch.no_grad()
    def body(self, image, device):
        """A soft map of 0 to 1 that covers each person's limbs, torso and head, at the image's size, or None."""
        height, width = image.shape[-2:]
        with self.looking(device) as net:
            found = net([image[0].float().to(device)])[0]
        canvas = torch.zeros(height, width, device=device)
        ys, xs = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
        people = 0
        for box, score, points, sure in zip(found["boxes"], found["scores"], found["keypoints"], found["keypoints_scores"]):
            if score < PERSON_SCORE:
                continue
            people += 1
            tall = float(box[3] - box[1])
            thick = max(tall / 14, 3.0)
            ok = sure > POINT_SCORE
            for a, b in LIMBS:
                if ok[a] and ok[b]:
                    canvas = torch.maximum(canvas, _stroke(xs, ys, points[a, :2], points[b, :2], thick))
            if all(ok[i] for i in TORSO):
                canvas = torch.maximum(canvas, _inside(xs, ys, points[list(TORSO), :2]))
            if ok[0]:
                canvas = torch.maximum(canvas, _stroke(xs, ys, points[0, :2], points[0, :2], tall / 12))
        if not people:
            return None
        return blur(canvas[None, None], max(min(height, width) / 200, 1.0))


class People(Detector):
    """Mask R-CNN from torchvision, which cuts out each person."""

    def __init__(self, status):
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        super().__init__(maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None), models.PERSON, status)

    @torch.no_grad()
    def silhouettes(self, image, device):
        """A soft map of 0 to 1 over every person found, at the image's size, or None when there is nobody."""
        with self.looking(device) as net:
            found = net([image[0].float().to(device)])[0]
        keep = (found["labels"] == 1) & (found["scores"] >= PERSON_SCORE)  # COCO's class 1 is a person
        if not keep.any():
            return None
        return found["masks"][keep].amax(0, keepdim=True)  # [1, 1, height, width], already soft at the edges


class Faces(Detector):
    """RetinaFace, which finds faces with their landmarks."""

    def __init__(self, status):
        import retinaface
        super().__init__(retinaface.RetinaFace(), models.FACES, status, prefix="module.")
        self.detect = retinaface.detect

    def find(self, image, device, threshold=0.6):
        """Faces in an image of 0 to 1 as boxes (x0, y0, x1, y1) in shares of its width and height."""
        height, width = image.shape[-2:]
        with self.looking(device) as net:
            found = self.detect(net, image.float().to(device), threshold)
        return [(x0 / width, y0 / height, x1 / width, y1 / height) for x0, y0, x1, y1, *_ in found]


def _face_size(crop):
    """A face's crop at FACE_SIDE on its long side, so its outline is measured at the same scale whatever its size."""
    height, width = crop.shape[-2:]
    shrink = FACE_SIDE / max(height, width)
    size = (max(round(height * shrink), 8), max(round(width * shrink), 8))
    return F.interpolate(crop, size=size, mode="bilinear", align_corners=False, antialias=shrink < 1)


def _face_box(face, height, width):
    """A face's box, a little larger so the jaw and hairline count, as slices in pixels of a picture this size.
    None when it is too small to hold."""
    x0, y0, x1, y1 = face
    cx, cy = (x0 + x1) / 2 * width, (y0 + y1) / 2 * height
    w, h = (x1 - x0) * width * 1.3, (y1 - y0) * height * 1.3
    left, top = max(int(cx - w / 2), 0), max(int(cy - h / 2), 0)
    right, bottom = min(int(math.ceil(cx + w / 2)), width), min(int(math.ceil(cy + h / 2)), height)
    if right - left < 4 or bottom - top < 4:
        return None
    return (slice(None), slice(None), slice(top, bottom), slice(left, right))


def _stroke(xs, ys, a, b, radius):
    """1 within radius of the segment from a to b."""
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    t = ((xs - ax) * dx + (ys - ay) * dy) / length if length > 0 else torch.zeros_like(xs)
    t = t.clamp(0, 1)
    distance = ((xs - ax - t * dx).square() + (ys - ay - t * dy).square()).sqrt()
    return (distance <= radius).float()


def _inside(xs, ys, corners):
    """1 inside the polygon through corners, in order."""
    inside = torch.zeros_like(xs, dtype=torch.bool)
    n = corners.shape[0]
    for i in range(n):
        x1, y1 = float(corners[i, 0]), float(corners[i, 1])
        x2, y2 = float(corners[(i + 1) % n, 0]), float(corners[(i + 1) % n, 1])
        if y1 == y2:
            continue
        crosses = ((ys >= min(y1, y2)) & (ys < max(y1, y2)))
        at_x = x1 + (ys - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (xs < at_x)
    return inside.float()


class Control:
    """Pushes a render toward a control picture's edges, depth and pose. Pushes are measured against the prompt's
    push, so a strength of 1 is about as strong as the prompt when the render is furthest from the control, and
    each one eases off as the render comes to match."""

    def __init__(self, cfg, schedule, device, status, depth_model=None, keep=lambda key, build: build()):
        """`depth_model` gives the render's one depth model, which Depth control and Face shape share, and `keep`
        keeps the other helper models in RAM between renders."""
        self.device = device
        # each control lets go at its own point: depth and pose shape the layout, edges matter into the detail
        self.curves = {kind: schedule(cfg[f"{kind}_curve"]) for kind in ("edge", "depth", "pose", "person", "face", "face_shape")}
        self.strength = {"edges": float(cfg["edge_control"] or 0), "depth": float(cfg["depth_control"] or 0),
                         "pose": float(cfg["pose_control"] or 0)}
        self.face_shape = float(cfg.get("face_shape") or 0)
        self.depth = None
        if self.strength["depth"] or self.face_shape:
            if depth_model is None:
                import depth
                depth_model = lambda: depth.Depth(device, status)
            self.depth = depth_model()
        self.pose = None
        if self.strength["pose"]:
            self.pose = keep(("pose",), lambda: Pose(status))  # kept in RAM between renders, as Disco's own models are
        self.hold = float(cfg.get("person_control") or 0)
        self.people = keep(("people",), lambda: People(status)) if self.hold else None
        self.face_hold = float(cfg.get("face_control") or 0)
        self.face_cuts = int(cfg.get("face_detail") or 0)
        self.faces_net = keep(("faces",), lambda: Faces(status)) if self.face_hold or self.face_shape or self.face_cuts else None
        self.faces = []  # the control picture's faces, as shares of the frame
        self.seen_faces, self.seen_at = [], None  # faces found in Disco's own estimate, when there is no control picture
        self.status = status
        self.given = None
        self.source = None  # the control picture, 0 to 1, at the frame size
        self.targets = {}
        self.peaks = {}
        self.found_nobody = False

    def set_source(self, image):
        """A new control picture, as a tensor of 0 to 1. The targets are made again when they are first needed."""
        if image is self.given:
            return
        self.given = image
        self.source = None if image is None else image.float().to(self.device)
        self.targets, self.peaks = {}, {}
        self.faces, self.seen_faces, self.seen_at = [], [], None
        if self.source is not None and self.faces_net is not None:
            self.faces = self.faces_net.find(self.source, self.device)
            if not self.faces and (self.face_hold or self.face_shape):
                self.status("Face control found no face in the control picture")
        if self.source is not None and self.strength["pose"]:
            body = self.pose.body(self.source, self.device)
            self.found_nobody = body is None
            if body is None:
                self.status("Pose control found nobody in the control picture")
            self.targets["body"] = body
        if self.source is not None and self.people is not None:
            people = self.people.silhouettes(self.source, self.device)
            if people is None:
                self.status("Person control found nobody in the control picture")
            self.targets["people"] = people

    def _target(self, kind, height, width):
        key = (kind, height, width)
        if key not in self.targets:
            with torch.no_grad():
                sized = F.interpolate(self.source, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
                if kind == "edges":
                    made = edges(sized)
                    # a strong outline in the control picture counts as 1, and the render's are measured the same way
                    self.targets[("scale", height, width)] = torch.quantile(made.flatten(), 0.98).clamp(min=1e-4)
                    made = (made / self.targets[("scale", height, width)]).clamp(max=1)
                elif kind == "depth":
                    made = self.depth.relief(sized)
                else:  # the outline of the bodies Pose found, with their edge counting as a strong outline
                    body = F.interpolate(self.targets["body"], size=(height, width), mode="bilinear", align_corners=False)
                    made = edges(body)
                    made = made / made.max().clamp(min=1e-4)
            self.targets[key] = made
        return self.targets[key]

    def push(self, x_in, clip_grad, at):
        if self.source is None:
            return clip_grad
        image = x_in.add(1).div(2)
        height, width = image.shape[-2:]
        size = clip_grad.square().mean().sqrt()
        total = clip_grad
        for kind, strength in self.strength.items():
            weight = self.curves["edge" if kind == "edges" else kind][at]
            if not strength or not weight or (kind == "pose" and self.targets.get("body") is None):
                continue
            target = self._target(kind, height, width)
            if kind == "depth":
                loss = (self.depth.relief(image) - target).abs().mean()
            else:
                self._target("edges", height, width)  # the scale of a strong outline
                loss = one_sided(target, edges(image) / self.targets[("scale", height, width)])
            grad = torch.autograd.grad(loss, x_in, retain_graph=True)[0]
            rms = grad.square().mean().sqrt()
            peak = self.peaks[kind] = max(self.peaks.get(kind, 0.0), float(rms))
            if peak > 0:  # as strong as the prompt's push when furthest off, and less as the render comes to match
                total = total + strength * weight * grad * size / peak
        return self._shape_faces(x_in, image, clip_grad, total, at)

    def _face_targets(self, height, width):
        """Each face's box, with its outline, the scale of a strong outline in it, and its near and far, close up."""
        key = ("face shape", height, width)
        if key not in self.targets:
            made = []
            with torch.no_grad():
                picture = F.interpolate(self.source, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
                for face in self.faces:
                    box = _face_box(face, height, width)
                    if box is None:
                        continue
                    crop = _face_size(picture[box])
                    outline = edges(crop)
                    scale = torch.quantile(outline.flatten(), 0.98).clamp(min=1e-4)
                    made.append((box, (outline / scale).clamp(max=1), scale, self.depth.relief(crop)))
            self.targets[key] = made
        return self.targets[key]

    def _shape_faces(self, x_in, image, clip_grad, total, at):
        """Face shape: pushes each face's outline and near and far toward the control picture's, measured on a close-up
        of the face, so the jaw, the side of the nose and which cheek is nearer stay where a turned head has them. Both
        pushes are measured against the prompt's push inside the faces, as the other controls are against the whole
        picture, so they hold a head's turn without pasting its features in."""
        weight = self.curves["face_shape"][at] if self.face_shape else 0
        faces = self._face_targets(*image.shape[-2:]) if weight and self.faces else []
        if not faces:
            return total
        inside = torch.zeros_like(clip_grad[:, :1], dtype=torch.bool)
        outline_loss = relief_loss = 0
        for box, outline, scale, relief in faces:
            crop = _face_size(image[box])
            # Both ways, unlike Edge control: a nose or eye painted where the photo has none is what turns a head to
            # face forward. Compared at half size, so the features count and the brushwork on the skin doesn't.
            painted = (edges(crop) / scale).clamp(max=1)
            outline_loss = outline_loss + (F.avg_pool2d(painted, 2) - F.avg_pool2d(outline, 2)).abs().mean()
            relief_loss = relief_loss + (self.depth.relief(crop) - relief).abs().mean()
            inside[box] = True
        inside = inside.expand_as(clip_grad)
        size = clip_grad[inside].square().mean().sqrt()
        for name, loss in (("face outline", outline_loss), ("face depth", relief_loss)):
            grad = torch.autograd.grad(loss, x_in, retain_graph=True)[0]
            rms = grad[inside].square().mean().sqrt()
            peak = self.peaks[name] = max(self.peaks.get(name, 0.0), float(rms))
            if peak > 0:
                total = total + self.face_shape * weight * grad * size / peak
        return total

    def pull(self, x, estimate, i, diffusion):
        """Person and Face control, after the step at index i. Person moves the render's broad color and light and
        dark inside each person's silhouette toward the control picture's. Face does the same inside each face at a
        finer size, which holds where the eyes, nose and mouth are and their light and dark. The rest of the picture,
        and the finest detail, are left alone."""
        if self.source is None or i == 0:
            return x
        at = schedule_at(i, diffusion.num_timesteps)
        scale = math.sqrt(float(diffusion.alphas_cumprod_prev[i]))
        x = self._hold_faces(x, estimate, at, scale)
        people = self.targets.get("people")
        weight = self.curves["person"][at]
        if not self.hold or people is None or not weight:
            return x
        height, width = x.shape[-2:]
        key = ("people", height, width)
        if key not in self.targets:
            mask = F.interpolate(people, size=(height, width), mode="bilinear", align_corners=False)
            picture = F.interpolate(self.source, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
            self.targets[key] = (mask, picture.mul(2).sub(1))
        mask, target = self.targets[key]
        size = max(4, min(height, width) // 48)  # shapes broader than this are held: about 7 pixels at 576x320
        change = lab.low_pass(mask * (target - estimate.detach().float()), size)
        return x + (self.hold * weight * scale * change).to(x.dtype)

    def _hold_faces(self, x, estimate, at, scale):
        weight = self.curves["face"][at]
        if not self.face_hold or not self.faces or not weight:
            return x
        height, width = x.shape[-2:]
        target = self.targets.get(("face picture", height, width))
        if target is None:
            target = F.interpolate(self.source, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
            target = self.targets[("face picture", height, width)] = target.mul(2).sub(1)
        change = torch.zeros_like(x, dtype=torch.float32)
        for face in self.faces:
            box = _face_box(face, height, width)
            if box is None:
                continue
            top, bottom, left, right = box[2].start, box[2].stop, box[3].start, box[3].stop
            size = max(1, round(min(right - left, bottom - top) / 12))  # about an eye's width: features held, skin free
            gap = target[box] - estimate[box].detach().float()
            if size > 1:
                gap = F.interpolate(F.avg_pool2d(gap, size, ceil_mode=True), size=gap.shape[-2:], mode="bilinear", align_corners=False)
            ys = torch.linspace(-1, 1, bottom - top, device=x.device)[:, None]
            xs = torch.linspace(-1, 1, right - left, device=x.device)[None, :]
            oval = (1 - (xs.square() + ys.square())).clamp(0, 1).sqrt()  # strongest in the middle, nothing at the corners
            change[box] = gap * oval
        return x + (self.face_hold * weight * scale * change).to(x.dtype)

    def face_crops(self, x_in, at):
        """Close-ups for Face detail, as (left, top, size) squares in x_in's pixels: Face detail of them for each face.
        Faces come from the control picture, or else from Disco's own estimate, looked for again every 50 points of
        the 1000-step schedule once the picture is clear enough to show one."""
        if not self.face_cuts:
            return []
        height, width = x_in.shape[-2:]
        faces = self.faces
        if self.source is None:
            if at >= 250 and (self.seen_at is None or at - self.seen_at >= 50 or at < self.seen_at):
                picture = x_in.detach().add(1).div(2).clamp(0, 1)
                self.seen_faces, self.seen_at = self.faces_net.find(picture, self.device, 0.5), at
            faces = self.seen_faces
        crops = []
        for x0, y0, x1, y1 in faces:
            side = max((x1 - x0) * width, (y1 - y0) * height)
            for n in range(self.face_cuts):
                size = int(min(max(side * (1.5 + 0.6 * n) * (0.9 + 0.2 * float(torch.rand([]))), 48), width, height))
                cx = (x0 + x1) / 2 * width + (float(torch.rand([])) - 0.5) * side * 0.3
                cy = (y0 + y1) / 2 * height + (float(torch.rand([])) - 0.5) * side * 0.3
                left = int(min(max(cx - size / 2, 0), width - size))
                top = int(min(max(cy - size / 2, 0), height - size))
                crops.append((left, top, size))
        return crops
