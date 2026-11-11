"""RetinaFace with a ResNet-50 backbone (Deng et al.), which finds faces, from tiny to frame-filling, with five
landmarks each. The network layout follows biubug6's Pytorch_Retinaface (MIT), whose weights facexlib publishes."""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import nms

MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
STEPS = [8, 16, 32]
VARIANCE = (0.1, 0.2)
MEAN_BGR = (104.0, 117.0, 123.0)


def _conv_bn(inp, out, kernel=3, relu=True):
    layers = [nn.Conv2d(inp, out, kernel, 1, kernel // 2, bias=False), nn.BatchNorm2d(out)]
    return nn.Sequential(*layers, nn.ReLU(inplace=True)) if relu else nn.Sequential(*layers)


class _SSH(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv3X3 = _conv_bn(inp, out // 2, relu=False)
        self.conv5X5_1 = _conv_bn(inp, out // 4)
        self.conv5X5_2 = _conv_bn(out // 4, out // 4, relu=False)
        self.conv7X7_2 = _conv_bn(out // 4, out // 4)
        self.conv7x7_3 = _conv_bn(out // 4, out // 4, relu=False)

    def forward(self, x):
        first = self.conv5X5_1(x)
        return F.relu(torch.cat([self.conv3X3(x), self.conv5X5_2(first), self.conv7x7_3(self.conv7X7_2(first))], 1))


class _FPN(nn.Module):
    def __init__(self, ins, out):
        super().__init__()
        self.output1, self.output2, self.output3 = (_conv_bn(c, out, kernel=1) for c in ins)
        self.merge1, self.merge2 = _conv_bn(out, out), _conv_bn(out, out)

    def forward(self, feats):
        o1, o2, o3 = self.output1(feats[0]), self.output2(feats[1]), self.output3(feats[2])
        o2 = self.merge2(o2 + F.interpolate(o3, size=o2.shape[-2:], mode="nearest"))
        o1 = self.merge1(o1 + F.interpolate(o2, size=o1.shape[-2:], mode="nearest"))
        return [o1, o2, o3]


class _Head(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.conv1x1 = nn.Conv2d(256, 2 * outputs, 1)
        self.outputs = outputs

    def forward(self, x):
        return self.conv1x1(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.outputs)


class RetinaFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = IntermediateLayerGetter(resnet50(weights=None), {"layer2": 1, "layer3": 2, "layer4": 3})
        self.fpn = _FPN([512, 1024, 2048], 256)
        self.ssh1, self.ssh2, self.ssh3 = _SSH(256, 256), _SSH(256, 256), _SSH(256, 256)
        self.ClassHead = nn.ModuleList(_Head(2) for _ in range(3))
        self.BboxHead = nn.ModuleList(_Head(4) for _ in range(3))
        self.LandmarkHead = nn.ModuleList(_Head(10) for _ in range(3))

    def forward(self, x):
        feats = self.fpn(list(self.body(x).values()))
        feats = [self.ssh1(feats[0]), self.ssh2(feats[1]), self.ssh3(feats[2])]
        boxes = torch.cat([h(f) for h, f in zip(self.BboxHead, feats)], 1)
        scores = torch.cat([h(f) for h, f in zip(self.ClassHead, feats)], 1).softmax(-1)
        marks = torch.cat([h(f) for h, f in zip(self.LandmarkHead, feats)], 1)
        return boxes, scores, marks


def _priors(height, width, device):
    found = []
    for step, sizes in zip(STEPS, MIN_SIZES):
        rows, cols = math.ceil(height / step), math.ceil(width / step)
        ys, xs = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing="ij")
        cx = ((xs + 0.5) * step / width).reshape(-1, 1)
        cy = ((ys + 0.5) * step / height).reshape(-1, 1)
        per = [torch.cat([cx, cy, torch.full_like(cx, s / width), torch.full_like(cy, s / height)], 1) for s in sizes]
        found.append(torch.stack(per, 1).reshape(-1, 4))
    return torch.cat(found)


@torch.no_grad()
def detect(net, image, threshold=0.6):
    """Faces in an image of 0 to 1, [1, 3, height, width], as a list of (x0, y0, x1, y1, score, landmarks) in pixels."""
    height, width = image.shape[-2:]
    bgr = image[:, [2, 1, 0]].float() * 255 - torch.tensor(MEAN_BGR, device=image.device)[None, :, None, None]
    loc, conf, marks = net(bgr)
    priors = _priors(height, width, image.device)
    centers = priors[:, :2] + loc[0, :, :2] * VARIANCE[0] * priors[:, 2:]
    sizes = priors[:, 2:] * torch.exp(loc[0, :, 2:] * VARIANCE[1])
    scale = torch.tensor([width, height], device=image.device, dtype=torch.float32)
    boxes = torch.cat([(centers - sizes / 2) * scale, (centers + sizes / 2) * scale], 1)
    points = (priors[:, None, :2] + marks[0].view(-1, 5, 2) * VARIANCE[0] * priors[:, None, 2:]) * scale
    score = conf[0, :, 1]
    keep = score > threshold
    boxes, score, points = boxes[keep], score[keep], points[keep]
    keep = nms(boxes, score, 0.4)
    return [(*boxes[i].tolist(), float(score[i]), points[i].tolist()) for i in keep]
