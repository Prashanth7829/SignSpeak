"""GPU transfer learning for the full INCLUDE vocabulary."""
import json

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

from signspeak.corpus import CACHE, CATALOG, SAMPLES, sample
from signspeak.recognition import ROOT, IMAGE_TYPES, cuda_device

from signspeak.settings import FULL_MODEL
WEIGHTS = ROOT / "artifacts/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"


class VideoCNN(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.encoder = mobilenet_v3_small().features
        self.project = nn.Sequential(nn.LayerNorm(576 * 4), nn.Linear(576 * 4, 192),
                                     nn.LayerNorm(192), nn.GELU(), nn.Dropout(.25))
        self.temporal = nn.GRU(192, 128, batch_first=True, bidirectional=True)
        self.frame_head = nn.Linear(192, count)
        self.video_head = nn.Linear(256, count)

    def encode(self, rgb):
        batch, length = rgb.shape[:2]
        images = rgb.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
        std = images.new_tensor([.229, .224, .225])[None, :, None, None]
        images = F.interpolate(images, size=(160, 160), mode="bilinear", align_corners=False)
        spatial = self.encoder((images - mean) / std)
        return F.adaptive_avg_pool2d(spatial, 2).reshape(batch, length, -1)

    def forward(self, features):
        frames = self.project(features)
        temporal, _ = self.temporal(frames)
        return self.video_head(temporal.mean(1)), self.frame_head(frames)


def train():
    torch.manual_seed(42)
    device = cuda_device()
    rows = [row for row in json.loads(CATALOG.read_text(encoding="utf8")) if row["corpus"] == "INCLUDE"]
    labels = sorted({row["label"] for row in rows if row["split"] == "train"})
    targets = torch.tensor([labels.index(row["label"]) for row in rows], device=device)
    indices = {s: torch.tensor([i for i, r in enumerate(rows) if r["split"] == s], device=device)
               for s in ("train", "val", "test")}
    model = VideoCNN(len(labels)).to(device)
    pretrained = mobilenet_v3_small()
    pretrained.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.encoder.load_state_dict(pretrained.features.state_dict())
    model.encoder.requires_grad_(False)
    model.eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(rows), 12):
            rgb = np.stack([np.load(SAMPLES / (r["key"] + ".npy")) for r in rows[start:start+12]])
            features.append(model.encode(torch.from_numpy(rgb).to(device)))
    features = torch.cat(features)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.0006)
    best_loss = float("inf")
    def evaluate(split):
        model.eval()
        with torch.no_grad():
            video, still = zip(*(model(features[b]) for b in indices[split].split(64)))
            expected = targets[indices[split]]
            video, still = torch.cat(video), torch.cat(still)
            return dict(clips=len(expected), loss=F.cross_entropy(video, expected).item(),
                        video_accuracy=(video.argmax(1)==expected).float().mean().item(),
                        frame_accuracy=(still.argmax(2)==expected[:, None]).float().mean().item())
    for epoch in range(1, 46):
        model.train()
        idx = indices["train"]
        for batch in idx[torch.randperm(len(idx), device=device)].split(48):
            video, still = model(features[batch])
            expected = targets[batch]
            loss = F.cross_entropy(video, expected) + .35 * F.cross_entropy(
                still.flatten(0,1), expected.repeat_interleave(still.shape[1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = evaluate("val")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best = dict(epoch=epoch, validation=val, labels=labels, weights={
                k: v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if epoch % 5 == 0:
            print(epoch, val, flush=True)
    model.load_state_dict(best["weights"])
    model.eval()
    minimum_scores = {}
    with torch.no_grad():
        batch_indices = indices["val"]
        video, still = zip(*(model(features[b]) for b in batch_indices.split(64)))
        for mode, logits in (("video", torch.cat(video)), ("image", torch.cat(still)[:, 4])):
            scores, choices = logits.softmax(1).max(1)
            correct = choices == targets[batch_indices]
            minimum_scores[mode] = 1.01
            for threshold in np.arange(.25, 1, .025):
                selected = scores >= threshold
                if selected.sum() >= 20 and correct[selected].float().mean() >= .80:
                    minimum_scores[mode] = float(threshold)
                    break
    best["minimum_scores"] = minimum_scores
    report = dict(classes=len(labels), training_clips=len(indices["train"]), epoch=best["epoch"],
                  validation=best["validation"], test=evaluate("test"), gpu=torch.cuda.get_device_name(),
                  minimum_scores=minimum_scores)
    with torch.no_grad():
        test = indices["test"]
        video, still = zip(*(model(features[b]) for b in test.split(64)))
        report["accepted_test_predictions"] = {}
        for mode, logits in (("video", torch.cat(video)), ("image", torch.cat(still)[:, 4])):
            scores, choices = logits.softmax(1).max(1)
            accepted = scores >= minimum_scores[mode]
            report["accepted_test_predictions"][mode] = {
                "accepted": accepted.sum().item(),
                "correct": ((choices == targets[test]) & accepted).sum().item(),
                "total": len(test),
            }
    temporary = FULL_MODEL.with_suffix(".tmp")
    torch.save(best, temporary)
    temporary.replace(FULL_MODEL)
    FULL_MODEL.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


def load_full():
    state = torch.load(FULL_MODEL, map_location="cpu", weights_only=True)
    model = VideoCNN(len(state["labels"]))
    model.load_state_dict(state["weights"])
    model.minimum_scores = state["minimum_scores"]
    return model.to(cuda_device()).eval(), state["labels"]


@torch.inference_mode()
def predict_full(path, model, labels):
    image = path.suffix.lower().lstrip(".") in IMAGE_TYPES
    if image:
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unreadable image")
        rgb = cv2.cvtColor(cv2.resize(frame, (96,96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)[None]
    else:
        rgb = sample(path)
    features = model.encode(torch.from_numpy(rgb)[None].to(cuda_device()))
    video, still = model(features)
    scores = (still[:, 0] if image else video).softmax(1)[0]
    score, index = scores.max(0)
    return labels[index.item()], score.item()


if __name__ == "__main__":
    train()
