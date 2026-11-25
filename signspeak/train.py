import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from signspeak.recognition import CHECKPOINT, PILOT, Recognizer, cuda_device


def main():
    parser = argparse.ArgumentParser(description="Train SignSpeak on the local pilot landmark cache.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--features", type=Path, default=PILOT / "pilot_hand_landmarks.npz")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    device = cuda_device()
    torch.manual_seed(42)
    print(f"Training on {torch.cuda.get_device_name(device)}", flush=True)
    with np.load(args.features, allow_pickle=False) as data:
        frames = torch.as_tensor(data["features"], dtype=torch.float32, device=device)
        labels, splits = data["labels"], data["splits"]
    classes = sorted(set(labels[splits == "train"]))
    if not classes or set(labels) != set(classes) or any(
        not (splits == name).any() for name in ("train", "val", "test")
    ):
        raise ValueError("Expected train, validation and test clips with known labels.")
    classes = [str(label) for label in classes]
    targets = torch.tensor([classes.index(label) for label in labels], device=device)
    indices = {
        split: torch.tensor(np.flatnonzero(splits == split), device=device)
        for split in ("train", "val", "test")
    }
    model = Recognizer(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)

    def image_examples(batch):
        clip, frame = frames[batch, :, -2:].any(dim=-1).nonzero(as_tuple=True)
        return frames[batch[clip], frame], targets[batch[clip]]

    @torch.inference_mode()
    def evaluate(batch):
        model.eval()
        images, image_targets = image_examples(batch)
        video_logits = model(frames[batch])
        image_logits = model(images, image=True)
        return {
            "loss": (F.cross_entropy(video_logits, targets[batch])
                     + F.cross_entropy(image_logits, image_targets)).item(),
            "video_accuracy": (video_logits.argmax(1) == targets[batch]).float().mean().item(),
            "image_frame_accuracy": (image_logits.argmax(1) == image_targets).float().mean().item(),
            "clips": len(batch),
            "image_frames": len(images),
        }

    best_loss = float("inf")
    best = None
    CHECKPOINT.parent.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train = indices["train"]
        for batch in train[torch.randperm(len(train), device=device)].split(16):
            images, image_targets = image_examples(batch)
            loss = F.cross_entropy(model(frames[batch]), targets[batch])
            if len(images):
                loss = loss + F.cross_entropy(model(images, image=True), image_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation = evaluate(indices["val"])
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best = {"epoch": epoch, "validation": validation, "labels": classes,
                    "weights": {key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()}}
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch} | validation loss {validation['loss']:.3f}", flush=True)
    if best is None:
        raise RuntimeError("Training did not produce a finite validation loss.")
    model.load_state_dict(best["weights"])
    report = {"epoch": best["epoch"], "validation": best["validation"],
              "test": evaluate(indices["test"]), "device": torch.cuda.get_device_name(device)}
    temporary = CHECKPOINT.with_suffix(".tmp")
    torch.save(best, temporary)
    temporary.replace(CHECKPOINT)
    CHECKPOINT.with_name("metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
