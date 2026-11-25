from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn


from signspeak.settings import ROOT, PILOT, CHECKPOINT, IMAGE_TYPES, VIDEO_TYPES


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the SignSpeak CUDA-enabled Python environment.")
    return torch.device("cuda")


class Recognizer(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(128, 96), nn.LayerNorm(96), nn.GELU(),
                                   nn.Dropout(0.15), nn.Linear(96, classes))
        self.temporal = nn.GRU(128, 96, batch_first=True, bidirectional=True)
        self.video = nn.Linear(192, classes)

    def forward(self, frames, image=False):
        if image:
            return self.image(frames)
        sequence, _ = self.temporal(frames)
        return self.video(sequence.mean(dim=1))


def extract(path, image):
    def encode(frame, detector):
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (max(1, round(width * .5)), max(1, round(height * .5))),
                             interpolation=cv2.INTER_AREA)
        result = detector.process(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        features = np.zeros(128, dtype=np.float32)
        for hand, side in zip(result.multi_hand_landmarks or [], result.multi_handedness or []):
            slot = int(side.classification[0].label == "Right")
            features[slot * 63:(slot + 1) * 63] = [
                value for point in hand.landmark for value in (point.x, point.y, point.z)
            ]
            features[126 + slot] = 1
        return features

    with mp.solutions.hands.Hands(static_image_mode=image, model_complexity=0) as detector:
        if image:
            frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode the image.")
            frames = [encode(frame, detector)]
        else:
            video = cv2.VideoCapture(str(path))
            frames = []
            try:
                for index in range(1801):
                    ok, frame = video.read()
                    if not ok:
                        break
                    if index == 1800:
                        raise ValueError("Use a shorter clip for each sign (up to 1,800 frames).")
                    if index % 3 == 0:
                        frames.append(encode(frame, detector))
            finally:
                video.release()
            if not frames:
                raise ValueError("Could not decode the video.")
            frames = np.stack(frames)[np.linspace(0, len(frames) - 1, 32, dtype=int)]
    features = np.asarray(frames)
    if not features[:, -2:].any():
        raise ValueError("No hands were detected. Use a clear view of the signer’s hands.")
    return features


def load_model(path=CHECKPOINT):
    device = cuda_device()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    labels = checkpoint["labels"]
    model = Recognizer(len(labels)).to(device)
    model.load_state_dict(checkpoint["weights"])
    return model.eval(), labels


@torch.inference_mode()
def predict(path, model, labels):
    image = path.suffix.lower().lstrip(".") in IMAGE_TYPES
    features = torch.from_numpy(extract(path, image)).to(next(model.parameters()).device)
    scores = model(features if image else features.unsqueeze(0), image=image).softmax(1)[0]
    score, index = scores.max(0)
    return labels[index.item()], score.item()
