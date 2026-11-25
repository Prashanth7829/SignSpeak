from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PILOT_DIR = PROJECT_DIR / "dataset" / "include50_pilot"
SEQUENCE_LENGTH = 32
FRAME_STRIDE = 3
COORDINATES_PER_HAND = 21 * 3
FEATURE_COUNT = 2 * COORDINATES_PER_HAND + 2
HAND_SLOTS = {"Left": 0, "Right": 1}


def extract_sequence(video_path):
    video = cv2.VideoCapture(str(video_path))
    frames = []
    frame_index = 0

    try:
        with mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as detector:
            while True:
                decoded, frame = video.read()
                if not decoded:
                    break

                if frame_index % FRAME_STRIDE == 0:
                    resized = cv2.resize(
                        frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                    )
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    result = detector.process(rgb)
                    features = np.zeros(FEATURE_COUNT, dtype=np.float32)

                    for hand, handedness in zip(
                        result.multi_hand_landmarks or [],
                        result.multi_handedness or [],
                    ):
                        slot = HAND_SLOTS[handedness.classification[0].label]
                        start = slot * COORDINATES_PER_HAND
                        features[start : start + COORDINATES_PER_HAND] = [
                            coordinate
                            for point in hand.landmark
                            for coordinate in (point.x, point.y, point.z)
                        ]
                        features[2 * COORDINATES_PER_HAND + slot] = 1.0

                    frames.append(features)

                frame_index += 1
    finally:
        video.release()

    if not frames:
        raise RuntimeError(f"Could not decode any frames from {video_path}")

    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
    return np.stack(frames)[indices]


def main():
    manifest = pd.read_csv(PILOT_DIR / "pilot_manifest.csv")
    sequences = []
    no_hand_paths = []

    for index, clip in enumerate(manifest.itertuples(index=False), start=1):
        sequence = extract_sequence(PILOT_DIR / clip.video_path)
        sequences.append(sequence)

        if not sequence[:, -2:].any():
            no_hand_paths.append(clip.video_path)

        print(f"[{index}/{len(manifest)}] {clip.video_path}", flush=True)

    features = np.stack(sequences)
    output_path = PILOT_DIR / "pilot_hand_landmarks.npz"
    np.savez_compressed(
        output_path,
        features=features,
        labels=manifest["label"].to_numpy(dtype=str),
        splits=manifest["split"].to_numpy(dtype=str),
        video_paths=manifest["video_path"].to_numpy(dtype=str),
    )

    print(f"Saved: {output_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Clips with no hands detected in the saved frames: {len(no_hand_paths)}")
    for path in no_hand_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()