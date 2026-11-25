"""Run native inference outside the Streamlit server process."""
import json
from pathlib import Path
import sys

from signspeak.full_model import load_full, predict_full
from signspeak.recognition import load_model, predict
from signspeak.settings import FULL_MODEL, IMAGE_TYPES


def main(paths):
    full = FULL_MODEL.is_file()
    model, labels = load_full() if full else load_model()
    infer = predict_full if full else predict
    results = []
    for path in paths:
        try:
            label, score = infer(path, model, labels)
            mode = "image" if path.suffix.lower().lstrip(".") in IMAGE_TYPES else "video"
            results.append({"sign": label.partition(". ")[2] or label, "score": score,
                            "uncertain": full and score < model.minimum_scores[mode]})
        except (ValueError, OSError, RuntimeError) as error:
            results.append({"error": str(error)})
    print(json.dumps(results))


if __name__ == "__main__":
    main(map(Path, sys.argv[1:]))
