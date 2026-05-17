from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from visualisation.demo_app.inference import load_model, run_inference
from visualisation.demo_app.landcover import (
    discover_checkpoints,
    index_samples,
    load_split_samples,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

_MODEL_CACHE: dict[tuple[str, str], object] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_ROOT = (
    Path("/Users/jan/Desktop/Programowanie/pwr/semestr_5/nn-2/data/landcover.ai.v1")
).resolve()
IMAGE_DIR = DATA_ROOT / "output"
TRAIN_SPLIT = DATA_ROOT / "train.txt"


def _default_checkpoint(checkpoints: list) -> str:
    if not checkpoints:
        return ""
    return checkpoints[0].path


def _get_loaded_model(checkpoint_path: str, device: str):
    cache_key = (checkpoint_path, device)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_model(checkpoint_path, requested_device=device)
    return _MODEL_CACHE[cache_key]


@app.route("/", methods=["GET", "POST"])
def index():
    checkpoints = discover_checkpoints(CHECKPOINT_DIR)
    samples = load_split_samples(IMAGE_DIR, TRAIN_SPLIT)
    sample_index = index_samples(samples)

    context = {
        "checkpoint_options": checkpoints,
        "sample_options": samples,
        "sample_count": len(samples),
        "default_checkpoint": _default_checkpoint(checkpoints),
        "selected_device": "auto",
        "selected_checkpoint": "",
        "selected_sample_id": "",
        "result": None,
        "error": None,
    }

    if request.method == "POST":
        checkpoint_path = request.form.get("checkpoint_path", "").strip()
        requested_device = request.form.get("device", "auto")
        sample_id = request.form.get("sample_id", "").strip()
        image_file = request.files.get("image_file")
        mask_file = request.files.get("mask_file")

        context["selected_checkpoint"] = checkpoint_path
        context["selected_device"] = requested_device
        context["selected_sample_id"] = sample_id

        try:
            if not checkpoint_path:
                raise ValueError("Choose a checkpoint from the picker.")

            image_bytes: bytes
            mask_bytes: bytes | None
            source_label = None

            if sample_id:
                sample = sample_index.get(sample_id)
                if sample is None:
                    raise ValueError(f"Unknown dataset sample: {sample_id}")
                if not sample.image_path.is_file() or not sample.mask_path.is_file():
                    raise FileNotFoundError(
                        f"Dataset files are missing for sample {sample_id}."
                    )
                image_bytes = sample.image_path.read_bytes()
                mask_bytes = sample.mask_path.read_bytes()
                source_label = sample.sample_id
            else:
                if image_file is None or image_file.filename == "":
                    raise ValueError(
                        "Choose a training sample or upload a custom image before running inference."
                    )
                image_bytes = image_file.read()
                mask_bytes = (
                    mask_file.read() if mask_file and mask_file.filename else None
                )

            loaded_model = _get_loaded_model(checkpoint_path, requested_device)
            context["result"] = run_inference(
                loaded_model=loaded_model,
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                source_label=source_label,
            )
        except Exception as exc:  # noqa: BLE001
            context["error"] = str(exc)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
