from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request

from visualisation.demo_app.inference import LoadedModel, load_model, run_inference
from visualisation.demo_app.landcover import (
    DatasetSample,
    discover_checkpoints,
    index_samples,
    load_split_samples,
)

_MODEL_CACHE: dict[tuple[str, str], object] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def resolve_data_root() -> Path:
    configured = os.environ.get("LANDCOVER_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (PROJECT_ROOT / "data" / "landcover.ai.v1").resolve()


def dataset_paths() -> tuple[Path, Path]:
    data_root = resolve_data_root()
    return data_root / "output", data_root / "test.txt"


def _default_checkpoint(checkpoints: list) -> str:
    if not checkpoints:
        return ""
    return checkpoints[0].path


def _get_loaded_model(checkpoint_path: str, device: str) -> LoadedModel:
    cache_key = (checkpoint_path, device)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_model(checkpoint_path, requested_device=device)
    return _MODEL_CACHE[cache_key]


def build_base_context() -> dict[str, Any]:
    image_dir, test_split = dataset_paths()
    checkpoints = discover_checkpoints(CHECKPOINT_DIR)
    samples = load_split_samples(image_dir, test_split)

    return {
        "checkpoint_options": checkpoints,
        "sample_options": samples,
        "sample_count": len(samples),
        "default_checkpoint": _default_checkpoint(checkpoints),
        "dataset_root": str(resolve_data_root()),
        "selected_device": "auto",
        "selected_checkpoint": "",
        "selected_compare_checkpoint": "",
        "selected_sample_id": "",
        "result": None,
        "results": [],
        "error": None,
    }

def extract_request_payload(
    checkpoint_path: str,
    sample_id: str,
    samples: list[DatasetSample],
) -> tuple[bytes, bytes | None, str | None]:
    sample_index = index_samples(samples)
    image_file = request.files.get("image_file")
    mask_file = request.files.get("mask_file")

    if not checkpoint_path:
        raise ValueError("Choose a checkpoint from the picker.")

    if sample_id:
        sample = sample_index.get(sample_id)
        if sample is None:
            raise ValueError(f"Unknown dataset sample: {sample_id}")
        if not sample.image_path.is_file() or not sample.mask_path.is_file():
            raise FileNotFoundError(
                f"Dataset files are missing for sample {sample_id}."
            )
        return (
            sample.image_path.read_bytes(),
            sample.mask_path.read_bytes(),
            sample.sample_id,
        )

    if image_file is None or image_file.filename == "":
        raise ValueError(
            "Choose a training sample or upload a custom image before running inference."
        )

    mask_bytes = mask_file.read() if mask_file and mask_file.filename else None
    return image_file.read(), mask_bytes, None


def run_request(
    checkpoint_path: str,
    requested_device: str,
    sample_id: str,
    samples: list[DatasetSample],
) -> dict[str, Any]:
    image_bytes, mask_bytes, source_label = extract_request_payload(
        checkpoint_path=checkpoint_path,
        sample_id=sample_id,
        samples=samples,
    )
    loaded_model = _get_loaded_model(checkpoint_path, requested_device)
    return run_inference(
        loaded_model=loaded_model,
        image_bytes=image_bytes,
        mask_bytes=mask_bytes,
        source_label=source_label,
    )


def run_comparison(
    checkpoint_paths: list[str],
    requested_device: str,
    sample_id: str,
    samples: list[DatasetSample],
) -> list[dict[str, Any]]:
    unique_paths: list[str] = []
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path and checkpoint_path not in unique_paths:
            unique_paths.append(checkpoint_path)

    if not unique_paths:
        raise ValueError("Choose a checkpoint from the picker.")

    image_bytes, mask_bytes, source_label = extract_request_payload(
        checkpoint_path=unique_paths[0],
        sample_id=sample_id,
        samples=samples,
    )

    results: list[dict[str, Any]] = []
    for checkpoint_path in unique_paths:
        loaded_model = _get_loaded_model(checkpoint_path, requested_device)
        results.append(
            run_inference(
                loaded_model=loaded_model,
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                source_label=source_label,
            )
        )
    return results


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        context = build_base_context()
        samples = context["sample_options"]

        if request.method == "POST":
            checkpoint_path = request.form.get("checkpoint_path", "").strip()
            compare_checkpoint_path = request.form.get(
                "compare_checkpoint_path", ""
            ).strip()
            requested_device = request.form.get("device", "auto")
            sample_id = request.form.get("sample_id", "").strip()

            context["selected_checkpoint"] = checkpoint_path
            context["selected_compare_checkpoint"] = compare_checkpoint_path
            context["selected_device"] = requested_device
            context["selected_sample_id"] = sample_id

            try:
                results = run_comparison(
                    checkpoint_paths=[checkpoint_path, compare_checkpoint_path],
                    requested_device=requested_device,
                    sample_id=sample_id,
                    samples=samples,
                )
                context["results"] = results
                context["result"] = results[0]
            except Exception as exc:  # noqa: BLE001
                context["error"] = str(exc)

        return render_template("index.html", **context)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
