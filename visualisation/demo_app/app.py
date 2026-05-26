from __future__ import annotations

import os
from pathlib import Path
from statistics import mean
from typing import Any

from flask import Flask, render_template, request

from visualisation.demo_app.inference import LoadedModel, load_model, run_inference
from visualisation.demo_app.landcover import (
    DatasetSample,
    discover_checkpoints,
    index_samples,
    load_split_samples,
)

_MODEL_CACHE: dict[tuple[str, str], LoadedModel] = {}
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
    default_checkpoint = _default_checkpoint(checkpoints)

    return {
        "checkpoint_options": checkpoints,
        "sample_options": samples,
        "sample_count": len(samples),
        "dataset_root": str(resolve_data_root()),
        "selected_device": "auto",
        "selected_checkpoint_paths": [default_checkpoint] if default_checkpoint else [""],
        "selected_sample_id": "",
        "selected_mode": "interactive",
        "selected_benchmark_count": min(len(samples), 25) if samples else 0,
        "result": None,
        "results": [],
        "benchmark": None,
        "error": None,
    }


def extract_request_payload(
    sample_id: str,
    samples: list[DatasetSample],
) -> tuple[bytes, bytes | None, str | None]:
    sample_index = index_samples(samples)
    image_file = request.files.get("image_file")
    mask_file = request.files.get("mask_file")

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


def resolve_benchmark_samples(
    samples: list[DatasetSample],
    requested_count: int,
) -> list[DatasetSample]:
    if requested_count <= 0:
        raise ValueError("Benchmark image count must be at least 1.")
    return samples[: min(requested_count, len(samples))]


def _mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def run_benchmark(
    checkpoint_paths: list[str],
    requested_device: str,
    samples: list[DatasetSample],
    benchmark_count: int,
) -> dict[str, Any]:
    unique_paths: list[str] = []
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path and checkpoint_path not in unique_paths:
            unique_paths.append(checkpoint_path)

    if not unique_paths:
        raise ValueError("Choose at least one checkpoint for the benchmark.")
    if not samples:
        raise ValueError("No test samples are available for benchmarking.")

    benchmark_samples = resolve_benchmark_samples(samples, benchmark_count)
    model_reports: list[dict[str, Any]] = []

    for checkpoint_path in unique_paths:
        loaded_model = _get_loaded_model(checkpoint_path, requested_device)
        sample_reports: list[dict[str, Any]] = []

        for sample in benchmark_samples:
            result = run_inference(
                loaded_model=loaded_model,
                image_bytes=sample.image_path.read_bytes(),
                mask_bytes=sample.mask_path.read_bytes(),
                source_label=sample.sample_id,
                render_images=False,
            )
            sample_reports.append(
                {
                    "sample_id": sample.sample_id,
                    "mean_iou": result["metrics"]["mean_iou"],
                    "mean_dice": result["metrics"]["mean_dice"],
                    "pixel_accuracy": result["metrics"]["pixel_accuracy"],
                    "inference_ms": result["inference_ms"],
                    "mean_confidence": result["summary"]["mean_confidence"],
                    "low_confidence_pct": result["summary"]["low_confidence_pct"],
                    "error_rate_pct": result["diagnostics"]["error_rate_pct"] or 0.0,
                    "per_class": result["metrics"]["per_class"],
                }
            )

        per_class_rows: list[dict[str, Any]] = []
        class_names = [row["name"] for row in sample_reports[0]["per_class"]]
        for class_index, class_name in enumerate(class_names):
            per_class_rows.append(
                {
                    "name": class_name,
                    "iou": _mean_or_zero(
                        [
                            sample_report["per_class"][class_index]["iou"]
                            for sample_report in sample_reports
                        ]
                    ),
                    "dice": _mean_or_zero(
                        [
                            sample_report["per_class"][class_index]["dice"]
                            for sample_report in sample_reports
                        ]
                    ),
                }
            )

        ranked_by_iou = sorted(
            sample_reports,
            key=lambda sample_report: sample_report["mean_iou"],
        )
        latency_values = [sample_report["inference_ms"] for sample_report in sample_reports]
        latency_values_sorted = sorted(latency_values)
        p95_index = max(0, min(len(latency_values_sorted) - 1, int(len(latency_values_sorted) * 0.95) - 1))

        model_reports.append(
            {
                "checkpoint": str(loaded_model.checkpoint_path),
                "config_name": type(loaded_model.config).__name__,
                "model_name": getattr(loaded_model.config, "model", "unknown"),
                "device": loaded_model.device,
                "images_benchmarked": len(sample_reports),
                "mean_iou": _mean_or_zero(
                    [sample_report["mean_iou"] for sample_report in sample_reports]
                ),
                "mean_dice": _mean_or_zero(
                    [sample_report["mean_dice"] for sample_report in sample_reports]
                ),
                "pixel_accuracy": _mean_or_zero(
                    [sample_report["pixel_accuracy"] for sample_report in sample_reports]
                ),
                "mean_confidence": _mean_or_zero(
                    [sample_report["mean_confidence"] for sample_report in sample_reports]
                ),
                "low_confidence_pct": _mean_or_zero(
                    [sample_report["low_confidence_pct"] for sample_report in sample_reports]
                ),
                "error_rate_pct": _mean_or_zero(
                    [sample_report["error_rate_pct"] for sample_report in sample_reports]
                ),
                "latency_ms_mean": _mean_or_zero(latency_values),
                "latency_ms_median": latency_values_sorted[len(latency_values_sorted) // 2],
                "latency_ms_p95": latency_values_sorted[p95_index],
                "per_class": per_class_rows,
                "best_samples": list(reversed(ranked_by_iou[-5:])),
                "worst_samples": ranked_by_iou[:5],
            }
        )

    model_reports.sort(key=lambda report: report["mean_iou"], reverse=True)
    return {
        "split_name": "test",
        "requested_count": benchmark_count,
        "actual_count": len(benchmark_samples),
        "available_count": len(samples),
        "models": model_reports,
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        context = build_base_context()
        samples = context["sample_options"]

        if request.method == "POST":
            checkpoint_paths = [
                value.strip()
                for value in request.form.getlist("checkpoint_paths")
                if value.strip()
            ]
            requested_device = request.form.get("device", "auto")
            sample_id = request.form.get("sample_id", "").strip()
            selected_mode = request.form.get("mode", "interactive").strip() or "interactive"
            benchmark_count_raw = request.form.get("benchmark_count", "").strip()
            benchmark_count = int(benchmark_count_raw or context["selected_benchmark_count"] or 0)

            context["selected_checkpoint_paths"] = checkpoint_paths or [""]
            context["selected_device"] = requested_device
            context["selected_sample_id"] = sample_id
            context["selected_mode"] = selected_mode
            context["selected_benchmark_count"] = benchmark_count

            try:
                if selected_mode == "benchmark":
                    context["benchmark"] = run_benchmark(
                        checkpoint_paths=checkpoint_paths,
                        requested_device=requested_device,
                        samples=samples,
                        benchmark_count=benchmark_count,
                    )
                else:
                    results = run_comparison(
                        checkpoint_paths=checkpoint_paths,
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
