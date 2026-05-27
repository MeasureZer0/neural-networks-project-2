from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask.typing import ResponseReturnValue

from visualisation.demo_app.inference import LoadedModel, load_model, run_inference
from visualisation.demo_app.landcover import (
    DatasetSample,
    discover_checkpoints,
    index_samples,
    load_split_samples,
)

_MODEL_CACHE: dict[tuple[str, str], LoadedModel] = {}
_JOBS: dict[str, dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()
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


def _checkpoint_label(checkpoint_path: str) -> str:
    return Path(checkpoint_path).stem.replace("_", " ")


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
        "loading_job": None,
        "error": None,
    }


def normalize_checkpoint_paths(checkpoint_paths: list[str]) -> list[str]:
    unique_paths: list[str] = []
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path and checkpoint_path not in unique_paths:
            unique_paths.append(checkpoint_path)
    return unique_paths


def extract_request_payload(
    sample_id: str,
    samples: list[DatasetSample],
    image_bytes: bytes | None,
    mask_bytes: bytes | None,
) -> tuple[bytes, bytes | None, str | None]:
    sample_index = index_samples(samples)

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

    if image_bytes is None:
        raise ValueError(
            "Choose a training sample or upload a custom image before running inference."
        )

    return image_bytes, mask_bytes, None


def resolve_benchmark_samples(
    samples: list[DatasetSample],
    requested_count: int,
) -> list[DatasetSample]:
    if requested_count <= 0:
        raise ValueError("Benchmark image count must be at least 1.")
    return samples[: min(requested_count, len(samples))]


def _mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def create_job_record(
    *,
    selected_mode: str,
    selected_device: str,
    selected_checkpoint_paths: list[str],
    selected_sample_id: str,
    selected_benchmark_count: int,
) -> dict[str, Any]:
    return {
        "status": "pending",
        "selected_mode": selected_mode,
        "selected_device": selected_device,
        "selected_checkpoint_paths": selected_checkpoint_paths or [""],
        "selected_sample_id": selected_sample_id,
        "selected_benchmark_count": selected_benchmark_count,
        "completed_steps": 0,
        "total_steps": 1,
        "current_model_label": "",
        "current_sample_label": "",
        "current_message": "Queued.",
        "started_at": None,
        "result": None,
        "results": [],
        "benchmark": None,
        "error": None,
    }


def update_job(job_id: str, **updates: object) -> None:
    with _JOB_LOCK:
        job = _JOBS[job_id]
        job.update(updates)


def load_job(job_id: str) -> dict[str, Any] | None:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None


def progress_payload(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    total_steps = max(int(job.get("total_steps", 1)), 1)
    completed_steps = max(0, min(int(job.get("completed_steps", 0)), total_steps))
    progress_pct = int(round((completed_steps / total_steps) * 100))
    started_at = job.get("started_at")
    elapsed_seconds = 0.0
    eta_seconds: float | None = None
    estimated_end: str | None = None
    if isinstance(started_at, str):
        started_at_dt = datetime.fromisoformat(started_at)
        elapsed_seconds = max(
            0.0,
            (datetime.now(UTC) - started_at_dt).total_seconds(),
        )
        if completed_steps > 0 and completed_steps < total_steps:
            seconds_per_step = elapsed_seconds / completed_steps
            eta_seconds = max(0.0, seconds_per_step * (total_steps - completed_steps))
            estimated_end = (
                datetime.now(UTC) + timedelta(seconds=eta_seconds)
            ).astimezone().strftime("%H:%M:%S")
    return {
        "job_id": job_id,
        "status": job["status"],
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "progress_pct": progress_pct,
        "current_model_label": job.get("current_model_label") or "",
        "current_sample_label": job.get("current_sample_label") or "",
        "current_message": job.get("current_message") or "",
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "estimated_end": estimated_end,
        "redirect_url": url_for("job_page", job_id=job_id),
    }


def build_context_from_job(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    context = build_base_context()
    context["selected_device"] = job["selected_device"]
    context["selected_checkpoint_paths"] = job["selected_checkpoint_paths"]
    context["selected_sample_id"] = job["selected_sample_id"]
    context["selected_mode"] = job["selected_mode"]
    context["selected_benchmark_count"] = job["selected_benchmark_count"]
    context["result"] = job.get("result")
    context["results"] = job.get("results", [])
    context["benchmark"] = job.get("benchmark")
    context["error"] = job.get("error")

    if job["status"] in {"pending", "running"}:
        context["loading_job"] = progress_payload(job_id, job)

    return context


def build_model_report(
    loaded_model: LoadedModel,
    sample_reports: list[dict[str, Any]],
) -> dict[str, Any]:
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
    p95_index = max(
        0,
        min(
            len(latency_values_sorted) - 1,
            int(len(latency_values_sorted) * 0.95) - 1,
        ),
    )

    return {
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


def run_interactive_job(
    job_id: str,
    checkpoint_paths: list[str],
    requested_device: str,
    image_bytes: bytes | None,
    mask_bytes: bytes | None,
    sample_id: str,
    samples: list[DatasetSample],
) -> None:
    unique_paths = normalize_checkpoint_paths(checkpoint_paths)
    if not unique_paths:
        raise ValueError("Choose a checkpoint from the picker.")

    resolved_image_bytes, resolved_mask_bytes, source_label = extract_request_payload(
        sample_id=sample_id,
        samples=samples,
        image_bytes=image_bytes,
        mask_bytes=mask_bytes,
    )

    update_job(
        job_id,
        status="running",
        total_steps=len(unique_paths),
        current_message="Preparing interactive comparison.",
        started_at=datetime.now(UTC).isoformat(),
    )

    results: list[dict[str, Any]] = []
    for index, checkpoint_path in enumerate(unique_paths, start=1):
        update_job(
            job_id,
            current_model_label=_checkpoint_label(checkpoint_path),
            current_sample_label=source_label or "custom upload",
            current_message=f"Running model {index} of {len(unique_paths)}.",
        )
        loaded_model = _get_loaded_model(checkpoint_path, requested_device)
        results.append(
            run_inference(
                loaded_model=loaded_model,
                image_bytes=resolved_image_bytes,
                mask_bytes=resolved_mask_bytes,
                source_label=source_label,
            )
        )
        update_job(job_id, completed_steps=index)

    update_job(
        job_id,
        status="completed",
        result=results[0],
        results=results,
        current_message="Interactive comparison finished.",
    )


def run_benchmark_job(
    job_id: str,
    checkpoint_paths: list[str],
    requested_device: str,
    samples: list[DatasetSample],
    benchmark_count: int,
) -> None:
    unique_paths = normalize_checkpoint_paths(checkpoint_paths)
    if not unique_paths:
        raise ValueError("Choose at least one checkpoint for the benchmark.")
    if not samples:
        raise ValueError("No test samples are available for benchmarking.")

    benchmark_samples = resolve_benchmark_samples(samples, benchmark_count)
    total_steps = len(unique_paths) * len(benchmark_samples)
    update_job(
        job_id,
        status="running",
        total_steps=total_steps,
        current_message="Preparing benchmark.",
        started_at=datetime.now(UTC).isoformat(),
    )

    model_reports: list[dict[str, Any]] = []
    completed_steps = 0

    for model_index, checkpoint_path in enumerate(unique_paths, start=1):
        loaded_model = _get_loaded_model(checkpoint_path, requested_device)
        sample_reports: list[dict[str, Any]] = []
        model_label = _checkpoint_label(checkpoint_path)

        for sample_index, sample in enumerate(benchmark_samples, start=1):
            update_job(
                job_id,
                current_model_label=model_label,
                current_sample_label=sample.sample_id,
                current_message=(
                    f"Benchmarking model {model_index} of {len(unique_paths)}, "
                    f"image {sample_index} of {len(benchmark_samples)}."
                ),
            )
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
            completed_steps += 1
            update_job(job_id, completed_steps=completed_steps)

        model_reports.append(build_model_report(loaded_model, sample_reports))

    model_reports.sort(key=lambda report: report["mean_iou"], reverse=True)
    update_job(
        job_id,
        status="completed",
        benchmark={
            "split_name": "test",
            "requested_count": benchmark_count,
            "actual_count": len(benchmark_samples),
            "available_count": len(samples),
            "models": model_reports,
        },
        current_message="Benchmark finished.",
    )


def start_job_thread(
    job_id: str,
    target: Callable[..., None],
    *args: object,
) -> None:
    def runner() -> None:
        try:
            target(job_id, *args)
        except Exception as exc:  # noqa: BLE001
            update_job(
                job_id,
                status="failed",
                error=str(exc),
                current_message="Run failed.",
            )

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    @app.route("/", methods=["GET", "POST"])
    def index() -> ResponseReturnValue:
        if request.method == "POST":
            return redirect(url_for("create_job"), code=307)
        return render_template("index.html", **build_base_context())

    @app.route("/jobs", methods=["GET", "POST"])
    def create_job() -> ResponseReturnValue:
        if request.method == "GET":
            return redirect(url_for("index"))

        context = build_base_context()
        samples = context["sample_options"]
        checkpoint_paths = [
            value.strip()
            for value in request.form.getlist("checkpoint_paths")
            if value.strip()
        ]
        requested_device = request.form.get("device", "auto")
        sample_id = request.form.get("sample_id", "").strip()
        selected_mode = request.form.get("mode", "interactive").strip() or "interactive"
        benchmark_count_raw = request.form.get("benchmark_count", "").strip()

        try:
            benchmark_count = int(
                benchmark_count_raw or context["selected_benchmark_count"] or 0
            )
        except ValueError:
            benchmark_count = 0

        image_file = request.files.get("image_file")
        mask_file = request.files.get("mask_file")
        image_bytes = image_file.read() if image_file and image_file.filename else None
        mask_bytes = mask_file.read() if mask_file and mask_file.filename else None

        job_id = uuid.uuid4().hex
        job = create_job_record(
            selected_mode=selected_mode,
            selected_device=requested_device,
            selected_checkpoint_paths=checkpoint_paths,
            selected_sample_id=sample_id,
            selected_benchmark_count=benchmark_count,
        )
        with _JOB_LOCK:
            _JOBS[job_id] = job

        if selected_mode == "benchmark":
            start_job_thread(
                job_id,
                run_benchmark_job,
                checkpoint_paths,
                requested_device,
                samples,
                benchmark_count,
            )
        else:
            start_job_thread(
                job_id,
                run_interactive_job,
                checkpoint_paths,
                requested_device,
                image_bytes,
                mask_bytes,
                sample_id,
                samples,
            )

        redirect_url = url_for("job_page", job_id=job_id)
        if request.headers.get("X-Requested-With") == "fetch":
            return jsonify({"job_id": job_id, "redirect_url": redirect_url})
        return redirect(redirect_url)

    @app.route("/jobs/<job_id>", methods=["GET"])
    def job_page(job_id: str) -> str:
        job = load_job(job_id)
        if job is None:
            context = build_base_context()
            context["error"] = f"Unknown job: {job_id}"
            return render_template("index.html", **context), 404
        return render_template("index.html", **build_context_from_job(job_id, job))

    @app.route("/api/jobs/<job_id>", methods=["GET"])
    def job_status(job_id: str) -> ResponseReturnValue:
        job = load_job(job_id)
        if job is None:
            return jsonify({"error": f"Unknown job: {job_id}"}), 404

        payload = progress_payload(job_id, job)
        if job["status"] == "failed":
            payload["error"] = job.get("error")
        return jsonify(payload)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
