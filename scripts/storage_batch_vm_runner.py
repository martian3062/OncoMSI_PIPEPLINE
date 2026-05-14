#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def normalize_stem(name: str) -> str:
    value = name.strip()
    if value.lower().endswith(".svs"):
        value = value[:-4]
    return value.upper()


def patient_barcode_from_slide(name: str) -> str:
    parts = normalize_stem(name).split("-")
    return "-".join(parts[:3]).upper()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def list_bucket_files(bucket_uri: str) -> list[dict[str, str]]:
    result = run_command(["gcloud", "storage", "ls", bucket_uri])
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or f"Unable to list bucket: {bucket_uri}")
    files: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        uri = line.strip()
        if not uri.endswith(".svs"):
            continue
        name = uri.rsplit("/", 1)[-1]
        stem = normalize_stem(name)
        files.append(
            {
                "uri": uri,
                "name": name,
                "stem": stem,
                "patient": patient_barcode_from_slide(name),
                "suffix": stem.split("-")[-1],
            }
        )
    return files


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def choose_batch(
    rows: list[dict[str, str]],
    bucket_files: list[dict[str, str]],
    first_n: int,
    preferred_exact_suffix: str,
    preferred_slide_pattern: str,
) -> list[dict[str, Any]]:
    exact_map = {row["stem"]: row for row in bucket_files}
    patient_map: dict[str, list[dict[str, str]]] = {}
    for item in bucket_files:
        patient_map.setdefault(item["patient"], []).append(item)

    selected: list[dict[str, Any]] = []
    seen_patients: set[str] = set()
    for row in rows:
        if len(selected) >= first_n:
            break
        patient = str(row.get("patient") or "").strip().upper()
        if not patient or patient in seen_patients:
            continue
        slide_stem = normalize_stem(str(row.get("slide") or ""))
        match = exact_map.get(slide_stem)
        if match is None:
            options = patient_map.get(patient, [])
            preferred = [item for item in options if item["suffix"] == preferred_exact_suffix]
            if not preferred:
                preferred = [item for item in options if preferred_slide_pattern in item["suffix"]]
            if not preferred:
                preferred = sorted(options, key=lambda item: item["name"])
            if preferred:
                match = preferred[0]
        if match is None:
            continue
        seen_patients.add(patient)
        selected.append(
            {
                "patient": patient,
                "msi_status": str(row.get("msi_status") or "").strip(),
                "bucket_name": match["name"],
                "bucket_uri": match["uri"],
            }
        )
    return selected


def download_slide(uri: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / uri.rsplit("/", 1)[-1]
    if target.exists():
        return
    gsutil_result = run_command(["gsutil", "cp", uri, str(target_dir)])
    if gsutil_result.returncode == 0:
        return
    gcloud_result = run_command(["gcloud", "storage", "cp", uri, str(target_dir)])
    if gcloud_result.returncode == 0:
        return
    raise RuntimeError(gsutil_result.stderr or gsutil_result.stdout or gcloud_result.stderr or gcloud_result.stdout or f"Failed to copy {uri}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_storage_manifest(project_root: Path, batch_name: str, rows: list[dict[str, Any]], storage_dir: Path) -> Path:
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        local_vm_path = storage_dir / row["bucket_name"]
        manifest_rows.append(
            {
                "patient": row["patient"],
                "msi_status": row["msi_status"],
                "bucket_name": row["bucket_name"],
                "bucket_uri": row["bucket_uri"],
                "local_vm_path": str(local_vm_path),
                "available_on_vm": local_vm_path.exists(),
                "source_group": batch_name,
            }
        )
    summary = {
        "requested_total": len(rows),
        "available_total": sum(1 for item in manifest_rows if item["available_on_vm"]),
        "msi_h_total": sum(1 for item in manifest_rows if item["msi_status"] == "MSI-H"),
        "mss_total": sum(1 for item in manifest_rows if item["msi_status"] == "MSS"),
        "note": f"Batch {batch_name} from matched_annotations_tcga3_vm.csv for staged VM inference over the 425-slide set.",
    }
    payload = {
        "title": f"Matched SVS batch {batch_name}",
        "summary": summary,
        "files": manifest_rows,
    }
    manifest_path = project_root / "runtime" / "storage_samples" / "storage_manifest.json"
    write_json(manifest_path, payload)
    return manifest_path


def post_json(url: str, form_payload: dict[str, str]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(form_payload).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def restart_local_django_server(project_root: Path, django_python: str, api_base: str) -> None:
    runserver_match = "manage.py runserver 0.0.0.0:8000"
    subprocess.run(["pkill", "-f", runserver_match], check=False)
    time.sleep(2)
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "django-web.log"
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            [django_python, "manage.py", "runserver", "0.0.0.0:8000", "--noreload"],
            cwd=str(project_root),
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    del process
    for _ in range(90):
        try:
            get_json(f"{api_base}/api/prediction-history/?limit=1")
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("Django backend did not come back after restart.")


def queue_and_wait(
    item: dict[str, Any],
    *,
    api_base: str,
    mode: str,
    poll_seconds: int,
    job_timeout_seconds: int,
) -> dict[str, Any]:
    queued = post_json(f"{api_base}/api/storage-samples/test/?mode={mode}", {"bucket_name": item["bucket_name"]})
    job_id = str(queued.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"Storage test did not return a job id for {item['bucket_name']}")
    started = time.time()
    while True:
        try:
            payload = get_json(f"{api_base}/api/predict-jobs/{job_id}/")
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and time.time() - started < max(job_timeout_seconds, 60):
                time.sleep(max(2, poll_seconds))
                continue
            raise
        status = str(payload.get("status") or "")
        if status in {"completed", "failed"}:
            return payload
        if time.time() - started > max(30, job_timeout_seconds):
            raise TimeoutError(f"Prediction job timed out for {item['bucket_name']} in mode={mode}. job_id={job_id}")
        time.sleep(max(2, poll_seconds))


def load_processed_results(results_jsonl: Path) -> set[str]:
    if not results_jsonl.exists():
        return set()
    processed: set[str] = set()
    for raw_line in results_jsonl.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        bucket_name = str(payload.get("bucket_name") or "").strip()
        if bucket_name:
            processed.add(bucket_name)
    return processed


def load_result_rows(results_jsonl: Path) -> list[dict[str, Any]]:
    if not results_jsonl.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in results_jsonl.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and score a matched SVS batch on the VM.")
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--bucket-uri", default="gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD")
    parser.add_argument("--first-n", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-name", default="batch-001")
    parser.add_argument("--mode", default="exact")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--preferred-exact-suffix", default="DX1")
    parser.add_argument("--preferred-slide-pattern", default="DX")
    parser.add_argument("--poll-seconds", type=int, default=8)
    parser.add_argument("--job-timeout-seconds", type=int, default=900)
    parser.add_argument("--fallback-mode", default="fast")
    parser.add_argument("--django-python", default="/home/pardeep/.venvs/pathology310-hybrid/bin/python")
    args = parser.parse_args()

    annotations_csv = Path(args.annotations_csv).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Missing annotations csv: {annotations_csv}")

    rows = load_rows(annotations_csv)
    bucket_files = list_bucket_files(args.bucket_uri)
    matched_rows = choose_batch(
        rows[args.offset :],
        bucket_files,
        args.first_n,
        args.preferred_exact_suffix,
        args.preferred_slide_pattern,
    )
    if not matched_rows:
        raise RuntimeError("No matched rows were selected for this batch.")

    batch_dir = project_root / "runtime" / "storage_batches" / args.batch_name
    storage_dir = project_root / "runtime" / "storage_samples" / args.batch_name
    write_json(batch_dir / "selected_rows.json", matched_rows)

    for index, item in enumerate(matched_rows, start=1):
        print(f"[download {index}/{len(matched_rows)}] {item['bucket_name']}", flush=True)
        download_slide(item["bucket_uri"], storage_dir)

    manifest_path = write_storage_manifest(project_root, args.batch_name, matched_rows, storage_dir)
    print(f"[manifest] {manifest_path}", flush=True)

    results_jsonl = batch_dir / "prediction_results.jsonl"
    processed_bucket_names = load_processed_results(results_jsonl)
    for index, item in enumerate(matched_rows, start=1):
        if item["bucket_name"] in processed_bucket_names:
            print(f"[skip {index}/{len(matched_rows)}] {item['bucket_name']}", flush=True)
            continue
        print(f"[predict {index}/{len(matched_rows)}] {item['bucket_name']}", flush=True)
        effective_mode = args.mode
        try:
            payload = queue_and_wait(
                item,
                api_base=args.api_base,
                mode=effective_mode,
                poll_seconds=args.poll_seconds,
                job_timeout_seconds=args.job_timeout_seconds,
            )
        except TimeoutError as exc:
            print(f"[timeout {index}/{len(matched_rows)}] {item['bucket_name']} mode={effective_mode} detail={exc}", flush=True)
            if args.fallback_mode and args.fallback_mode != effective_mode:
                restart_local_django_server(project_root, args.django_python, args.api_base)
                effective_mode = args.fallback_mode
                print(f"[fallback {index}/{len(matched_rows)}] {item['bucket_name']} mode={effective_mode}", flush=True)
                payload = queue_and_wait(
                    item,
                    api_base=args.api_base,
                    mode=effective_mode,
                    poll_seconds=args.poll_seconds,
                    job_timeout_seconds=max(300, args.job_timeout_seconds // 2),
                )
            else:
                row = {
                    "patient": item["patient"],
                    "msi_status": item["msi_status"],
                    "bucket_name": item["bucket_name"],
                    "job_id": "",
                    "status": "timeout",
                    "prediction": None,
                    "probability": None,
                    "confidence_level": None,
                    "tile_count": None,
                    "mode_used": effective_mode,
                }
                append_jsonl(results_jsonl, row)
                print(json.dumps(row), flush=True)
                continue
        result = payload.get("result") or {}
        row = {
            "patient": item["patient"],
            "msi_status": item["msi_status"],
            "bucket_name": item["bucket_name"],
            "job_id": payload.get("job_id"),
            "status": payload.get("status"),
            "prediction": result.get("label"),
            "probability": result.get("probability"),
            "confidence_level": result.get("confidence_level"),
            "tile_count": result.get("tile_count"),
            "mode_used": effective_mode,
        }
        append_jsonl(results_jsonl, row)
        print(json.dumps(row), flush=True)

    summary = load_result_rows(results_jsonl)
    write_json(
        batch_dir / "summary.json",
        {
            "batch_name": args.batch_name,
            "count": len(summary),
            "results": summary,
        },
    )
    print(f"[done] batch={args.batch_name} slides={len(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
