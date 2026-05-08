#!/usr/bin/env python3
"""Check overlap between a training cohort and external label files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prefix12(value: str) -> str:
    text = (value or "").strip()
    return text[:12] if text else ""


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, str]], patient_field: str = "patient_id") -> dict[str, object]:
    patient_ids = {prefix12(row.get(patient_field, "")) for row in rows}
    patient_ids.discard("")
    return {
        "rows": len(rows),
        "patients": len(patient_ids),
        "label_counts": dict(Counter(row.get("label", "") for row in rows if row.get("label"))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-summary",
        default="ten/run-8635c038adcc/final_summary.json",
        help="final_summary.json for the exact training run",
    )
    parser.add_argument("--pub", default="runtime/annotations/coadread_tcga_pub_annotations.csv")
    parser.add_argument(
        "--pancan",
        default="runtime/annotations/coadread_tcga_pan_can_atlas_2018_annotations.csv",
    )
    parser.add_argument("--cptac", default="runtime/annotations/cptac_coad_annotations.csv")
    parser.add_argument(
        "--out-dir",
        default="runtime/annotations/final_external_overlap",
        help="Output folder for filtered overlap/non-overlap files",
    )
    args = parser.parse_args()

    summary = json.loads(Path(args.run_summary).read_text(encoding="utf-8"))
    train_ids = {prefix12(slide) for slide in summary["selected_slides"] if slide}
    train_ids.discard("")

    datasets = {
        "coadread_tcga_pub": read_csv(Path(args.pub)),
        "coadread_tcga_pan_can_atlas_2018": read_csv(Path(args.pancan)),
        "cptac_coad": read_csv(Path(args.cptac)),
    }

    out_dir = Path(args.out_dir)
    report: dict[str, object] = {
        "run_id": summary.get("bundle_id"),
        "training_patients": len(train_ids),
        "sources": {},
    }

    for source_name, rows in datasets.items():
        overlap_rows = [row for row in rows if prefix12(row.get("patient_id", "")) in train_ids]
        holdout_rows = [row for row in rows if prefix12(row.get("patient_id", "")) not in train_ids]
        fieldnames = list(rows[0].keys()) if rows else []
        write_csv(out_dir / f"{source_name}_overlap.csv", overlap_rows, fieldnames)
        write_csv(out_dir / f"{source_name}_holdout.csv", holdout_rows, fieldnames)
        report["sources"][source_name] = {
            "total": summarize_rows(rows),
            "overlap": summarize_rows(overlap_rows),
            "holdout": summarize_rows(holdout_rows),
        }

    report_path = out_dir / "overlap_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
