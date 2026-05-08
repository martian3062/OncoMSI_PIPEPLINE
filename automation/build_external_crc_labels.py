#!/usr/bin/env python3
"""Build runner-friendly external CRC MSI label files from public sources."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen


CPTAC_PATH = Path("runtime/annotations/cptac_coad_annotations.csv")

TCGA_PUB_URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
    "public/coadread_tcga_pub/data_clinical_sample.txt"
)
TCGA_PANCAN_URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
    "public/coadread_tcga_pan_can_atlas_2018/data_clinical_sample.txt"
)


def fetch_tsv(url: str) -> list[dict[str, str]]:
    req = Request(url, headers={"User-Agent": "OncoMSI-external-labels/1.0"})
    with urlopen(req, timeout=90) as resp:
        text = resp.read().decode("utf-8")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    return list(reader)


def normalize_pub_label(value: str) -> str | None:
    text = (value or "").strip().upper()
    if text == "MSI-H":
        return "MSI-H"
    if text in {"MSS", "MSI-L"}:
        return "MSS"
    return None


def parse_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text or text == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def derive_pancan_label(mantis: float | None, sensor: float | None) -> tuple[str | None, str]:
    """Derive a categorical label from PanCan MSI scores.

    Rules:
    - MSI-H if any available score is decisively MSI-H.
    - MSS if all available scores are decisively MSS.
    - otherwise unlabeled / indeterminate.
    """
    evidence: list[str] = []
    if mantis is not None:
        if mantis > 0.6:
            evidence.append("mantis_msi")
        elif mantis < 0.4:
            evidence.append("mantis_mss")
        else:
            evidence.append("mantis_indeterminate")
    if sensor is not None:
        if sensor > 10:
            evidence.append("sensor_msi")
        elif sensor < 4:
            evidence.append("sensor_mss")
        else:
            evidence.append("sensor_indeterminate")

    if any(item.endswith("_msi") for item in evidence):
        return "MSI-H", ",".join(evidence) or "no_score"

    decisive = [item for item in evidence if item.endswith("_mss")]
    indeterminate = [item for item in evidence if item.endswith("_indeterminate")]
    if decisive and not indeterminate:
        return "MSS", ",".join(evidence)

    return None, ",".join(evidence) or "no_score"


def build_cptac_rows() -> list[dict[str, str]]:
    with CPTAC_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "slide": row["slide"],
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "label": row["label"],
                "msi_status": row["msi_status"],
                "study_id": row["study_id"],
                "cancer_type": row["cancer_type"],
                "cancer_type_detailed": row["cancer_type_detailed"],
                "source": row["source"],
                "label_basis": "clinical_msi_status",
                "mantis_score": "",
                "msisensor_score": "",
            }
        )
    return out_rows


def build_pub_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw_rows = fetch_tsv(TCGA_PUB_URL)
    out_rows = []
    for row in raw_rows:
        label = normalize_pub_label(row.get("MSI_STATUS", ""))
        if not label:
            continue
        out_rows.append(
            {
                "slide": row["SAMPLE_ID"],
                "sample_id": row["SAMPLE_ID"],
                "patient_id": row["PATIENT_ID"],
                "label": label,
                "msi_status": row.get("MSI_STATUS", ""),
                "study_id": "coadread_tcga_pub",
                "cancer_type": row.get("CANCER_TYPE", ""),
                "cancer_type_detailed": row.get("CANCER_TYPE_DETAILED", ""),
                "source": "cBioPortal COADREAD TCGA Pub clinical sample MSI_STATUS",
                "label_basis": "clinical_msi_status",
                "mantis_score": "",
                "msisensor_score": "",
            }
        )
    return out_rows, raw_rows


def build_pancan_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw_rows = fetch_tsv(TCGA_PANCAN_URL)
    out_rows = []
    for row in raw_rows:
        mantis = parse_float(row.get("MSI_SCORE_MANTIS", ""))
        sensor = parse_float(row.get("MSI_SENSOR_SCORE", ""))
        label, basis = derive_pancan_label(mantis, sensor)
        if not label:
            continue
        out_rows.append(
            {
                "slide": row["SAMPLE_ID"],
                "sample_id": row["SAMPLE_ID"],
                "patient_id": row["PATIENT_ID"],
                "label": label,
                "msi_status": label,
                "study_id": "coadread_tcga_pan_can_atlas_2018",
                "cancer_type": row.get("CANCER_TYPE", ""),
                "cancer_type_detailed": row.get("CANCER_TYPE_DETAILED", ""),
                "source": "cBioPortal COADREAD TCGA PanCan Atlas clinical sample MSI scores",
                "label_basis": basis,
                "mantis_score": row.get("MSI_SCORE_MANTIS", ""),
                "msisensor_score": row.get("MSI_SENSOR_SCORE", ""),
            }
        )
    return out_rows, raw_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runtime/annotations")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fieldnames = [
        "slide",
        "sample_id",
        "patient_id",
        "label",
        "msi_status",
        "study_id",
        "cancer_type",
        "cancer_type_detailed",
        "source",
        "label_basis",
        "mantis_score",
        "msisensor_score",
    ]

    cptac_rows = build_cptac_rows()
    pub_rows, pub_raw = build_pub_rows()
    pancan_rows, pancan_raw = build_pancan_rows()
    merged_rows = sorted(
        cptac_rows + pub_rows + pancan_rows,
        key=lambda row: (row["study_id"], row["sample_id"]),
    )

    write_csv(out_dir / "cptac_coad_annotations_extended.csv", cptac_rows, fieldnames)
    write_csv(out_dir / "coadread_tcga_pub_annotations.csv", pub_rows, fieldnames)
    write_csv(out_dir / "coadread_tcga_pan_can_atlas_2018_annotations.csv", pancan_rows, fieldnames)
    write_csv(out_dir / "external_crc_msi_labels_merged.csv", merged_rows, fieldnames)

    raw_fieldnames = [
        "study_id",
        "patient_id",
        "sample_id",
        "msi_status",
        "mantis_score",
        "msisensor_score",
        "cancer_type",
        "cancer_type_detailed",
    ]
    raw_rows = []
    for row in pub_raw:
        raw_rows.append(
            {
                "study_id": "coadread_tcga_pub",
                "patient_id": row.get("PATIENT_ID", ""),
                "sample_id": row.get("SAMPLE_ID", ""),
                "msi_status": row.get("MSI_STATUS", ""),
                "mantis_score": "",
                "msisensor_score": "",
                "cancer_type": row.get("CANCER_TYPE", ""),
                "cancer_type_detailed": row.get("CANCER_TYPE_DETAILED", ""),
            }
        )
    for row in pancan_raw:
        raw_rows.append(
            {
                "study_id": "coadread_tcga_pan_can_atlas_2018",
                "patient_id": row.get("PATIENT_ID", ""),
                "sample_id": row.get("SAMPLE_ID", ""),
                "msi_status": "",
                "mantis_score": row.get("MSI_SCORE_MANTIS", ""),
                "msisensor_score": row.get("MSI_SENSOR_SCORE", ""),
                "cancer_type": row.get("CANCER_TYPE", ""),
                "cancer_type_detailed": row.get("CANCER_TYPE_DETAILED", ""),
            }
        )
    write_csv(out_dir / "external_crc_source_rows.csv", raw_rows, raw_fieldnames)

    summary = {
        "cptac_coad_annotations_extended.csv": Counter(row["label"] for row in cptac_rows),
        "coadread_tcga_pub_annotations.csv": Counter(row["label"] for row in pub_rows),
        "coadread_tcga_pan_can_atlas_2018_annotations.csv": Counter(row["label"] for row in pancan_rows),
        "external_crc_msi_labels_merged.csv": Counter(row["label"] for row in merged_rows),
    }
    for name, counts in summary.items():
        print(name)
        print(f"  rows={sum(counts.values())}")
        for label, count in sorted(counts.items()):
            print(f"  {label}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
