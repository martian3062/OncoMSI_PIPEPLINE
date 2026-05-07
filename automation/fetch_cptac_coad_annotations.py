#!/usr/bin/env python3
"""Fetch CPTAC-COAD MSI labels from cBioPortal into runner-friendly CSV files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

STUDY_ID = "coad_cptac_2019"
CBIO_BASE = "https://www.cbioportal.org/api"


def fetch_json(url: str):
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "OncoMSI-CPTAC-fetch/1.0"})
    with urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_label(value: str) -> str | None:
    text = (value or "").strip().upper()
    if text in {"MSI", "MSI-H", "MSI HIGH", "MSI-HIGH", "MSI_H", "MSI_HIGH"}:
        return "MSI-H"
    if text in {"MSS", "MSI-L", "MSI LOW", "MSI-LOW", "MSI_L", "MSI_LOW"}:
        return "MSS"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runtime/annotations/cptac_coad_annotations.csv")
    parser.add_argument("--raw-out", default="runtime/annotations/cptac_coad_clinical_sample_long.csv")
    args = parser.parse_args()

    rows = fetch_json(
        f"{CBIO_BASE}/studies/{STUDY_ID}/clinical-data?clinicalDataType=SAMPLE&projection=SUMMARY&pageSize=100000"
    )
    per_sample: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        sample_id = row.get("sampleId", "")
        if not sample_id:
            continue
        per_sample[sample_id][row.get("clinicalAttributeId", "")] = row.get("value", "")
        per_sample[sample_id]["patient_id"] = row.get("patientId", sample_id)

    out_rows = []
    for sample_id in sorted(per_sample):
        attrs = per_sample[sample_id]
        msi_status = attrs.get("MSI_STATUS", "")
        label = normalize_label(msi_status)
        if not label:
            continue
        out_rows.append(
            {
                "slide": sample_id,
                "sample_id": sample_id,
                "patient_id": attrs.get("patient_id", sample_id),
                "label": label,
                "msi_status": msi_status,
                "study_id": STUDY_ID,
                "cancer_type": attrs.get("CANCER_TYPE", ""),
                "cancer_type_detailed": attrs.get("CANCER_TYPE_DETAILED", ""),
                "source": "cBioPortal CPTAC-COAD clinical sample MSI_STATUS",
            }
        )

    out_path = Path(args.out)
    raw_path = Path(args.raw_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["slide", "label"])
        writer.writeheader()
        writer.writerows(out_rows)

    raw_fields = ["sampleId", "patientId", "studyId", "clinicalAttributeId", "value"]
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["label"] for row in out_rows)
    print(json.dumps({"study_id": STUDY_ID, "rows": len(out_rows), "label_counts": counts, "out": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
