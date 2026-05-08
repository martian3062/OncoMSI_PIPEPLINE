# Top-4 Hybrid Rebuild

This rebuild targets only the validated top 4 approaches:

- `Approach1-UNI2-h`
- `Approach2-Virchow2`
- `Approach5-H-Optimus-0`
- `Approach6-Midnight-12k`

It also keeps the next-stage evaluation goals tied to this reduced roster:

1. `CPTAC-COAD` external validation
2. 5-seed Monte Carlo stability rerun
3. Fold-level comparison and reporting
4. Top-4 hybrid ensemble as the final deliverable

## What To Recreate On The VM

Remote project root:

```text
/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc
```

Key remote layout:

```text
project_1_slideflow_msi_tcga_crc/
  django_rebuild_cleaned_msi/
    apps/
    msi_platform/
    runtime/
      annotations/
      bundle_configs/
      launch_logs/
      run_status/
      tmp/
    static/
    manage.py
    requirements.txt
    .env
  scripts/
    run_tcga_coad_automated_triad.py
    hybrid_extractors.py
  models/
    virchow/
      pytorch_model.bin
  automation/
    tcga_slide_triads/
```

## Priority Mapping

### Priority 1: External validation

Keep `external_cohorts` in the run payload:

- `name`: `CPTAC-COAD`
- `annotations_csv`:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/cptac_coad_annotations.csv`
- `source_uri`: `gs://wsi_aiml_repo/CPTAC/CPTAC_COAD`

Current code already carries the cohort in the top-4 launcher payload. The
remaining missing piece is end-to-end external scoring inside the runner output.

### Priority 2: Monte Carlo stability

Keep:

- `n_repeats = 5`
- `repeat_seeds = [310, 42, 7, 2025, 1337]`

Weighted ranking target:

```text
0.40 * mean_auroc
+ 0.25 * mean_auprc
+ 0.15 * balanced_accuracy
+ 0.10 * msi_h_sensitivity
+ 0.10 * calibration_score
- 0.20 * seed_std
```

The local analysis helper in `tools/top4_priority_report.py` computes the same
ranking shape from saved metrics, using `seed_std` when present and otherwise
falling back to `auroc_std` as a temporary proxy for offline review.

### Priority 3: Fold-level analysis

The saved `fold_metrics.csv` files are enough to report:

- mean ± std across folds
- weakest folds by AUROC / F1 / MSI-H recall

True DeLong testing still needs prediction-level fold outputs, not just summary
CSVs.

### Priority 4: Top-4 hybrid ensemble

Desired ensemble tracks:

1. Late fusion:
   average the 4 model probabilities per slide
2. Stacked meta-learner:
   logistic regression on fold-level prediction columns

For this project, the default path should be:

1. train the 4 base models
2. score CPTAC external validation for each
3. build one `Top4-Hybrid-Ensemble` late-fusion result
4. treat that ensemble as the final production-facing result

This cannot be completed from the current local artifacts alone because the
prediction parquet files referenced in `metrics.json` are not present locally.
Once they are restored on the VM, this becomes the main post-hoc aggregation
step, not just an optional comparison.

## Final Recommendation

If you want one end result instead of 4 separate finalists, use:

- `late_fusion`
- equal-weight average over:
  - `Approach1-UNI2-h`
  - `Approach2-Virchow2`
  - `Approach5-H-Optimus-0`
  - `Approach6-Midnight-12k`

Reason:

- no retraining required beyond the 4 base models
- easiest to reproduce
- easiest to explain in a paper or product note
- strongest chance of beating any single model on external validation
