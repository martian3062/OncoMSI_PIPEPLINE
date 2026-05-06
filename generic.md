# Generic MSI System Reference

This file is the framework-agnostic reference for the current `4basecare-MSI`
system. It is meant to help when rebuilding the product in a different
framework, re-platforming the UI/API, or recreating the VM pipeline from
scratch without depending on older branch-specific README text.

It documents:

- what the system does today
- what code paths are actually active
- what runs locally versus on the VM
- what models, extractors, and artifacts are used
- how training is launched, tracked, archived, and interpreted
- what VM paths and commands matter operationally

This file is intentionally more detailed than the top-level README.

## 1. System Purpose

The current system is a TCGA colorectal MSI training and tracking workstation.
Its active production-like flow is:

1. select a TCGA COAD slide cohort from annotation CSVs
2. prefer `DX1` slides
3. download `.svs` files from a Google Cloud Storage bucket to the VM
4. create Slideflow TFRecords and feature bags
5. train two MIL approaches in parallel
6. aggregate fold metrics
7. archive run outputs for later summary and comparison
8. expose live and archived status to a dashboard

There is also an older CRC patch-classification path still present in the repo,
but the main active end-to-end workflow is the TCGA slide-level MSI pipeline.

## 2. Current Architecture

The current repo shape under `main/` is:

```text
main/
  apps/
    api/
    web/
  annotations/
  automation/
  datasets/
  scripts/
```

The active runtime split is:

- `apps/web`: Next.js dashboard
- `apps/api`: FastAPI backend
- VM: Slideflow, pathology environment, `.svs` downloads, TFRecords, bags, MIL training

The heavy pathology work does not run on the Windows laptop. The laptop is used
for source editing and optional local UI/API development. The real slide
processing and MIL training run on the VM.

## 3. Direct VM Reference

This is the current operational VM target used in this repo.

### SSH

Windows PowerShell login command:

```powershell
ssh -i "%USERPROFILE%\.ssh\evolet_rsa" pardeep@34.59.145.240
```

Equivalent PowerShell syntax used in many commands:

```powershell
ssh -i "$env:USERPROFILE\.ssh\evolet_rsa" pardeep@34.59.145.240
```

### VM identity

- user: `pardeep`
- host: `34.59.145.240`
- conda env: `pathology310`
- preferred runner Python:
  `/home/pardeep/.venvs/pathology310-fastai/bin/python`
- shared project root:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc`

### Important VM paths

- repo/runtime root:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc`
- live triad bundles:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/automation/tcga_slide_triads`
- archived batch runs:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/automation/tcga_batch_archives*`
- feature/model assets:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/models`
- Virchow weights used in validated runs:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/models/virchow/pytorch_model.bin`

### Hardware

Validated live training has been running on:

- GPU: `NVIDIA L4`
- visible CUDA device count: `1`

## 4. Active Web Surface

The current frontend is a single dashboard centered on the TCGA runner.

Main files:

- `apps/web/src/app/page.tsx`
- `apps/web/src/components/msi-workbench.tsx`
- `apps/web/src/app/globals.css`

Current frontend behavior:

- checks API health
- launches TCGA runs
- polls bundle status
- polls latest archive summary
- stores the tracked `bundle_id` in browser local storage
- renders label balance and approach metrics

Important details from code:

- API base default:
  `http://127.0.0.1:8001`
- polling cadence:
  `15` seconds
- local storage key:
  `msi-single-system-latest-bundle`

The dashboard prefers the currently tracked bundle, but falls back to the
latest bundle if the tracked one is stale, missing, or invalid.

## 5. Active API Surface

The real FastAPI entrypoint is:

- `apps/api/app/main.py`

The currently mounted routes are:

```text
GET  /health

GET  /vm/status
GET  /vm/files
POST /vm/upload
POST /vm/downloader/start
POST /vm/jupyter/start
POST /vm/tunnel/start
POST /vm/monte-carlo/workspace

POST /approach-2/pipeline/preprocess
POST /approach-2/pipeline/extract_features
POST /approach-2/pipeline/train
POST /approach-2/pipeline/train-triad
POST /approach-2/pipeline/train-tcga-slide-triad
GET  /approach-2/pipeline/train-tcga-slide-triad-latest
GET  /approach-2/pipeline/tcga-batch-archive-latest
GET  /approach-2/pipeline/train-tcga-slide-triad/{bundle_id}
POST /approach-2/pipeline/predict
POST /approach-2/pipeline/predict-upload
```

Static artifacts are mounted at:

```text
/approach-2/artifacts
```

Important note:

Many older route files still exist in the repo, but they are not part of the
current mounted app unless re-wired manually.

## 6. Main Code Paths

### Active TCGA slide-level path

- route layer:
  `apps/api/app/approach_2/api/pipeline.py`
- runtime orchestration:
  `apps/api/app/approach_2/services/triad_runtime.py`
- per-bundle runner:
  `scripts/run_tcga_coad_automated_triad.py`
- batch/archive orchestrator:
  `scripts/run_tcga_coad_four_batches.py`

### Older CRC patch path still present

- patch training pipeline:
  `apps/api/app/approach_2/pipelines/patch_trainer.py`
- VM patch training script:
  `scripts/run_crc_patch_training.py`
- upload prediction helper:
  `scripts/predict_crc_patch_image.py`

## 7. TCGA Request Defaults

The dashboard currently launches this default shape from
`apps/web/src/components/msi-workbench.tsx`:

```json
{
  "experiment_name": "tcga-coad-dx1-single-system",
  "bucket_uri": "gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD",
  "slide_limit": 110,
  "n_folds": 2,
  "n_repeats": 2,
  "preferred_slide_pattern": "DX",
  "preferred_exact_suffix": "DX1",
  "annotations_csv": "annotations/tcga_coad_bucket_annotations_final_all3_live_dx1.csv",
  "feature_extractor": "virchow,ctranspath",
  "allow_generic_fallback": false,
  "tile_px": 256,
  "tile_um": 128,
  "max_parallel_approaches": 2,
  "max_tiles_per_slide": 96,
  "mpp_override": 0.25,
  "qc_method": "otsu"
}
```

These are UI defaults, not hard system limits. Larger and more specialized
runs have also been launched directly through the batch script.

## 8. Bundle Lifecycle

The TCGA runner creates a bundle under:

```text
automation/tcga_slide_triads/{bundle_id}
```

Core files in a live bundle:

- `bundle_config.json`
- `status.json`
- `prepared_bundle.json`
- `final_summary.json`
- `runner.log`
- `approaches/Approach1/status.json`
- `approaches/Approach1/metrics.json`
- `approaches/Approach1/fold_metrics.csv`
- `approaches/Approach2/status.json`
- `approaches/Approach2/metrics.json`
- `approaches/Approach2/fold_metrics.csv`

Important runtime behavior:

- the live bundle directory may be cleaned after archival
- after cleanup, `metrics.json` may no longer exist in the live working folder
- the true post-run source of truth becomes the archived copy under the batch archive root

## 9. Bundle Status States

The dashboard and API expect these main states:

- `matching_annotations`
- `downloading_slides`
- `extracting_tiles`
- `retrying_tiles`
- `generating_features`
- `prepared`
- `training_parallel`
- `completed`
- `failed`

Per-approach states are tracked separately under
`approaches/{ApproachLabel}/status.json`.

## 10. Data Selection Logic

The current TCGA slide flow is pathology-first and selective.

High-level selection behavior:

- reads a CSV of TCGA annotations
- expects `msi_status`
- matches rows to bucket `.svs` files
- prefers slide IDs containing `DX`
- prefers the exact suffix `DX1`
- builds a balanced or constrained subset
- downloads only the chosen slides for the run

The active bundle runner logic lives in:

- `scripts/run_tcga_coad_automated_triad.py`

The batch splitting logic lives in:

- `scripts/run_tcga_coad_four_batches.py`

## 11. Batch Orchestration

The file:

- `scripts/run_tcga_coad_four_batches.py`

is the sequential batch loop driver.

It does:

1. read a source annotation CSV
2. split rows into batches
3. write one batch CSV per batch
4. build a bundle config for that batch
5. run the per-bundle TCGA script
6. archive outputs
7. clean the working bundle
8. move to the next batch

Key knobs:

- `--batch-count`
- `--batch-size`
- `--positive-per-batch`
- `--feature-extractor`
- `--virchow-weights`
- `--max-tiles-per-slide`
- `--n-folds`

This is the right place to modify the loop when rebuilding in another
framework, because it expresses the orchestration shape more clearly than the
UI.

## 12. Preprocessing and Feature Extraction

The validated slide-level runner uses:

- tile size:
  `256 px`
- tile microns:
  `128 um`
- optional MPP override:
  usually `0.25`
- QC method:
  `otsu`

Important runtime detail from the current script:

- `EXTRACTION_WORKERS = 1`

This was kept deliberately because the VM TCGA `.svs` extraction path was more
stable with a single extraction worker. Multi-process extraction had caused
zero-byte TFRecords or unfinished tile outputs on this cohort.

## 13. Feature Extractors

The code supports candidate lists and fallback ordering.

Core extractor logic:

- request candidates are parsed from `feature_extractor` or `feature_extractors`
- generic CNN fallback can be blocked with `allow_generic_fallback = false`
- extractor success is recorded in `prepared_bundle.json` and `final_summary.json`

Validated extractors in this repo family:

- `virchow`
- `ctranspath`
- `retccl`

Other extractors may exist in code or environment history, but the recent live
validated TCGA runs used the above.

### Virchow

Recent live validated runs used:

- extractor name: `virchow`
- weights path:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/models/virchow/pytorch_model.bin`

This required Hugging Face gated access plus a valid token at runtime.

## 14. Training Approaches

The current two slide-level approaches share the same upstream data and feature
bags, but differ in the MIL head and some trainer settings.

### Approach1

- label: `Approach1`
- MIL model: `transmil`
- epochs: `20`
- learning rate: `5e-5`
- weight decay: `1e-4`
- MIL batch size: `12`
- bag size: `128`
- max val bag size: `128`
- weighted loss: `true`
- fit one cycle: `true`
- seed: `310`

### Approach2

- label: `Approach2`
- MIL model: `attention_mil`
- epochs: `20`
- learning rate: `7e-5`
- weight decay: `1e-4`
- MIL batch size: `16`
- bag size: `128`
- max val bag size: `128`
- weighted loss: `true`
- fit one cycle: `true`
- seed: `310`

These presets are defined in:

- `apps/api/app/approach_2/services/triad_runtime.py`
- `scripts/run_tcga_coad_four_batches.py`

The actual per-fold training is executed by Slideflow in:

- `scripts/run_tcga_coad_automated_triad.py`

through:

```python
project.train_mil(...)
```

## 15. Parallel Training Behavior

The two approaches are launched in parallel.

The runner:

- writes one `runner.log` per approach
- spawns one process per approach
- watches process liveness
- reads `metrics.json` or `status.json` when a worker exits
- updates the bundle `status.json` with running and completed approaches

This is why the dashboard can show bundle-level progress even while each
approach is still writing its own metrics separately.

## 16. Metrics and Artifacts

The main training outputs are:

- `metrics.json`
- `fold_metrics.csv`
- prediction files
- bundle `final_summary.json`
- archive-level `orchestration_status.json`

Common metric fields:

- `mean_auroc`
- `mean_f1_macro`
- `default_threshold_f1_macro`
- `folds`
- `epochs`

Important practical note:

Some summary surfaces use `default_threshold_f1_macro`, while older docs or
UI logic may refer to `mean_f1_macro_default_threshold`. When rebuilding, it is
better to normalize the metrics contract rather than mirror every historical
field name.

## 17. Archive Structure

Archived batches are stored under a root like:

```text
automation/tcga_batch_archives_YYYYMMDD_<run_name>
```

Each batch archive can contain:

- `batch_annotations.csv`
- `status.json`
- `final_summary.json`
- `prepared_bundle.json`
- `runner.log`
- `approaches/Approach1/metrics.json`
- `approaches/Approach2/metrics.json`
- `approaches/Approach1/fold_metrics.csv`
- `approaches/Approach2/fold_metrics.csv`

The archive controller writes:

- `orchestration_status.json`

This becomes the best long-lived summary source after live bundle cleanup.

## 18. Latest Validated High-Value Run

As of `2026-05-05`, the strongest validated recent apples-to-apples run in this
repo family was:

- bundle: `live180virchow7f256_01`
- extractor: `virchow`
- slides: `180`
- labels: `70 MSI-H`, `110 MSS`
- folds: `7`
- repeats: `2`
- max tiles per slide: `256`
- epochs: `20`
- return code: `0`

Results:

- `Approach1`
  - AUROC: `0.8686904761904762`
  - macro F1: `0.8365702053222058`
- `Approach2`
  - AUROC: `0.8685119047619047`
  - macro F1: `0.8391286225694833`

Comparison with the earlier validated `RetCCL` `256`-tile run:

- `Approach1` improved from `0.8413988095238095` to `0.8686904761904762`
- `Approach2` improved from `0.8406845238095236` to `0.8685119047619047`

That means Virchow gave roughly a `+0.027` to `+0.028` AUROC lift on this
setup.

## 19. Known Operational Constraints

These are the main practical constraints that shaped the current implementation.

### Storage pressure

Large `.svs` runs create heavy storage usage from:

- downloaded slides
- TFRecords
- tiles
- feature bags
- MIL output folders
- archives

The system therefore uses batching, archival, and cleanup aggressively.

### Extraction stability

The current script favors a single extraction worker because parallel tile
extraction was less reliable on the TCGA cohort in this VM environment.

### Gated model access

Virchow required:

- a valid Hugging Face token
- approved access to the gated model
- an accessible weights path on the VM

### Small-cohort variance

Slide-level MSI AUROC is sensitive to:

- total slide count
- positive-class count
- fold count
- repeat count
- tile budget
- morphology signal variability across slides

## 20. Why This Matters For A Framework Rewrite

If this system is rebuilt in another framework, the core logic to preserve is
not the exact UI code. It is the operational contract:

1. launch a VM-backed bundle
2. store a stable `bundle_id`
3. read `status.json` during live execution
4. read archive summaries after cleanup
5. separate bundle state from approach state
6. preserve the distinction between live bundle files and archived results
7. treat the VM project root and model-weight paths as configurable
8. keep data selection, extraction, feature generation, and MIL training as separate phases

The most important framework-independent entities are:

- `bundle`
- `approach`
- `archive`
- `feature_extractor`
- `fold metrics`
- `selected slides`
- `label balance`
- `remote status path`

## 21. Suggested Generic Domain Model

If rebuilding from scratch, a cleaner generic model would be:

### Run

- `run_id`
- `source_type`
- `source_uri`
- `requested_slide_limit`
- `selected_slide_count`
- `label_counts`
- `feature_extractor_candidates`
- `feature_extractor_used`
- `tile_px`
- `tile_um`
- `max_tiles_per_slide`
- `n_folds`
- `n_repeats`
- `state`
- `remote_status_path`
- `archive_path`

### Approach

- `run_id`
- `approach_name`
- `model_family`
- `trainer_params`
- `state`
- `mean_auroc`
- `mean_f1_macro`
- `prediction_artifacts`

### BatchArchive

- `archive_root`
- `state`
- `completed_batches`
- `aggregate_label_counts`
- `aggregate_approaches`

This is cleaner than binding everything directly to old route names or old file
shapes.

## 22. Reference Commands

### Local API dev

```powershell
cd E:\4basecare-MSI\main\apps\api
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### Local web dev

```powershell
cd E:\4basecare-MSI\main\apps\web
npm.cmd run dev -- --hostname 127.0.0.1 --port 3000
```

### VM-side runner pattern

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate pathology310
RUNNER_PYTHON=/home/pardeep/.venvs/pathology310-fastai/bin/python
"$RUNNER_PYTHON" scripts/run_tcga_coad_automated_triad.py --bundle-config "<bundle_config_path>"
```

### VM-side batch loop pattern

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate pathology310
python scripts/run_tcga_coad_four_batches.py ...
```

## 23. What Is Safe To Treat As Current Truth

These are the best current source-of-truth layers, in order:

1. active code under `main/apps/api`, `main/apps/web`, and `main/scripts`
2. live bundle `status.json` during a run
3. archived `orchestration_status.json` after a run
4. this `generic.md`
5. older README text

If any README disagrees with code, prefer code.

## 24. Short Rebuild Checklist

If the system is reimplemented in another framework, preserve these first:

1. VM config and SSH contract
2. bundle config schema
3. live status polling
4. archive summary polling
5. feature extractor selection
6. two-approach MIL contract
7. batch orchestration and cleanup
8. clear artifact paths
9. separation between live and archived state
10. metric normalization

## 25. Files To Read First

If someone new needs to understand the current system quickly, start here:

1. `main/generic.md`
2. `main/README.md`
3. `main/apps/api/app/main.py`
4. `main/apps/api/app/approach_2/api/pipeline.py`
5. `main/apps/api/app/approach_2/services/triad_runtime.py`
6. `main/scripts/run_tcga_coad_automated_triad.py`
7. `main/scripts/run_tcga_coad_four_batches.py`
8. `main/apps/web/src/components/msi-workbench.tsx`

That sequence gives the clearest top-down understanding of the live system.
