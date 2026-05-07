# OncoMSI_PIPEPLINE

OncoMSI is a Django-based MSI training workbench for pathology experiments on
TCGA COAD whole-slide images. It was rebuilt from scratch as a modular control
plane that can launch real remote training jobs, track multi-approach runs, and
surface live results in a single integrated web application.

This repository is not a static demo dashboard. It is a working orchestration
layer around a real VM-side pathology pipeline.

## Results Snapshot

This top section is the single combined scoreboard for the last completed
`hybrid-02` run and the preserved legacy 7-approach baseline. The detailed
writeups remain later in this README, but the duplicated summary tables are
collapsed here so the opening is easier to scan.

### Combined Completed Results

| Track | Approach | Extractor used | AUROC | F1 macro | AUPRC | Bal Acc | MSI-H Recall | Specificity | Best threshold | State |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Latest completed `run-69fb62874c` | Approach2-Phikon-v2 | `phikon-v2` | `0.9422` | `0.9289` | `0.9489` | `0.9243` | `0.9718` | `0.8768` | `-` | `completed` |
| Latest completed `run-69fb62874c` | Approach3-Prov-GigaPath | `prov-gigapath` | `0.9555` | `0.9302` | `0.9692` | `0.9320` | `0.9336` | `0.9304` | `-` | `completed` |
| Latest completed `run-69fb62874c` | Approach4-PRISM | `prism-virchow` | `0.9402` | `0.9044` | `0.9603` | `0.9008` | `0.9427` | `0.8589` | `-` | `completed` |
| Latest completed `run-69fb62874c` | Approach7-Midnight-12k | `midnight` | `0.9535` | `0.9394` | `0.9632` | `0.9390` | `0.9636` | `0.9143` | `-` | `completed` |
| Legacy baseline `run-c167be196bac` | Approach 1 - Virchow | `virchow` | `0.9265` | `0.8915` | `-` | `-` | `-` | `-` | `0.4605` | `completed` |
| Legacy baseline `run-c167be196bac` | Approach 2 - RetCCL | `retccl` | `0.9444` | `0.9114` | `-` | `-` | `-` | `-` | `0.3997` | `completed` |
| Legacy baseline `run-c167be196bac` | Approach 3 - CTransPath | `ctranspath` | `0.9281` | `0.9046` | `-` | `-` | `-` | `-` | `0.4222` | `completed` |
| Legacy baseline `run-c167be196bac` | Approach 4 - CONCH | `conch` | `0.9329` | `0.9106` | `-` | `-` | `-` | `-` | `0.4131` | `completed` |
| Legacy baseline `run-c167be196bac` | Approach 5 - Virchow2 | `virchow2` | `0.9684` | `0.9388` | `-` | `-` | `-` | `-` | `0.3612` | `completed` |
| Legacy baseline `run-c167be196bac` | Approach 6 - UNI2-H | `uni2-h` | `0.9819` | `0.9660` | `-` | `-` | `-` | `-` | `0.3940` | `completed` |
| Legacy baseline `run-c167be196bac` | Approach 7 - H-Optimus-0 | `h-optimus-0` | `0.9594` | `0.9452` | `-` | `-` | `-` | `-` | `0.3003` | `completed` |

### Quick Read

- best legacy baseline result: `Approach 6 - UNI2-H` with AUROC `0.9819`
- best latest completed `hybrid-02` result: `Approach3-Prov-GigaPath` with AUROC `0.9555`
- top-of-file note: later sections keep the full run notes and branch history, so
  this section replaces the duplicate opening summary instead of deleting it

### Semi-Final Currently Selected Roster

This is the active strict semi-final roster for the in-progress `200`-slide
foundation-model run. These are the models currently selected for the fresh
semi-final benchmark, not completed result rows yet.

| Slot | Semi-final approach | Extractor / model used | Access | Status in plan |
| ---: | --- | --- | --- | --- |
| 1 | UNI2-h | `MahmoodLab/UNI2-h` | gated | selected |
| 2 | Virchow2 | `paige-ai/Virchow2` | gated | selected |
| 3 | Prov-GigaPath | `prov-gigapath/prov-gigapath` | gated | selected |
| 4 | CONCHv1.5 | `MahmoodLab/conchv1_5` | gated | selected |
| 5 | H-Optimus-0 | `bioptimus/H-optimus-0` | open | selected |
| 6 | Midnight-12k | `kaiko-ai/midnight` | open | selected |
| 7 | DINOv2-Large | `facebook/dinov2-large` | open | selected |
| 8 | DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | gated | selected |
| 9 | CHIEF | `github.com/hms-dbmi/CHIEF` + official Docker weights | request/docker | selected |
| 10 | RetCCL | `retccl` | slideflow/native | selected |

### Semi-Final Current Run Settings

| Item | Value |
| --- | --- |
| run intent | `semi-final` |
| active run id | `run-8635c038adcc` |
| requested slides | `200` |
| folds | `10` |
| repeats | `1` |
| max tiles per slide | `256` |
| max parallel approaches | `3` |
| generic fallback | disabled |

## Hybrid Direction

The next architectural direction is a hybrid pipeline:

- keep `Django` as the control plane
- keep `Slideflow` only for the stable paths it already handles well
- move new foundation-model integration into framework-agnostic extractor
  adapters
- preserve history, archives, and multi-approach orchestration from the same UI

Detailed plan:

- [docs/hybrid_pipeline_architecture.md](./docs/hybrid_pipeline_architecture.md)

## What We Built

The project started from a generic MSI orchestration concept and was turned into
an operational Django system with these core capabilities:

- integrated UI inside Django using HTMX, Alpine.js, Plotly, and custom CSS
- modular experiment architecture with separate app boundaries
- real VM execution support over SSH or local mode
- run blueprint creation from the web UI
- remote bundle-config generation for the pathology runner
- live status syncing from remote `status.json` and approach-level `metrics.json`
- multi-approach training support with extractor-specific settings
- n8n-ready automation hooks
- Hugging Face token-aware extractor support
- archive sync entry points for completed bundle outputs

During this phase we also pushed the system beyond scaffold level:

- uploaded and deployed the Django app on the pathology VM
- integrated real TCGA bucket selection and annotation-driven matching
- preserved the old 7-approach leaderboard as a historical baseline
- switched the active roster to the `hybrid-02` foundation-model lineup
- patched the remote runner pathing so training branches could restart from a
  prepared bundle instead of re-downloading slides
- synchronized completed metrics back into the frontend
- expanded the metric schema to store AUPRC, balanced accuracy, MSI-H recall,
  specificity, confusion counts, fold-level variance, calibration, and
  external validation metadata

## Why Django For This Project

This system is not just a training script and not just a website.

It needs:

- a durable database-backed control layer
- admin-friendly models for runs, approaches, archives, and VM targets
- a server-rendered dashboard that can move quickly without a separate frontend
- API endpoints for launch, sync, status, and automation triggers
- room to grow into a full internal platform

`FastAPI` would be excellent for a thin ML API, but `Django` is the better fit
for an integrated control product where data models, admin workflows, UI pages,
and orchestration all live together.

## Framework Stack

### Backend

- `Django 5`
- `Django REST Framework`
- `SQLite` for the current control-plane persistence layer

### Frontend

- Django templates
- `HTMX` for partial refresh and live panels
- `Alpine.js` for small UI state and toggles
- `Plotly` for Python-generated charts
- custom gradient/glass CSS

### ML / Runtime Integration

- remote Linux pathology VM
- SSH-based command execution
- remote JSON artifact sync
- n8n webhook integration
- Hugging Face token pass-through for gated models

## Architecture

The architecture is intentionally split into control-plane modules instead of
one large monolith file set.

```text
apps/
  core/         UI shell, dashboard pages, HTMX partials, app presentation
  runs/         Run records, run lifecycle, API endpoints, VM launch/sync logic
  approaches/   Pluggable approach templates and default model configurations
  vm/           VM target registry, SSH/local execution, remote file helpers
  archives/     Imported archive snapshots and comparison-ready summaries

msi_platform/   Django project settings, URL root, ASGI/WSGI bootstrapping
automation/
  n8n/          Import-ready workflow files and VM helper scripts
vm_patch/       Patched remote runner assets used to align the VM pipeline
static/         Shared static root placeholder
```

### Architectural Layers

#### 1. Presentation Layer

Implemented in `apps/core`.

Responsibilities:

- render the main dashboard
- show live run cards
- expose launch forms
- display metric summaries
- keep the UI lightweight and server-driven

This layer avoids a separate React or Next.js frontend on purpose. The goal was
to make a production-lean internal platform, not a split FE/BE product with
extra coordination overhead.

#### 2. Control Plane / Domain Layer

Implemented mostly in `apps/runs` and `apps/approaches`.

Responsibilities:

- define what a run is
- define what an approach slot is
- persist experiment settings
- connect a run to one or more approaches
- store synchronized outcome metrics

This is the heart of the application. The web UI and VM runner both revolve
around these persisted objects.

#### 3. Execution Layer

Implemented in `apps/vm` and `apps/runs/vm_runtime.py`.

Responsibilities:

- resolve the default VM target
- create bundle configs
- upload remote config files
- launch the remote runner
- read `status.json`, `metrics.json`, and `final_summary.json`
- normalize remote artifacts into Django model state

This layer is what turns the project from "dashboard only" into a working
orchestration system.

#### 4. Automation Layer

Implemented in `automation/n8n` and `apps/runs/n8n.py`.

Responsibilities:

- emit webhook-ready payloads when runs are created
- provide importable n8n workflow assets
- enable future fan-out into notifications, archive sync, or callback flows

## Data Model

### `Run`

Defined in [apps/runs/models.py](./apps/runs/models.py).

Represents a full MSI experiment request and its remote execution state.

Key fields:

- `run_id`
- `experiment_name`
- `source_uri`
- `annotations_csv`
- `requested_slide_limit`
- `selected_slide_count`
- `label_counts`
- `feature_extractor_candidates`
- `feature_extractor_used`
- `tile_px`, `tile_um`, `max_tiles_per_slide`
- `n_folds`, `n_repeats`
- `external_cohorts`
- `state`
- `remote_status_path`
- `bundle_config_path`
- `remote_launch_log_path`
- `remote_pid`
- `vm_target`

### `RunApproachLink`

Also defined in [apps/runs/models.py](./apps/runs/models.py).

Represents one approach branch inside a run.

Key fields:

- `approach_template`
- `state`
- `trainer_params`
- `mean_auroc`
- `mean_f1_macro`
- `mean_auprc`
- `mean_balanced_accuracy`
- `mean_recall_msi_h`
- `mean_specificity`
- `aggregate_confusion_matrix`
- `auroc_per_fold`, `auroc_std`, `auroc_ci_low`, `auroc_ci_high`
- `external_metrics`
- `metrics_path`
- `prediction_artifacts`

This model is important because one run can branch into multiple extractor and
model combinations while still remaining part of one bundle-level experiment.

### `ApproachTemplate`

Defined in [apps/approaches/models.py](./apps/approaches/models.py).

This stores reusable experiment templates such as:

- label
- model family
- default hyperparameters
- default extractor
- UI color token

The active `hybrid-02` catalog includes 7 approach templates:

- `Approach 1 - CONCHv1.5`
- `Approach 2 - Phikon-v2`
- `Approach 3 - Prov-GigaPath`
- `Approach 4 - PRISM`
- `Approach 5 - CHIEF`
- `Approach 6 - DINOv3`
- `Approach 7 - Midnight-12k`

Notes:

- `results-history/roster_snapshots/hybrid_01_legacy_seven_snapshot.json`
  preserves the previous `Virchow` / `RetCCL` / `CTransPath` / `CONCH` /
  `Virchow2` / `UNI2-H` / `H-Optimus-0` leaderboard before the swap
- `PRISM` currently runs through a PRISM-compatible `Virchow` tile path
- `CHIEF` currently runs through a CHIEF-compatible `CTransPath` patch path
  while the full WSI-level CHIEF head remains future work

### `VMTarget`

Defined in [apps/vm/models.py](./apps/vm/models.py).

This stores the execution destination:

- SSH host
- SSH user
- SSH key path
- execution mode
- project root
- runner Python path

That makes the control plane portable between:

- local execution
- remote SSH execution
- future multiple VM targets

### `BatchArchive`

Defined in [apps/archives/models.py](./apps/archives/models.py).

This stores imported summary information from completed archive folders and lets
the control plane compare historical outputs later.

## Current Runtime Shape

The current system is tuned around a real pathology VM flow.

### Default remote target

- host: `34.59.145.240`
- user: `pardeep`
- project root:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc`
- Django app root:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi`

### Default runner contract

When Django launches a run it:

1. creates a `Run`
2. generates a bundle config
3. uploads that config to the VM
4. also writes `bundle_config.json` into the remote bundle root
5. spawns the real TCGA runner process
6. polls and syncs remote JSON artifacts back into Django

### Remote bundle lifecycle

```text
Run form submit
  -> create Run + RunApproachLink rows
  -> build bundle config
  -> upload config to VM
  -> start remote runner
  -> download slides
  -> extract tiles
  -> generate feature bags
  -> train all approach branches in parallel
  -> write metrics and summary JSON
  -> sync results into Django
  -> render completed outcomes on the dashboard
```

## Legacy Baseline

The strongest preserved TCGA-only baseline run is:

- bundle id:
  `run-c167be196bac`
- state:
  `completed`

This run used:

- bucket source:
  `gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD`
- annotation file:
  `tcga3_vm_annotations.csv`
- exact selected subset:
  `150` slides
- label mix:
  `74 MSI-H`
  `76 MSS`
- folds:
  `10`
- epochs:
  `30`
- max tiles per slide:
  `256`

Legacy approach layout used in the preserved baseline run:

- `Approach 1 - Virchow`
- `Approach 2 - RetCCL`
- `Approach 3 - CTransPath`
- `Approach 4 - CONCH`
- `Approach 5 - Virchow2`
- `Approach 6 - UNI2-H`
- `Approach 7 - H-Optimus-0`

Notes:

- all `7` approaches above completed successfully in `run-c167be196bac`
- `Atlas-2` and `PLUTO-4G` were dropped from the project because public
  weights were not released, so they are not part of the active catalog

Completed metrics from `run-c167be196bac`:

| Approach | Extractor | AUROC | F1 macro | Best threshold |
| --- | --- | ---: | ---: | ---: |
| Approach 1 - Virchow | `virchow` | `0.9265` | `0.8915` | `0.4605` |
| Approach 2 - RetCCL | `retccl` | `0.9444` | `0.9114` | `0.3997` |
| Approach 3 - CTransPath | `ctranspath` | `0.9281` | `0.9046` | `0.4222` |
| Approach 4 - CONCH | `conch` | `0.9329` | `0.9106` | `0.4131` |
| Approach 5 - Virchow2 | `virchow2` | `0.9684` | `0.9388` | `0.3612` |
| Approach 6 - UNI2-H | `uni2-h` | `0.9819` | `0.9660` | `0.3940` |
| Approach 7 - H-Optimus-0 | `h-optimus-0` | `0.9594` | `0.9452` | `0.3003` |

Best result in the current validated run:

- `Approach 6 - UNI2-H`
- AUROC `0.9819`
- F1 macro `0.9660`

Important note:

- this is still a TCGA-only number and is not a defensible external
  leaderboard by itself
- the branch now treats external validation as the top priority, because
  published MSI work usually drops by roughly `5-10` AUROC points on
  non-TCGA cohorts

## Active hybrid-02 Direction

The `hybrid-02` branch is now optimized around these goals:

- replace the visible `Virchow` and `CTransPath` slots with the new roster
- keep the old seven stored in `results-history`
- add richer per-fold and clinical-style metrics directly to the run records
- wire external cohort metadata into the control plane before claiming a new
  best model

The active roster now targets:

- `CONCHv1.5`
- `Phikon-v2`
- `Prov-GigaPath`
- `PRISM`
- `CHIEF`
- `DINOv3`
- `Midnight-12k`

## External Validation Policy

This branch now assumes the following:

1. TCGA cross-validation is only the development score.
2. Any serious leaderboard claim must be repeated on an external cohort.
3. `Run.external_cohorts` stores the requested external cohort configs.
4. `RunApproachLink.external_metrics` stores cohort-specific test outputs after
   sync.

Target external cohorts:

- `CPTAC-COAD`
- `DACHS`
- `PAIP`

The intended comparison rule is simple:

- train on `TCGA`
- test on the external cohort
- choose the winner from external metrics, not from the TCGA-only mean

## Core Runtime Files

### Project settings

- [msi_platform/settings.py](./msi_platform/settings.py)

Holds:

- environment loading
- installed apps
- static config
- VM connection config
- Hugging Face token config
- n8n config
- default approach definitions

### Dashboard UI

- [apps/core/templates/core/base.html](./apps/core/templates/core/base.html)
- [apps/core/templates/core/dashboard.html](./apps/core/templates/core/dashboard.html)
- [apps/core/templates/core/partials/live_runs_panel.html](./apps/core/templates/core/partials/live_runs_panel.html)
- [apps/core/templates/core/partials/metrics_panel.html](./apps/core/templates/core/partials/metrics_panel.html)
- [apps/core/static/core/app.css](./apps/core/static/core/app.css)

### Run orchestration

- [apps/runs/services.py](./apps/runs/services.py)
- [apps/runs/vm_runtime.py](./apps/runs/vm_runtime.py)
- [apps/runs/api_views.py](./apps/runs/api_views.py)
- [apps/runs/api_urls.py](./apps/runs/api_urls.py)

### VM helpers

- [apps/vm/services.py](./apps/vm/services.py)
- [apps/vm/registry.py](./apps/vm/registry.py)
- [apps/vm/models.py](./apps/vm/models.py)

### n8n automation

- [automation/n8n/msi_django_launch.json](./automation/n8n/msi_django_launch.json)
- [automation/n8n/start_n8n_vm.sh](./automation/n8n/start_n8n_vm.sh)
- [automation/n8n/import_workflow_vm.sh](./automation/n8n/import_workflow_vm.sh)

## API Surface

### Health and runtime

- `GET /api/health/`
- `GET /api/vm/status/`

### Runs

- `GET /api/runs/`
- `POST /api/runs/<run_id>/launch-vm/`
- `POST /api/runs/<run_id>/sync-status/`

### Archives

- `POST /api/archives/sync-latest/`

These APIs are intentionally small. The main control surface is the
Django UI, with API routes supporting automation, debugging, and remote ops.

## Frontend Philosophy

The frontend is intentionally integrated, not separated.

We chose:

- Django templates over a separate SPA
- HTMX over a large client-side state layer
- Alpine.js for tiny interactive needs
- Plotly for Python-first chart rendering

Why this works well here:

- faster iteration for internal tools
- easier deployment on the VM
- simpler synchronization between backend truth and UI
- less friction for a pathology workflow where most complexity is backend and
  runner-side

This is a good fit when the UI exists to control and observe experiments rather
than to serve as a public consumer web app.

## Environment Variables

The app loads `.env` at startup.

Important values:

```env
HF_TOKEN=

N8N_ENABLED=true
N8N_BASE_URL=http://127.0.0.1:5678
N8N_WEBHOOK_SECRET=
N8N_WORKFLOW_PATH=automation/n8n/msi_django_launch.json

VM_EXECUTION_MODE=ssh
VM_NAME=pathology310-primary
VM_SSH_USER=pardeep
VM_SSH_HOST=34.59.145.240
VM_SSH_KEY_PATH=C:\Users\<you>\.ssh\evolet_rsa
VM_CONDA_ENV=pathology310
VM_RUNNER_PYTHON=/home/pardeep/.venvs/pathology310-hybrid/bin/python
VM_PROJECT_ROOT=/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc
VM_RUNNER_SCRIPT=/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/scripts/run_tcga_coad_automated_triad.py
VM_DEFAULT_ANNOTATIONS=annotations/tcga_coad_bucket_annotations_final_all3_live_dx1.csv
VM_VIRCHOW_WEIGHTS=/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/models/virchow/pytorch_model.bin
```

See [.env.example](./.env.example).

## Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py createsuperuser
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/admin/`
- `http://127.0.0.1:8000/api/health/`

## VM Deployment Notes

The current Django control plane is deployed to the pathology VM and validated
there.

The live dashboard endpoint used during validation:

- `http://34.59.145.240:8000/`

Important note:

- this repository still runs with `DEBUG=True` and SQLite
- it is production-lean for an internal research platform, not yet a hardened
  internet-facing deployment

## Engineering Decisions We Made During The Build

### 1. Integrated frontend instead of separate web app

This reduced complexity and made live status rendering much easier.

### 2. Modular app boundaries

We separated UI, runs, VM, approaches, and archives so future scaling does not
collapse into one giant `views.py` architecture.

### 3. Multi-approach run model

Instead of one model = one run, the system lets a single run branch into
multiple extractor/model configurations through `RunApproachLink`.

### 4. Remote artifacts as source-of-truth inputs

We do not fake metrics in the UI. Metrics are read from the VM-side generated
files and normalized into Django.

### 5. Real runner compatibility over perfect abstraction

The system was adapted to the actual pathology runner contract, including remote
JSON files, bundle roots, and VM-side script realities.

## Current Limits

- database is SQLite, not PostgreSQL yet
- no job queue like Celery or Temporal yet
- remote sync is polling-driven, not event-driven
- no authenticated role model or multi-user permissions yet
- archive analytics are still early
- deployment still uses Django `runserver` in the current VM validation path

## Best Next Steps

### Platform

1. Move from SQLite to PostgreSQL.
2. Add a dedicated production process manager and reverse proxy.
3. Add real auth and role-based access control.

### ML orchestration

1. Add a first-class run detail page with fold-level artifacts.
2. Add true external cohort execution paths beside the new metadata fields.
3. Add callback-based sync from n8n or the runner instead of pure polling.

### Research workflow

1. Add configurable approach presets from the UI.
2. Add archive comparison and leaderboard views.
3. Add upload-to-predict workflow paths beside train-time orchestration.

## Current Summary

The current platform delivers:

- integrated production-lean dashboard
- real VM launch and sync endpoints
- actual TCGA bucket + annotation selection
- preserved legacy 7-approach baseline history
- active `hybrid-02` seven-model roster defaults
- richer synchronized metrics including AUPRC, balanced accuracy, MSI-H recall,
  specificity, confusion counts, fold variance, calibration, and external
  cohort placeholders
- synchronized completed metrics on the frontend

This gives the project a strong base for deeper experiment analytics,
production hardening, archive comparison, and future expansion into additional
foundation models.

## Latest Completed Run (May 7, 2026)

Most recent full run on VM:

- run id: `run-69fb62874c`
- state: `completed`
- selected slides: `180`
- label mix: `74 MSI-H`, `106 MSS`
- best approach: `Approach3-Prov-GigaPath`

Per-approach metrics:

| Approach | Extractor used | AUROC | F1 macro | AUPRC | Bal Acc | MSI-H Recall | Specificity | State |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Approach1-CONCHv1.5 | `phikon-v2` | `0.9243` | `0.9066` | `0.9363` | `0.9083` | `0.9255` | `0.8911` | `completed` |
| Approach2-Phikon-v2 | `phikon-v2` | `0.9422` | `0.9289` | `0.9489` | `0.9243` | `0.9718` | `0.8768` | `completed` |
| Approach3-Prov-GigaPath | `prov-gigapath` | `0.9555` | `0.9302` | `0.9692` | `0.9320` | `0.9336` | `0.9304` | `completed` |
| Approach4-PRISM | `prism-virchow` | `0.9402` | `0.9044` | `0.9603` | `0.9008` | `0.9427` | `0.8589` | `completed` |
| Approach5-CHIEF | `phikon-v2` | `0.9297` | `0.8948` | `0.9504` | `0.8966` | `0.9164` | `0.8768` | `completed` |
| Approach6-DINOv3 | `phikon-v2` | `0.9315` | `0.9018` | `0.9487` | `0.9000` | `0.9518` | `0.8482` | `completed` |
| Approach7-Midnight-12k | `midnight` | `0.9535` | `0.9394` | `0.9632` | `0.9390` | `0.9636` | `0.9143` | `completed` |

External cohort note:

- run config requested `CPTAC-COAD`, `DACHS`, and `PAIP`
- at runtime, those cohort annotation files were not present on VM, so these
  results are still TCGA-only for now

## Semi-Final Branch Addendum - Fresh 200-Slide Foundation Run

### Timestamp

- local documentation time: `2026-05-07 09:10:51 +05:30`
- VM launch day: `May 7, 2026`
- active branch intent: `semi-final`
- active VM run id: `run-8635c038adcc`
- active VM process id at launch: `3146389`

This section is an additive research log. It does not replace the earlier
`hybrid-02` notes or the completed `run-69fb62874c` metrics above. Those remain
important because they show the last completed TCGA-only baseline before the
strict semi-final rerun.

### Why We Made This Update

The previous completed run was useful, but it was not yet defensible as a final
research benchmark for three reasons:

1. Some requested approaches silently resolved to another extractor.
2. External validation was configured as metadata, but CPTAC-COAD annotations
   were not yet present on the VM.
3. The VM contained old run artifacts, which made it harder to reason about a
   clean fresh experiment state.

The semi-final update fixes those gaps by making the run stricter, cleaner, and
better documented.

### What Was Changed

#### 1. Clean Fresh VM State

Old generated VM artifacts were removed before the new launch.

Cleaned generated paths:

```text
/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/automation/tcga_slide_triads
/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/bundle_configs
/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/launch_logs
/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/run_status
/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/tmp
```

Space recovery:

| Item | Before | After |
| --- | ---: | ---: |
| `tcga_slide_triads` generated runs | `153 GB` | `4 KB` |
| VM free disk after cleanup | about `112 GB` | about `265 GB` |

Docker/build cleanup was also performed because the CHIEF container pull needed
more clean disk headroom.

#### 2. Slide Count Increased To 200

The new semi-final run uses:

| Setting | Value |
| --- | --- |
| requested slides | `200` |
| folds | `10` |
| repeats | `1` |
| max tiles per slide | `256` |
| max parallel approaches | `3` |
| generic fallback | disabled |

The reason for increasing to `200` slides is to reduce small-cohort noise while
still staying practical on the L4 24 GB GPU and 32 GB RAM VM.

#### 3. Controlled Parallelism

The VM is configured for controlled parallel execution:

```env
VM_MAX_PARALLEL_APPROACHES=3
```

This gives faster throughput than fully sequential execution, but avoids trying
to run all foundation extractors together on one L4 GPU. It is the safer middle
point for this VM.

#### 4. Strict Extractor Resolution

The runner was changed so each approach uses its own requested extractor. If an
extractor cannot load, that approach should fail clearly instead of silently
falling back to Phikon, CTransPath, ResNet, or another proxy.

This matters because the leaderboard should answer: "How did this foundation
model perform?" not "Which fallback happened to run?"

### Final Semi-Final Roster

The semi-final launch now uses ten strict approaches. DINOv2-Large is kept as
an open comparator, DINOv3 ViT-B/16 remains the gated Meta model, and RetCCL is
added back in as a native Slideflow comparator beside the hybrid extractor set.

| Slot | Approach | Source | Access | Embedding dim | Tile size | Notes |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | UNI2-h | `MahmoodLab/UNI2-h` | gated | `1536` | `224 x 224` | strongest prior legacy performer |
| 2 | Virchow2 | `paige-ai/Virchow2` | gated | `2560` | `224 x 224` | strong CRC/pathology foundation encoder |
| 3 | Prov-GigaPath | `prov-gigapath/prov-gigapath` | gated | `1536` | `256 x 256` | best true hybrid-02 performer |
| 4 | CONCHv1.5 | `MahmoodLab/conchv1_5` | gated | `768` | model transform driven | fixed to run as CONCH, not Phikon fallback |
| 5 | H-Optimus-0 | `bioptimus/H-optimus-0` | open | `1536` | `224 x 224` | proven legacy top performer |
| 6 | Midnight-12k | `kaiko-ai/midnight` | open | `1536` | `224 x 224` | strong hybrid-02 result and high F1 macro |
| 7 | DINOv2-Large | `facebook/dinov2-large` | open | `1024` | `224 x 224` | open Meta baseline kept for comparison |
| 8 | DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | gated | `768` | `224 x 224` | added after access was confirmed |
| 9 | CHIEF | `github.com/hms-dbmi/CHIEF` + official Docker weights | request/docker | `768` | `224 x 224` | now uses real CHIEF CTransPath weights |
| 10 | RetCCL | `retccl` | slideflow/native | `-` | Slideflow default | legacy strong comparator restored into the strict semi-final run |

### CHIEF Fix

CHIEF is no longer represented by a `chief-ctranspath -> ctranspath` proxy.

What was done:

- cloned/updated the official CHIEF GitHub repository on the VM
- pulled the official CHIEF Docker image after cleaning disk
- extracted model weights from the container
- placed the CTransPath checkpoint at the strict runner path

Verified checkpoint:

```text
/home/pardeep/models/CHIEF/model_weight/CHIEF_CTransPath.pth
```

Observed size:

```text
107 MB
```

The strict CHIEF loader now expects this file. If it is missing in the future,
CHIEF should fail clearly instead of using a substitute extractor.

### DINOv3 Fix

DINOv3 was added after access was confirmed for:

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
```

The model is loaded through Hugging Face Transformers with the configured token.
It is now a separate approach instead of replacing DINOv2-Large.

### CPTAC-COAD External Cohort Preparation

CPTAC-COAD annotations were fetched from cBioPortal study:

```text
coad_cptac_2019
```

Generated files:

```text
runtime/annotations/cptac_coad_annotations.csv
runtime/annotations/cptac_coad_clinical_sample_long.csv
```

VM path used by the semi-final run:

```text
/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/cptac_coad_annotations.csv
```

CPTAC-COAD label counts:

| Label | Count |
| --- | ---: |
| `MSI-H` | `24` |
| `MSS` | `81` |
| total labeled samples | `105` |

Important note: the annotations are now present and attached to the run config.
The actual external validation still depends on the runner finding compatible
CPTAC slide/image inputs for those sample IDs.

### Active Fresh Run

The fresh run launched from the Django control plane with this experiment name:

```text
tcga3-semi-final-200x10f-uni2h-virchow2-gigapath-conch15-hoptimus-midnight-dinov2large-dinov3vitb16-chief-256tiles
```

Run paths:

```text
status: /home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/automation/tcga_slide_triads/run-8635c038adcc/status.json
log:    /home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/launch_logs/run-8635c038adcc.log
config: /home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/bundle_configs/run-8635c038adcc.json
```

Initial selected TCGA cohort:

| Field | Value |
| --- | ---: |
| matched TCGA DX slides | `426` |
| selected slides | `200` |
| MSI-H selected | `74` |
| MSS selected | `126` |

Latest live VM status observed for the semi-final run:

| Field | Value |
| --- | --- |
| state | `failed` |
| downloaded SVS slides | `200 / 200` |
| downloaded slide size | `83.73 GB` |
| TFRecord-backed selected slides | `200` |
| completed feature bag sets | `UNI2-h 199 / 199`, `Virchow2 199 / 199`, `Prov-GigaPath 199 / 199` |
| approach outputs | `0` |
| approach metrics | `0` |
| VM disk free | about `118 GB` |
| failure point | `CONCHv1.5` initialization |
| failure reason | gated Hugging Face access denied for `MahmoodLab/conchv1_5` |

Failure excerpt from VM `status.json`:

```text
RuntimeError: Unable to initialize any requested feature extractor.
Tried: {'conchv1_5': 'GatedRepoError: 403 Client Error ... Access to model
MahmoodLab/conchv1_5 is restricted and you are not in the authorized list.'}
```

### What This Gives The Project

This update moves the project from a development benchmark toward a more
defensible semi-final benchmark.

It gives us:

- a clean VM artifact state before launch
- a larger 200-slide TCGA cohort than the previous 180-slide run
- strict no-proxy extractor behavior
- real CHIEF source and checkpoint availability
- DINOv3 ViT-B/16 added as a separate gated approach
- DINOv2-Large retained as an open comparator
- CPTAC-COAD labels present on the VM for external cohort work
- frontend defaults aligned with the actual semi-final roster
- the metric schema needed for AUROC, F1 macro, AUPRC, balanced accuracy,
  MSI-H recall, specificity, precision, Brier score, and best threshold

### Expected Output Metrics

When the run completes, each approach should report:

| Metric | Purpose |
| --- | --- |
| AUROC | ranking/separation quality |
| F1 macro | balanced class-wise F1 |
| AUPRC | precision-recall performance under class imbalance |
| balanced accuracy | average of sensitivity and specificity |
| MSI-H recall | sensitivity for MSI-H detection |
| specificity | MSS true-negative rate |
| precision | positive predictive value |
| Brier score | probability calibration error |
| best threshold | validation-selected operating threshold |

### Current Interpretation

Do not compare this run to the earlier table until it completes. The earlier
`run-69fb62874c` table is completed evidence. The new `run-8635c038adcc` table
is still in progress and should be treated as a live experiment until all
approach `metrics.json` files and the final summary are written.

### Recovery Update - Same-Day Relaunch On May 7, 2026

After the first strict semi-final attempt failed at `CONCHv1.5`, the run was
recovered in place instead of starting another large fresh bundle download.

What changed during recovery:

- verified that Hugging Face access to `MahmoodLab/conchv1_5` was no longer the
  main blocker
- identified the real incompatibility as the old `timm` HF-hub path expecting a
  `config.json` layout that this repo does not provide
- patched `vm_patch/hybrid_extractors.py` so `CONCHv1.5` downloads
  `pytorch_model_vision.bin`, strips the `trunk.` prefix, builds the matching
  ViT backbone manually, and loads weights without the old hub-config path
- validated the patched manual loader against the real VM weights with no
  missing or unexpected load keys
- synced the patched extractor file to the VM script path
- added `RetCCL` back as `Approach 10`
- raised `VM_MAX_PARALLEL_APPROACHES` from `2` to `3`

Why the same bundle was reused:

- the existing failed bundle root was already about `87 GB`
- reusing `run-8635c038adcc` avoided another wasteful full TCGA slide download

Relaunch details:

- same bundle id: `run-8635c038adcc`
- relaunch log:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/launch_logs/run-8635c038adcc-relaunch-10a3.log`
- relaunched process id observed: `3980754`
- active extractor plan:
  `uni2-h, virchow2, prov-gigapath, conchv1_5, h-optimus-0, midnight, dinov2-large, dinov3-vitb16, chief, retccl`

Latest post-relaunch state observed:

| Field | Value |
| --- | --- |
| state | `extracting_tiles` |
| downloaded slides | `200 / 200` |
| selected slides | `200` |
| TFRecords observed | about `110 / 200` |
| approach status files | `0` |
| approach metrics files | `0` |
| GPU | mostly idle during extraction |
| CPU | busy during extraction |

Low-tile note still being tracked:

- slide `TCGA-A6-2675-01Z-00-DX1.d37847d6-c17f-44b9-b90a-84cd1946c8ab` again
  produced only `13` extracted tiles
- this is important to watch, but it was not the cause of the relaunch stop

Current interpretation after recovery:

- the earlier failed snapshot is still historically important because it records
  the original `CONCHv1.5` stop point
- the active semi-final run is now the relaunched `10`-approach,
  `parallel=3` recovery on the same bundle root
- this rerun had not yet reached `generating_features`, `CONCHv1.5` execution
  in the new pass, or MIL training at the time of the latest note
