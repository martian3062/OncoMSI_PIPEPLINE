# Hybrid Pipeline Architecture

This document defines the target hybrid MSI pathology pipeline for OncoMSI.

The goal is simple:

- keep `Django` as the control plane
- keep `Slideflow` only for the parts it already handles well
- move new foundation-model integration away from the `Slideflow` extractor
  registry
- make feature extraction, MIL training, experiment tracking, and statistics
  modular
- support multiple model families in parallel without pretending one framework
  should own everything

## Why Hybrid Instead Of Single-Framework

The current bottleneck is not the dashboard, VM launch, folds, or metric sync.
The bottleneck is the extractor onboarding layer.

Today, the working VM stack can launch:

- `virchow`
- `retccl`
- `ctranspath`

But newer requested models are blocked because the installed `Slideflow`
environment does not recognize their extractor names directly.

That means:

- `Slideflow` is fine as a stable pathology workflow engine
- `Slideflow` is not the right single source of truth for all future extractor
  onboarding

So the design should separate:

1. control plane
2. WSI I/O and tiling
3. feature extraction
4. feature storage
5. MIL training
6. search / tracking / statistics

## Core Principle

Use each framework only where it is strongest.

Do not force:

- `Slideflow` to be the only extractor integration layer
- `CLAM` to be the whole orchestration stack
- `MONAI` to become the pathology product shell
- `TRIDENT` to replace the Django app

Instead:

- `Django` owns runs, UI, orchestration, history, and APIs
- extraction backends become swappable
- MIL trainers become swappable
- storage stays framework-agnostic

## Recommended Division Of Responsibilities

### 1. Control Plane

Primary stack:

- `Django`
- `Django REST Framework`
- `HTMX`
- `Alpine.js`
- `Plotly`

Responsibilities:

- run creation
- run history
- archive/history tabs
- bundle config generation
- live status sync
- remote launch/sync APIs
- experiment comparison UI
- n8n hooks

This part should remain the long-lived product shell.

### 2. WSI Reading And Tiling

Primary stack:

- `openslide-python`
- `slideflow`
- `tiatoolbox`
- optional `cucim`
- optional `lazyslide`

Recommended roles:

- `OpenSlide`: lowest-level `.svs` reading compatibility
- `cuCIM`: optional accelerated image access where supported
- `TIAToolbox`: robust pathology preprocessing utilities
- `LazySlide`: large-slide data handling and preprocessing support
- `Slideflow`: keep for legacy/stable tiling flows already validated in this VM

Design rule:

- if the current `Slideflow` tiling path is stable for TCGA COAD, keep it
- if a new extractor needs a different preprocessing path, allow a non-Slideflow
  tile/patch backend to produce the same canonical tile outputs

### 3. Feature Extraction

Primary stack:

- `torch`
- `timm`
- `transformers`
- `huggingface_hub`
- `safetensors`

Optional pathology wrappers:

- `TRIDENT`
- model-specific local adapters

This is the most important separation in the hybrid design.

Instead of:

- `Slideflow build_feature_extractor(name)` as the only entry point

Use:

- a framework-agnostic extractor adapter interface

Example target contract:

```text
TileBatch -> ExtractorAdapter -> EmbeddingBatch
```

Each adapter should define:

- model name
- model source
- HF repo or local weights path
- required transforms
- normalization values
- tile input size
- output embedding dimension
- pooling strategy
- gated-token requirement

That lets us support:

- `Virchow`
- `Virchow2`
- `RetCCL`
- `CTransPath`
- `CONCH`
- `UNI2-H`
- `H-optimus-0`
- `H-optimus-1`
- future backbones

without depending on `Slideflow` to know every model by name.

### 4. Feature Storage

Primary stack:

- `h5py`
- `zarr`
- `pyarrow`
- `numpy`
- `pandas`
- `polars`
- `duckdb`

Recommended storage split:

- `zarr` or `h5` for tile embedding arrays
- `parquet` for tile / slide / fold metadata
- `duckdb` for fast experiment-level queries and comparisons

Design rule:

- do not make feature storage depend on a single trainer framework

This allows:

- reuse of one extractor output across many MIL models
- faster ablation studies
- easier archive/history preservation

### 5. MIL Training

Primary stack:

- `pytorch-lightning`
- `torchmetrics`
- `scikit-learn`
- `einops`

Candidate model frameworks:

- internal `TransMIL`
- `CLAM` / `CLAM2`
- attention MIL variants
- custom Lightning modules

Recommended role split:

- `Lightning` owns training loops, callbacks, checkpoints, resume behavior,
  mixed precision, logging hooks, device movement
- `TorchMetrics` owns AUROC, F1, precision/recall, calibration-friendly metrics
- `CLAM` is used as a model family or baseline, not as the whole platform
- `TransMIL` remains a strong current baseline for MSI-H vs MSS

Design rule:

- trainers should consume canonical precomputed embeddings, not re-own WSI I/O

### 6. Search / Config / Tracking

Primary stack:

- `hydra-core`
- `omegaconf`
- `optuna`
- `hydra-optuna-sweeper`
- `mlflow`
- optional `tensorboard`
- optional `wandb`

Recommended role split:

- `Hydra` for config composition
- `Optuna` for search
- `MLflow` for params, metrics, artifacts, and best-run comparison

This should become the new experiment backbone for:

- seeds
- folds
- bag size variants
- learning rate sweeps
- extractor comparisons
- stability ranking

### 7. Statistical Validation

Primary stack:

- `scipy`
- `statsmodels`
- `scikit-learn`
- `torchmetrics`

Recommended outputs:

- bootstrap confidence intervals
- per-fold variance
- AUROC comparison
- F1 stability
- Brier score
- calibration analysis

This is important because a better extractor is not enough. We need a better
and more stable system.

### 8. Testing And Data Contracts

Primary stack:

- `pytest`
- `pytest-cov`
- `hypothesis`
- `ruff`
- `mypy`
- `pre-commit`
- `pandera`

Recommended validation areas:

- annotation schema checks
- bucket-match logic
- fold split reproducibility
- embedding-shape checks
- model registry validity
- archive/history persistence

## What Slideflow Should Keep Owning

Keep `Slideflow` only where it already helps and is stable.

Recommended retained responsibilities:

- legacy TCGA COAD tiling path already validated on the VM
- project/dataset convenience where it already works
- currently supported extractors that are known-good in this environment
- stable pathology preprocessing flows already integrated into the runner

Good current use cases:

- `virchow`
- `retccl`
- `ctranspath`
- stable baseline experiments

## What Should Move Away From Slideflow

Move these responsibilities into the hybrid layer:

- all new foundation-model adapters
- gated HF model loading logic
- weight/config resolution
- preprocessing adapters per backbone
- feature extraction orchestration for unsupported models
- unified embedding serialization

This is the main architectural shift.

## Framework Matrix

### Slideflow

Best at:

- stable digital pathology workflows
- known extractor support
- current legacy runner compatibility

Weak at:

- rapid arbitrary backbone onboarding
- newest pathology model support by name

### TRIDENT

Best at:

- pathology foundation-model workflows
- WSI processing and large-scale representation extraction

Use it for:

- extraction backend candidates
- modern pathology FM pipelines

### LazySlide

Best at:

- WSI handling and preprocessing support
- large-slide data workflows

Use it for:

- preprocessing / data backend support
- not as the product shell

### CLAM / CLAM2

Best at:

- MIL model family / weakly supervised baseline

Use it for:

- trainer/model options
- not full orchestration

### MONAI

Best at:

- training infrastructure
- transforms
- medical imaging ecosystem components

Use it for:

- reusable transforms or modules where helpful
- not as the main pathology control plane

### PRISM

Treat as:

- optional specialist integration depending on the exact pathology task or
  representation workflow

Do not make it the center of the system without a concrete fit to the MSI
pipeline.

### STAMP

Best at:

- reproducible pathology biomarker workflow patterns
- benchmark-style weak supervision pipelines

Use it for:

- ideas, baselines, training/evaluation structure
- not as a replacement for the Django control plane

## Parallelism Strategy

Yes, the hybrid design should explicitly support parallelism.

### Parallelism levels

#### Level 1: Data preparation parallelism

- parallel slide download
- parallel tiling
- parallel stain/QC preprocessing

#### Level 2: Feature extraction parallelism

- parallel extractor jobs per backbone
- parallel tile batches within one backbone

#### Level 3: MIL training parallelism

- parallel approach runs
- parallel folds where GPU budget allows
- parallel seed restarts

#### Level 4: Experiment orchestration parallelism

- one control plane
- many remote workers
- model-specific job routing

## Multi-Architecture Support

The hybrid design must be multi-architectural by default.

That means it should support:

- multiple extractor families
- multiple MIL heads
- multiple storage backends
- multiple training recipes

Canonical design:

```text
WSI source
  -> tile backend
  -> extractor adapter
  -> feature store
  -> MIL trainer
  -> metrics/statistics
  -> archive/history
```

Each box should be replaceable without breaking the whole system.

## Recommended Phase Migration

### Phase A

Keep current working stack:

- Django
- current VM launcher
- current Slideflow stable extractors

Add:

- history-preserving run records
- framework-agnostic extractor registry

### Phase B

Introduce hybrid extraction path:

- `Virchow2`
- `CONCH`
- `UNI2-H`
- `H-optimus-0`

without forcing `Slideflow` to instantiate them by name

### Phase C

Move MIL training into Lightning-based modular trainers while still allowing
legacy Slideflow-supported runs to coexist.

### Phase D

Promote MLflow + Hydra + Optuna as the experiment backbone and let the Django
dashboard consume those results.

## Immediate Recommended Stack

For your MSI-H vs MSS WSI pipeline, the best immediate hybrid shape is:

### Core training stack

- WSI reading / tiling:
  - `openslide-python`
  - `slideflow`
  - `tiatoolbox`
  - optional `cucim`
  - optional `lazyslide`
- image preprocessing:
  - `opencv-python`
  - `scikit-image`
  - `albumentations`
  - `torchvision`
  - `kornia`
- feature extraction:
  - `torch`
  - `timm`
  - `transformers`
  - `huggingface_hub`
  - `safetensors`
- feature storage:
  - `h5py`
  - `zarr`
  - `pyarrow`
  - `numpy`
  - `pandas`
  - `polars`
  - `duckdb`
- MIL training:
  - `pytorch-lightning`
  - `torchmetrics`
  - `scikit-learn`
  - `einops`
- hyperparameter search:
  - `hydra-core`
  - `omegaconf`
  - `optuna`
  - `hydra-optuna-sweeper`
- experiment tracking:
  - `mlflow`
  - optional `tensorboard`
  - optional `wandb`
- statistical validation:
  - `scipy`
  - `statsmodels`
  - `scikit-learn`
  - `torchmetrics`
- testing / quality:
  - `pytest`
  - `pytest-cov`
  - `hypothesis`
  - `ruff`
  - `mypy`
  - `pre-commit`
  - `pandera`

## Bottom Line

The hybrid answer is not:

- replace everything with one new framework

The hybrid answer is:

- keep `Django` for orchestration and product surface
- keep `Slideflow` only for stable validated legacy paths
- use framework-agnostic extractor adapters for newer foundation models
- use Lightning-based MIL trainers and proper experiment tracking around them
- preserve history, archives, and parallel experiment control from the same app
