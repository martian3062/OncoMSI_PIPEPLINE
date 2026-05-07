# Why Slideflow Crashed for This Project

This file explains, in depth, why the original Slideflow-first design stopped being enough for this MSI pathology workflow, what specifically broke in practice, and what we built to work around those limits without throwing away the parts that still work well.

The short version is:

- Slideflow was still useful for parts of the pipeline.
- Slideflow was not flexible enough for rapid onboarding of newer pathology foundation models.
- The real failures were not only "training bugs." They were mostly integration-surface, environment, and recovery-state problems.
- We solved that by moving to a hybrid architecture instead of a pure Slideflow architecture.

## Quick Comparison

| Aspect | Slideflow Alone | Our Hybrid |
| --- | --- | --- |
| Core idea | One framework handles most of the pathology flow | Keep `Slideflow` for stable pathology pieces, add custom adapters and orchestration around it |
| Best use case | Stable, already-supported extractors and standard MIL workflows | Mixed modern foundation models, gated HF models, live VM runs, recovery-aware experimentation |
| Extractor onboarding | Narrower and more framework-dependent | Much more flexible through `vm_patch/hybrid_extractors.py` |
| New model support | Harder when the model needs custom transforms, pooling, or loading | Custom per-model builders for `UNI2-H`, `Virchow2`, `Prov-GigaPath`, `CONCH`, `DINOv3`, `CHIEF`, and others |
| Hugging Face gated models | More awkward | Explicit token-aware loading built in |
| Output-shape normalization | Limited by the native path | Custom wrappers enforce consistent embedding outputs |
| Fallback control | Easier to drift into silent proxy behavior | Much stricter, clearer "use requested extractor or fail" behavior |
| WSI / tiling / TFRecords | Strong | Still uses `Slideflow` for these strong parts |
| MIL training entrypoints | Strong | Still uses `Slideflow` MIL entrypoints, but with stronger bag filtering and orchestration |
| Recovery after partial failures | Weaker bundle truth | Better per-approach recovery and resync behavior |
| Frontend truthfulness | Can lag behind stale bundle summaries | Django sync reads live/per-approach truth and VM runtime state |
| Storage awareness | Not enough by itself for large multi-model runs | Added cleanup, archiving, bag filtering, and safer execution patterns |
| VM orchestration | Not really a control-plane solution | Integrated with Django plus VM launch/sync flow |
| Dashboard / UI integration | Not built for that by itself | Full control-plane/dashboard integration |
| External cohort metadata flow | Minimal by default | Wired into the run model and sync design |
| Engineering complexity | Simpler if your models fit its rails | More moving parts, but handles this project's needs better |
| Research flexibility | Lower for fast-changing foundation-model work | Higher for experimentation across many pathology encoders |
| Main weakness | Too rigid once the project moved beyond native supported extractors | More custom code to maintain |
| Main strength | Clean pathology core for standard workflows | Practical end-to-end system for modern models, VM orchestration, recovery, and live product-style tracking |

## What The Hybrid Layer Actually Is

In this repo's hybrid extractor layer, `Trident`, `Timestamp`, or some separate
new pathology framework is not what is being used.

What is actually in the layer:

- a custom adapter file: `vm_patch/hybrid_extractors.py`
- it wraps multiple model sources behind one common extractor interface for the
  runner
- it registers those models back into `Slideflow` as custom torch extractors,
  so the rest of the pipeline can still call them in a unified way

What models it currently supports:

- `Virchow2`
- `PRISM` via `Virchow`
- `UNI2-H`
- `H-Optimus-0`
- `CONCH`
- `CONCHv1.5`
- `Phikon-v2`
- `Prov-GigaPath`
- `DINOv2-Large`
- `DINOv3 ViT-B/16`
- `Midnight`
- `CHIEF`

What libraries and framework pieces it uses under the hood:

- `timm` for several pathology vision encoders
- `transformers` for HF-hosted models like `Phikon`, `DINOv2`, `DINOv3`, and
  `Midnight`
- `huggingface_hub` login and token wiring for gated models
- `torch` and `torchvision.transforms`
- optional `CONCH` package from Mahmood Lab GitHub
- official `CHIEF` repo plus local weight loading for `CHIEF`

What the hybrid layer is doing technically:

- normalizes model names and aliases like `dinov3`, `midnight-12k`, and
  `chief-ctranspath`
- decides whether a name should go through the `Slideflow` native backend or
  the hybrid backend
- builds model-specific preprocess transforms
- handles gated Hugging Face auth
- wraps different forward APIs into one common output contract
- applies pooling fixes like `cls`, `mean`, or `cls_mean`
- makes tensor-unfriendly preprocess pipelines work inside the bag-generation
  path
- registers everything into `Slideflow` through `register_torch(...)`

So the important point is:

- not a full replacement framework like `Trident`
- not `Timestamp`
- not a separate inference service

It is a custom adapter layer that lets `Slideflow` keep doing tiling,
`TFRecords`, and `MIL`, while newer foundation models are plugged in through
`timm`, `transformers`, and custom wrappers.

## 1. What Slideflow Was Good At

Slideflow was not a bad choice in general. In this project it was still strong at:

- WSI project structure
- tile extraction
- TFRecord generation
- pathology dataset flow
- MIL training entrypoints
- prediction artifact generation
- fast reuse of already-supported extractors

For models already supported by the VM environment, Slideflow gave us a usable end-to-end path:

- `virchow`
- `retccl`
- `ctranspath`

That means the statement is not "Slideflow is useless."

The real statement is:

**Slideflow is good when the model is already on the rails.**

## 2. Why Slideflow Became Unsuitable Here

The project changed from a simple MSI training stack into a broader benchmark and orchestration system with:

- many extractor families
- gated Hugging Face models
- non-gated pathology models
- VM execution
- recovery after partial failures
- live frontend progress
- history retention
- parallel or queued experiment handling

That is where the weaknesses showed up.

### 2.1 Extractor onboarding was too restrictive

Slideflow expects feature extractors to be constructible through its own extractor-building path.

In practice, that means a model name is not enough.

The framework also needs:

- a loader
- correct weights resolution
- preprocessing rules
- output shape conventions
- embedding pooling behavior
- compatibility with feature bag export

This project quickly moved beyond "supported by name" models.

The models that caused trouble were:

- `CONCH`
- `Virchow2`
- `UNI2-H`
- `H-Optimus-0`
- `Atlas-2`
- `PLUTO-4G`

Some were accessible on Hugging Face, but Slideflow still could not treat them as first-class extractors in this environment.

So the blocker was not just model access.

The blocker was:

**the installed Slideflow runtime did not know how to instantiate and normalize these newer models safely.**

### 2.2 New pathology models move faster than framework support

This repo needed modern pathology foundation models, but framework support lags model releases.

That created several problems:

- models existed on HF but not in the Slideflow registry
- some needed custom transforms
- some needed custom pooling
- some needed gated tokens
- some needed special config/weights handling

Slideflow is opinionated. That is good for stability, but bad for rapid experimental breadth.

### 2.3 The environment stack became fragile

Several failures were not algorithmic failures. They were environment-resolution failures.

Examples we hit:

- Virchow initially needed explicit weights wiring
- `nystrom_attention` missing for `TransMIL`
- `fastai` / plugin import issues
- mixed package resolution between the shared base env and overlay env
- NumPy / OpenCV plugin conflicts
- CuPy warnings from multiple package installations

This meant the real production problem was:

- the pipeline was correct in concept
- the runtime environment was not deterministic enough for repeated recoveries

### 2.4 Recovery after partial success was weak

A major practical problem was that partially completed runs did not recover cleanly.

We had cases where:

- download succeeded
- tiling succeeded
- feature extraction succeeded
- some training branches succeeded
- one branch failed
- final bundle state stayed stale

That made the UI and run status misleading.

Instead of reflecting per-approach truth, the system could get stuck on an old failed bundle summary.

This is one of the biggest reasons a pure Slideflow-centric flow stopped being enough:

**the surrounding orchestration and recovery story mattered as much as the training code itself.**

### 2.5 Storage pressure made "all-at-once" execution unrealistic

The pathology VM has finite disk.

For this project:

- `.svs` slides are large
- extracted tiles are large
- TFRecords are large
- multiple bag directories are large
- multiple extractors multiply feature storage

A large all-parallel run could fill the VM disk long before the science was done.

We saw this directly:

- the VM hit `100%` usage
- downloads stalled
- runs stopped for storage reasons, not modeling reasons

So the system needed storage-aware execution, not just more model support.

## 3. What Actually Broke in Practice

Below are the types of failures we saw.

### 3.1 Unsupported extractor names

Several requested extractors were simply not recognized by the active Slideflow stack.

This was the first major signal that pure Slideflow was too narrow for the benchmark we wanted.

### 3.2 Virchow weights path issues

Even when `virchow` conceptually existed, the runtime still needed explicit weight resolution.

This showed that "recognized model name" did not mean "fully runnable extractor."

### 3.3 CONCH preprocessing contract mismatch

`CONCH` initialized, but its transform path expected PIL-like behavior while the feature bag export path handed it tensors.

That caused:

- tensor vs PIL contract failure
- hybrid bridge patching work

This is a classic example of a model that exists and loads, but still is not integrated cleanly enough for the downstream framework contract.

### 3.4 Virchow2 output-shape failure

Virchow2 needed custom pooling logic.

Without that, the emitted representation could remain token-grid shaped instead of a strict 2D embedding matrix suitable for bag export and MIL training.

That caused:

- invalid bag shapes
- downstream training failure risk
- need for explicit shape assertions and pooled-output normalization

### 3.5 Missing bag artifacts for selected slides

Even after fixes, one selected slide could fail to yield a bag artifact for a specific extractor.

If training blindly assumed all selected slides had bags, that crashed the branch.

So we needed filtered split plans based on actually materialized bag files.

### 3.6 Bundle status drift

When one branch was retried later, the older bundle-level summary could remain stale.

That meant:

- the frontend could still show failed
- even when most branches or all branches were actually completed

This is not a Slideflow-only issue, but it is part of why the framework was no longer enough by itself.

## 4. What We Did Instead

We did not replace everything.

We built a hybrid pipeline.

## 5. Hybrid Strategy

The hybrid strategy was:

- keep Slideflow where it still works well
- replace or wrap the brittle model-onboarding parts
- add stronger orchestration and recovery outside Slideflow

That gave us a more realistic architecture for this project.

## 6. What We Kept from Slideflow

We still use Slideflow for:

- project organization
- tile extraction
- TFRecord-backed pathology preparation
- MIL training entrypoints
- supported extractor handling

So Slideflow still remains an execution engine for the stable parts.

## 7. What We Added to Counter the Problems

### 7.1 Hybrid extractor adapter layer

We created a hybrid extractor layer in:

- `vm_patch/hybrid_extractors.py`

This is one of the most important changes in the repo.

It gives us a controlled place to support models that Slideflow alone was not handling cleanly.

This layer lets us:

- map model names ourselves
- load gated HF models with token support
- customize transforms
- normalize outputs
- enforce expected embedding shapes
- separate "supported by our system" from "natively supported by Slideflow"

This changed the extractor story from:

- "If Slideflow knows it, it works"

to:

- "If our hybrid layer can build it, the runner can use it"

### 7.2 Slideflow backend vs hybrid backend split

We added a clear distinction between:

- `slideflow` backend
- `hybrid` backend

That lets each approach declare how its extractor should be handled.

Examples:

- `virchow`, `retccl`, `ctranspath` can stay closer to Slideflow-native paths
- `conch`, `virchow2`, `uni2-h`, `h-optimus-0` can use hybrid adapters

This separation was critical because it stopped us from forcing every model through the same brittle path.

### 7.3 Deterministic VM runtime

We standardized the runner to prefer a controlled overlay interpreter while still resolving core pathology dependencies from the validated base environment.

This was done to avoid:

- random user-site imports
- package leakage
- accidental replacement of core torch/CUDA dependencies
- plugin import drift between runs

The principle became:

- shared base env for stable pathology stack
- thin hybrid overlay only for additive packages

### 7.4 Stronger output-shape enforcement

For models like Virchow2, we added explicit shape assertions and pooling normalization.

That means extractor outputs now fail early if they do not conform to the contract expected by bag export and MIL training.

This is much better than discovering the mismatch much later inside training.

### 7.5 Recovery-aware finalization

We rewired bundle finalization so it rebuilds the final summary from:

- per-approach `metrics.json`
- per-approach `status.json`

instead of trusting stale bundle-level summaries.

This matters because real experiments often recover in pieces.

Now if:

- 6 branches completed earlier
- 1 branch gets fixed later

the final bundle summary can be rebuilt from truth instead of from history.

### 7.6 Django sync based on per-approach truth

We updated Django sync so the control plane uses per-approach metrics as the primary source of truth.

That means the frontend is now driven by:

- actual branch states
- actual AUROC/F1 values
- actual prediction artifact paths

instead of a single old final-summary snapshot.

### 7.7 Storage-aware execution

This was one of the most important operational countermeasures.

We introduced:

- result archiving to local `results-history`
- deletion of old heavy VM bundles after local preservation
- sequential approach execution for large hybrid runs
- lazy bag generation per approach
- bag cleanup after an approach completes when safe

This directly countered the disk exhaustion problem.

Without this, the full multi-extractor run would be much more likely to die from storage pressure than from modeling issues.

### 7.8 Sequential hybrid execution

For smaller smoke tests, parallelism was okay.

For the full 7-extractor, 150-slide run, it was not the best choice.

So we changed the large hybrid path to:

- run one approach at a time
- keep the selected cohort fixed
- reuse extracted tiles
- generate bags only for the current approach
- optionally clean those bags after completion

This makes the run slower overall than naive full parallelism, but much safer and more realistic on the VM we have.

That is a good tradeoff.

## 8. Why We Did Not Fully Drop Slideflow

There are three main reasons:

### 8.1 It still solves real pathology workflow problems

Slideflow still gives us:

- WSI project handling
- tile extraction
- MIL wiring
- experiment output structure

Rewriting all of that immediately would have slowed the project down.

### 8.2 The actual problem was uneven fit, not total failure

Some parts of Slideflow fit our problem well.

The weakest fit was:

- aggressive foundation-model expansion
- rapid adapter onboarding
- recovery-aware orchestration

So the correct architectural answer was selective replacement, not total replacement.

### 8.3 Hybrid migration is lower risk

A full framework rewrite during active experimentation would have introduced more risk.

The hybrid route lets us:

- preserve working pieces
- keep the control plane stable
- improve only the broken surfaces first

That is the safer engineering move.

## 9. What This Means Architecturally

The system is no longer a pure Slideflow app.

It is now better described as:

**Django control plane + VM orchestration + Slideflow pathology core + hybrid extractor adapters + recovery-aware sync**

That is the real architecture.

## 10. What Other Frameworks Might Do Better

We considered or discussed alternatives because of the extractor-onboarding pain.

Examples:

- TRIDENT
- LazySlide
- CLAM / CLAM2
- MONAI
- STAMP

The reason these came up was not fashion.

It was because the pain had shifted from:

- "Can we train at all?"

to:

- "Can we integrate many modern pathology encoders safely and repeatedly?"

Frameworks like TRIDENT are especially relevant when:

- feature extraction
- patch processing
- embedding generation

become the main source of complexity.

That said, we did not pivot the whole project immediately because the current hybrid path already fixed the critical blocker faster.

## 11. The Most Important Lesson

The biggest lesson from this project is:

**the whole stack matters more than the model name.**

The actual success of the system depended on:

- extractor compatibility
- transform contracts
- output shape normalization
- VM environment determinism
- per-approach recovery
- disk-aware execution
- truthful frontend sync

Slideflow only covered part of that.

Once the project expanded beyond a narrow supported extractor set, the missing parts mattered too much to ignore.

## 12. Final Conclusion

Slideflow was not unsuitable because it was weak at pathology.

It became unsuitable as the **only** foundation because this project demanded:

- broader model onboarding
- stronger environment control
- recovery after partial success
- live UI truthfulness
- storage-aware execution

So we did not abandon it completely.

We changed the architecture around it.

The result is a hybrid system where:

- Slideflow still handles the parts it does well
- our own adapter/runtime/orchestration layers handle the parts it did poorly

That is the reason this repo now looks the way it does.

## 13. Files That Represent This Shift

If you want to inspect the main places where this change happened, start here:

- `vm_patch/hybrid_extractors.py`
- `vm_patch/run_tcga_coad_automated_triad.py`
- `apps/runs/vm_runtime.py`
- `apps/core/views.py`
- `docs/hybrid_pipeline_architecture.md`

Those files together tell the story of how the project moved from a pure Slideflow idea to a hybrid pathology training platform.
