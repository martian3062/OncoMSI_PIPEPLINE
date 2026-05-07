# VM Last Hour Record

Date: `2026-05-07`
Scope: last roughly 1 hour of work on the semi-final TCGA MSI VM run
Primary run id: `run-8635c038adcc`
Branch intent: `semi-final`

## Summary

During the last hour, the original semi-final run was inspected, its failure
point was identified, access to `CONCHv1.5` was rechecked, the hybrid loader
path for `CONCHv1.5` was patched, the active run configuration was expanded to
`10` approaches with `RetCCL` added, parallel approaches were raised to `3`,
and the VM run was relaunched on the existing bundle root to avoid duplicating
another large slide download.

The rerun is currently active and is still in tile extraction.

## What Happened In The Last Hour

### 1. We checked the failed semi-final run

The older active bundle was:

- run id: `run-8635c038adcc`
- selected slides: `200`
- completed feature bag sets before failure:
  - `UNI2-h`: `199 / 199`
  - `Virchow2`: `199 / 199`
  - `Prov-GigaPath`: `199 / 199`

The run did not fail because of GPU availability or a general Slideflow crash.
It failed when it reached `CONCHv1.5`.

### 2. We verified the original failure reason

The original VM `status.json` failure was:

- state: `failed`
- failure point: `CONCHv1.5`
- earlier error class: gated Hugging Face access problem

That older failure was documented in `README.md`.

### 3. We rechecked Hugging Face access

After access changes, the repo became reachable:

- HF API for `MahmoodLab/conchv1_5` returned `200`
- visible files included:
  - `.gitattributes`
  - `README.md`
  - `meta.yaml`
  - `pytorch_model_vision.bin`

This showed that access was no longer the main blocker.

### 4. We found the new real blocker

The new blocker was not permission. It was loader compatibility.

The repo did not expose the `config.json` path that the existing `timm`
HF-hub loader expected. That caused a `404 Entry Not Found` when the old
`_build_conchv1_5(...)` path tried to build the model through `timm`'s
standard HF config flow.

### 5. We patched the local hybrid extractor path

We updated `vm_patch/hybrid_extractors.py` so `CONCHv1.5` is loaded through a
manual weight path instead of assuming a `config.json`-based `timm` repo.

The patched path now:

- downloads `pytorch_model_vision.bin`
- strips the `trunk.` prefix from the stored state dict
- builds a `vit_large_patch16_224` style backbone with the matching shape
- loads weights manually
- applies an explicit image transform pipeline

### 6. We validated the manual `CONCHv1.5` weight path

We tested the manual weight-loading logic against the real VM weights and it
passed shape loading checks:

- missing keys: `[]`
- unexpected keys: `[]`

This confirmed the manual load strategy is valid for the repo's actual file
layout.

### 7. We synced the loader patch to the VM

The patched `hybrid_extractors.py` was copied to the VM script path:

`/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/scripts/hybrid_extractors.py`

### 8. We expanded the active configuration

The active semi-final defaults were updated to:

- add `RetCCL` back as `Approach 10`
- treat `RetCCL` as a `slideflow`/native comparator
- increase `VM_MAX_PARALLEL_APPROACHES` from `2` to `3`

The dashboard launch defaults and README were also updated to match the new
`10`-approach plan.

### 9. We avoided a wasteful full new bundle download

The existing failed bundle root was already about `87G`.

To avoid wasting disk and redownloading another large slide set, we reused the
same bundle id/root and updated its config in place before relaunching.

### 10. We relaunched the run

The rerun was launched with:

- same bundle id: `run-8635c038adcc`
- new log:
  `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/launch_logs/run-8635c038adcc-relaunch-10a3.log`
- new process id observed:
  `3980754`

The updated config included:

- `10` approaches
- `parallel=3`
- extractors:
  `uni2-h, virchow2, prov-gigapath, conchv1_5, h-optimus-0, midnight, dinov2-large, dinov3-vitb16, chief, retccl`

## What The VM Is Doing Now

Current state at the latest check:

- state: `extracting_tiles`
- downloaded slides: `200`
- selected slides: `200`
- TFRecords observed: about `110 / 200`
- approach status files: `0`
- approach metrics files: `0`
- GPU: mostly idle during extraction, which matches this stage
- CPU: busy, which also matches this stage

The rerun is healthy and still moving.

## Important Observations

### Low-tile slide behavior still exists

The same slide continued to show low extracted tile count:

- `TCGA-A6-2675-01Z-00-DX1.d37847d6-c17f-44b9-b90a-84cd1946c8ab`

Observed extraction detail:

- only `13` tiles extracted for that slide

This is consistent with the earlier missing-bag pattern and should still be
tracked, but it is not what caused the latest rerun to stop.

### The rerun has not reached feature generation yet

The `10`-approach rerun is still in tile extraction, so these milestones have
not happened yet:

- `generating_features`
- `CONCHv1.5` start in the new rerun
- actual MIL training launch

## Current Best ETA

Based on the latest observed extraction progress:

- finish tile extraction: about `12-18 minutes` from the last check
- enter `generating_features`: immediately after extraction completes
- reach `CONCHv1.5`: likely later in the extractor sequence
- actual MIL training launch: still later after extractor/bag generation

These later ETAs are lower-confidence than the extraction ETA.

## Files Changed In This Window

- `vm_patch/hybrid_extractors.py`
- `msi_platform/settings.py`
- `apps/core/views.py`
- `apps/core/templates/core/dashboard.html`
- `README.md`

## Short Conclusion

In the last hour, we moved from:

- a failed `9`-approach semi-final bundle blocked at `CONCHv1.5`

to:

- a relaunched `10`-approach `parallel=3` rerun with a patched `CONCHv1.5`
  loader and `RetCCL` added back into the experiment plan

The rerun is currently active and still extracting tiles.
