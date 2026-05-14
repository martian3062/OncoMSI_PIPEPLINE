# End Result Clarification

Last updated: `2026-05-09 09:18:10 +05:30`

## Why You Are Seeing Different Results

You asked for a `hybrid` model direction, but this repo now contains **multiple result layers**:

1. old preserved baseline / semi-final results
2. older `hybrid-02` and `hybrid-03` lineup experiments
3. the newer final `Top-4` rerun recorded in `final.md`

So the mismatch is real, not your mistake.

The local app/results page is still reading a preserved local summary file from:

- `archive/local_cleanup_2026-05-08/ten/run-8635c038adcc/final_summary.json`

That file is the older **10-approach semi-final bundle**, not the newer final Top-4 completion log.

## What "Hybrid" Actually Means In This Repo

Here, `hybrid` does **not** mean one single fused ensemble model was produced in the last run.

It means:

- Django control plane + Slideflow stable parts + custom extractor adapters
- many foundation-model extractors can be compared in one system
- older and newer extractors can coexist in the same workflow

So the repo's `hybrid` idea is mainly an **architecture / multi-extractor pipeline**, not one final merged predictor.

## Actual End Result

The newest final run recorded in your last 3 hours is the Top-4 repeated-fold run:

- run id: `run-65512be1f9c4`
- source note: `E:\Cleaned_MSI\final.md`
- state: `completed`
- slides: `200`
- class balance: `74 MSI-H`, `126 MSS`
- protocol: `10 folds x 5 repeats`

### Final Top-4 Ranking

| Rank | Approach | Mean AUROC | Mean F1 Macro | Mean AUPRC | Mean Balanced Acc |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `Approach2-Virchow2` | `0.9718` | `0.9442` | `0.9835` | `0.9454` |
| 2 | `Approach5-H-Optimus-0` | `0.9700` | `0.9368` | `0.9819` | `0.9384` |
| 3 | `Approach1-UNI2-h` | `0.9646` | `0.9362` | `0.9797` | `0.9379` |
| 4 | `Approach6-Midnight-12k` | `0.9511` | `0.9305` | `0.9693` | `0.9330` |

### Winner

The final winner from the latest Top-4 completion is:

- `Approach2-Virchow2`

## Important Difference Between Old And New Results

The older preserved semi-final bundle says:

- bundle id: `run-8635c038adcc`
- approach count: `10`
- repeats: `1`
- best approach: `Approach2-Virchow2`

The newer final Top-4 log says:

- bundle id: `run-65512be1f9c4`
- approach count: `4`
- repeats: `5`
- best approach: `Approach2-Virchow2`

So even though both point to `Virchow2` as winner, the numbers differ because:

- the compared approach set changed
- the repeat count changed
- the evaluation protocol changed
- one source is an older preserved summary and the other is a newer final run log

## Local Files To Trust

### Main explanation

- `E:\Cleaned_MSI\final.md`
- `E:\Cleaned_MSI\README.md`
- `E:\Cleaned_MSI\END_RESULT_LOCAL.md`

### Older preserved semi-final artifacts

- `E:\Cleaned_MSI\archive\local_cleanup_2026-05-08\ten\run-8635c038adcc\final_summary.json`
- `E:\Cleaned_MSI\archive\local_cleanup_2026-05-08\ten\run-8635c038adcc\approaches\*\metrics.json`

### Hybrid-history / preserved result archives

- `E:\Cleaned_MSI\archive\local_cleanup_2026-05-08\final_local_results\iresuts-history\hybrid-03\final_summary.json`
- `E:\Cleaned_MSI\archive\local_cleanup_2026-05-08\final_local_results\results-history\vm-results-history-2026-05-06\`

## Bottom Line

If your question is:

`what is the final end result right now?`

Then the clean answer is:

- the latest final completed run is the Top-4 repeated-fold experiment in `final.md`
- the winner is `Approach2-Virchow2`
- the repo's `hybrid` wording refers to the pipeline architecture, not a single merged ensemble checkpoint
- the reason the visible numbers differ is that the local app is still reading the older preserved `run-8635c038adcc` summary

---

## Prompt Work Log

### `2026-05-09 09:18:10 +05:30`

This section records what was done in the current prompt thread so the app state,
runtime checks, and inference blockers are preserved locally.

### App changes completed in this prompt

- built a real local **Virchow2 feature-bag inference path** in:
  - `apps/core/inference.py`
- wired the Django results route to support upload-and-predict POST handling in:
  - `apps/core/views.py`
- updated the results surface into an upload-first predictor UI in:
  - `apps/core/templates/core/results_beta.html`
  - `apps/core/static/core/app.css`
- extended `requirements.txt` with the local inference/runtime packages needed for the new app path

### What the current app can do now

- upload trusted precomputed feature bags such as `.pt`, `.pth`, `.npy`, `.npz`
- run the saved `Approach2-Virchow2` TransMIL ensemble from:
  - `eraya/latest_approach_virchow2/checkpoints`
- return MSI prediction plus per-checkpoint vote details

### Local run verification completed

- `manage.py check` passed
- synthetic feature-bag inference ran successfully through the local ensemble helper
- Django GET and POST checks succeeded for:
  - `http://127.0.0.1:8010/`
- the local server was started on:
  - `http://127.0.0.1:8010/`

### Raw `.svs` pipeline investigation done in this prompt

- verified that the local Python environment now imports:
  - `slideflow`
  - `openslide`
  - `timm`
  - `PIL`
  - `cv2`
  - `transformers`
- verified that the repo already contains extractor registration and bag-generation logic in:
  - `vm_patch/hybrid_extractors.py`
  - `vm_patch/run_tcga_coad_automated_triad.py`
- confirmed local archived Virchow weights exist at:
  - `archive/local_cleanup_2026-05-08/ten/vm_weights/virchow/pytorch_model.bin`

### Hugging Face token verification done live

The token below was checked live in this prompt:

- `HF_TOKEN=[redacted]`

Verified access responses:

- `paige-ai/Virchow2` -> `200`
- `kaiko-ai/midnight` -> `200`
- `bioptimus/H-optimus-0` -> `200`
- `MahmoodLab/UNI2-h` -> `200`

So the earlier gating failure was not a bad token in principle; it means the extractor path needs to pass/authenticate the token correctly during model construction.

### Current raw-WSI status after all checks

- **feature-bag inference is working locally**
- **true raw `.svs` inference is not finished yet**

Reason:

- the repo has the required Slideflow-style extraction logic
- the local runtime is now much closer to ready
- but the exact app path still needs to be patched so uploaded `.svs` files are converted into a temporary one-slide Slideflow dataset, feature bags are generated with the chosen extractor, and then the correct MIL checkpoints are scored automatically

### Practical fallback status discovered

Among extractors tested directly in this prompt:

- `midnight` initialized successfully on this machine
- `virchow2`, `uni2-h`, and `h-optimus-0` require correct authenticated model construction in the app path

This means the most realistic next engineering step is:

- patch the raw-slide backend so it **tries latest `virchow2` first**
- and only falls back to another extractor if there is both:
  - a working extractor runtime
  - matching trained MIL checkpoints locally
