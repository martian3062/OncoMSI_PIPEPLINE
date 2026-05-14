# Final Top-4 Completion Log

Last updated: `2026-05-09 02:23:32 +05:30`  
Local workspace: `E:\Cleaned_MSI`  
Remote VM: `pardeep@34.126.112.227`  
Completed run: `run-65512be1f9c4`

## Final Results

Run status: `completed`  
Selected slides: `200`  
Class balance: `74 MSI-H`, `126 MSS`  
Cross-validation design: `10 folds x 5 repeats = 50 fold-runs per approach`

| Rank | Approach | Mean AUROC | Mean F1 Macro | Mean AUPRC | Mean Balanced Acc | Mean Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `Approach2-Virchow2` | `0.9718` | `0.9442` | `0.9835` | `0.9454` | `0.4112` |
| 2 | `Approach5-H-Optimus-0` | `0.9700` | `0.9368` | `0.9819` | `0.9384` | `0.4168` |
| 3 | `Approach1-UNI2-h` | `0.9646` | `0.9362` | `0.9797` | `0.9379` | `0.4190` |
| 4 | `Approach6-Midnight-12k` | `0.9511` | `0.9305` | `0.9693` | `0.9330` | `0.4002` |

Best approach recorded by the VM summary: `Approach2-Virchow2`

Primary finished artifacts on the VM:
- `automation/tcga_slide_triads/run-65512be1f9c4/status.json`
- `automation/tcga_slide_triads/run-65512be1f9c4/final_summary.json`
- `automation/tcga_slide_triads/run-65512be1f9c4/approaches/*/monte_carlo_summary.json`
- `automation/tcga_slide_triads/run-65512be1f9c4/approaches/*/per_slide_predictions.csv`
- `automation/tcga_slide_triads/run-65512be1f9c4/approaches/*/fold_metrics.csv`

## Last 3 Hours

This section covers the final stretch from roughly `2026-05-08 17:53 UTC` to `2026-05-08 20:53 UTC`, which is `2026-05-08 23:23 IST` to `2026-05-09 02:23 IST`.

### VM timeline

`2026-05-08 18:36:33 UTC`
- Recovery supervisor was already active and waiting for the first pair to finish cleanly.

`2026-05-08 18:47:33 UTC`
- First pair completed.
- Recovery handoff launched the second pair:
  - `Approach5-H-Optimus-0` with PID `259100`
  - `Approach6-Midnight-12k` with PID `259101`

`2026-05-08 18:47 UTC -> 20:33 UTC`
- Second pair trained in parallel on the `NVIDIA L4`.
- Live checks during this window consistently showed:
  - GPU utilization near `98% to 100%`
  - GPU memory around `5.9 GB to 6.9 GB / 23.0 GB`
  - worker CPU near `97% to 98%` per main trainer
- Progress checkpoints during this stretch showed:
  - both approaches moving through repeated `repeat_X_fold_Y` TransMIL jobs
  - fold counts advancing from the teens, through repeat 4, and into repeat 5
  - `Approach5-H-Optimus-0` reaching `50/50`
  - `Approach6-Midnight-12k` reaching `50/50`

`2026-05-08 20:33:57 UTC`
- Recovery log recorded: `second pair finished, finalizing existing bundle`

`2026-05-08 20:34:04 UTC`
- Recovery log recorded: `finalize-existing exit=0`
- That means the finalize stage exited successfully.

`2026-05-08 20:53 UTC`
- Final live status check showed:
  - bundle state: `completed`
  - `Approach5-H-Optimus-0`: `50/50`
  - `Approach6-Midnight-12k`: `50/50`
  - GPU utilization: `0%`
  - GPU memory: `0 / 23034 MiB`
  - no active `train-approach` processes remaining

### What was confirmed technically on the VM

- The full Top-4 run finished across all four active approaches:
  - `Approach1-UNI2-h`
  - `Approach2-Virchow2`
  - `Approach5-H-Optimus-0`
  - `Approach6-Midnight-12k`
- The bundle status file now reports:
  - `completed_approach_count = 4`
  - `failed_approach_count = 0`
  - `state = completed`
- The final summary includes:
  - per-approach aggregate confusion matrices
  - Monte Carlo summary artifact paths
  - per-slide prediction CSV artifact paths
  - prediction parquet output references
- The earlier external-cohort prep crash path was bypassed by the runner patch so the training bundle could complete and finalize successfully.

## Local Work Completed In Parallel

### Django live monitoring

A real monitor page was added to the cleaned app:
- route: `/monitor/`
- auto-refresh: `20s`
- visuals now include:
  - per-approach progress bars
  - repeat/fold completion
  - current epoch display
  - ETA card
  - GPU and RAM cards
  - worker CPU cards
  - latest log event feed
  - current fold epoch curves

Key files added or updated:
- `apps/core/live_monitor.py`
- `apps/core/views.py`
- `apps/core/urls.py`
- `apps/core/templates/core/live_monitor.html`
- `apps/core/templates/core/base.html`
- `apps/core/static/core/app.css`

Validation completed locally:
- `manage.py check`
- `GET /`
- `GET /monitor/`

### README and app state

README top was already updated earlier to reflect:
- cleaned `final` results-only app state
- final Top-4 training protocol
- Monte Carlo explanation and formula
- folds, repeats, epochs, and reporting contract

### Git and GitHub publishing

Local branch situation during this window:
- local `final` head remained:
  - `63ef336` `Finalize cleaned results app and top4 runtime`

GitHub publishing issue encountered:
- direct push of local `final` failed because unpublished branch history still contained giant slide and weight blobs from older local-only commits
- large hidden history included items like:
  - `new3/vm_weights/virchow/pytorch_model.bin`
  - `msi-h/downloads/gdc_5_exact/*.svs`

Publish-safe workaround completed:
- created a clean publish branch from `origin/semi-final`
- restored only the desired cleaned app/runtime state
- excluded the old giant local-only history
- pushed successfully to GitHub branch `final`

Published remote branch:
- `origin/final`

Published clean commit on GitHub:
- `5719e10` `Publish final cleaned results app`

Local backup branch kept:
- `final-backup-63ef336`

## Important Notes

- Local `final` and GitHub `final` are not the same commit right now.
- GitHub `final` points to the clean publish-safe commit `5719e10`.
- Local `final` still points to `63ef336`.
- This was intentional so the remote branch could be published without the old large unpublished history.

## Net Outcome

At the end of this 3-hour window:
- the Top-4 Monte Carlo run finished successfully on the VM
- the winner was recorded as `Approach2-Virchow2`
- the cleaned Django app gained a live `/monitor/` page
- the cleaned project was successfully published to GitHub on branch `final`
- the final bundle now has the artifact structure needed for downstream result review and inference packaging
