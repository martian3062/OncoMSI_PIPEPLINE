import json
import shlex
from datetime import timedelta

from django.conf import settings
from django.db.models import Q
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from apps.approaches.registry import build_approach_slots
from apps.archives.models import BatchArchive
from apps.runs.models import Run
from apps.runs.services import create_run_from_payload, dashboard_summary
from apps.runs.vm_runtime import sync_run_status
from apps.vm.services import default_vm_target, run_shell
from .services import integration_summary


STATE_LABELS = {
    "matching_annotations": "Matching annotations",
    "downloading_slides": "Downloading slides",
    "extracting_tiles": "Extracting tiles",
    "retrying_tiles": "Retrying tiles",
    "generating_features": "Generating features",
    "prepared": "Prepared",
    "training_parallel": "Training in parallel",
    "completed": "Completed",
    "failed": "Failed",
    "pending": "Pending",
    "spawned": "Queued",
    "training": "Training",
}

STATE_PROGRESS = {
    "matching_annotations": 8,
    "downloading_slides": 18,
    "extracting_tiles": 36,
    "retrying_tiles": 42,
    "generating_features": 56,
    "prepared": 64,
    "training_parallel": 72,
    "training": 72,
    "completed": 100,
    "failed": 100,
}

LIVE_STATES = {
    "matching_annotations",
    "downloading_slides",
    "extracting_tiles",
    "retrying_tiles",
    "generating_features",
    "prepared",
    "training_parallel",
    "training",
    "spawned",
    "pending",
}


def format_duration(delta: timedelta | None) -> str:
    if not delta:
        return "n/a"
    total_seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def metric_display(value, digits: int = 3) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def infer_run_progress(run: Run) -> tuple[int, str]:
    if run.state in {"training_parallel", "training"} and run.total_link_count:
        progress = 72 + int(round((run.completed_link_count / max(1, run.total_link_count)) * 24))
        detail = f"{run.completed_link_count}/{run.total_link_count} approaches synced"
        return min(96, progress), detail
    base = STATE_PROGRESS.get(run.state, 0)
    if run.state == "completed":
        return 100, "Bundle finished"
    if run.state == "failed":
        return 100, "Runner needs recovery"
    return base, run.state_display


def infer_eta_copy(run: Run) -> str:
    if run.state == "completed":
        return "Completed"
    if run.state == "failed":
        return "Needs recovery"

    elapsed = timezone.now() - run.created_at
    if run.state in {"training_parallel", "training"} and run.completed_link_count > 0 and run.total_link_count > run.completed_link_count:
        completed = max(1, run.completed_link_count)
        per_approach_seconds = elapsed.total_seconds() / completed
        remaining = run.total_link_count - run.completed_link_count
        eta = timedelta(seconds=per_approach_seconds * remaining)
        return f"ETA {format_duration(eta)}"

    if run.state in {"extracting_tiles", "generating_features"}:
        return "ETA stabilizes after the first synced branch"
    if run.state in {"matching_annotations", "downloading_slides", "prepared"}:
        return "ETA pending first runtime sync"
    return "ETA pending"


def build_scientific_metrics(link) -> list[dict[str, str]]:
    return [
        {"label": "AUROC", "value": metric_display(link.mean_auroc)},
        {"label": "AUPRC", "value": metric_display(getattr(link, "mean_auprc", None))},
        {"label": "Bal Acc", "value": metric_display(getattr(link, "mean_balanced_accuracy", None))},
        {"label": "MSI-H Recall", "value": metric_display(getattr(link, "mean_recall_msi_h", None))},
        {"label": "Specificity", "value": metric_display(getattr(link, "mean_specificity", None))},
        {"label": "Precision", "value": metric_display(getattr(link, "mean_precision", None))},
        {"label": "Brier", "value": metric_display(getattr(link, "mean_brier_score", None))},
        {"label": "Threshold", "value": metric_display(getattr(link, "mean_best_threshold", None))},
    ]


def build_stage_copy(run: Run) -> tuple[str, str]:
    if run.sync_error:
        return (
            f"Last known stage is {run.state_display}.",
            run.sync_detail or "Live VM status could not be refreshed, so this headline is showing the latest saved state.",
        )
    if run.state == "matching_annotations":
        return (
            "Matching annotations against the TCGA cohort.",
            f"Preparing the cohort map before downloads start for {run.requested_slide_limit} requested slides.",
        )
    if run.state == "downloading_slides":
        return (
            f"Downloading {run.selected_slide_display} slides from the bucket.",
            f"Current class balance is {run.label_counts_msi_h} MSI-H and {run.label_counts_mss} MSS.",
        )
    if run.state == "extracting_tiles":
        return (
            f"Tiling is running for {run.selected_slide_display} selected slides.",
            f"Slides are downloaded. The runner is now cutting pathology tiles for {run.extractor_display}.",
        )
    if run.state == "retrying_tiles":
        return (
            "Tile extraction is retrying failed slides.",
            "The VM is reprocessing the incomplete slides before feature generation continues.",
        )
    if run.state == "generating_features":
        return (
            f"Feature bags are building for {run.extractor_display}.",
            f"Tiles are ready. Embeddings are being generated across {run.selected_slide_display} slides.",
        )
    if run.state in {"training_parallel", "training"}:
        if run.active_link is not None:
            return (
                f"{run.completed_link_count}/{run.total_link_count} approaches completed. {run.active_link.approach_template.label} is active.",
                f"{run.running_link_count} approach branches are still running and {run.pending_link_count} are queued.",
            )
        return (
            "Approach training has started.",
            f"{run.running_link_count} approach branches are running across the current bundle.",
        )
    if run.state == "completed" and run.best_link:
        return (
            f"Run completed. {run.best_link.approach_template.label} is leading.",
            f"Best synced metrics are AUROC {run.best_link.mean_auroc_display} and F1 {run.best_link.mean_f1_display}.",
        )
    if run.state == "failed":
        return (
            "Run entered a failure state.",
            "The bundle needs recovery, relaunch, or a deeper VM-side log check.",
        )
    return (run.state_display, f"Latest known extractor is {run.extractor_display}.")


def build_milestone_items(live_runs: list[Run]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for run in live_runs:
        title, detail = build_stage_copy(run)
        meta = f"Run {run.run_id} • Updated {run.last_sync_display}"
        if run.sync_error:
            meta = f"{meta} • VM sync warning"
        items.append(
            {
                "run_id": run.run_id,
                "stage": run.state_display,
                "title": title,
                "detail": detail,
                "meta": meta,
                "variant": "warning" if run.sync_error else "live",
            }
        )

    if not items:
        items.append(
            {
                "run_id": "No live run",
                "stage": "Idle",
                "title": "No active milestone yet.",
                "detail": "Launch a run and this flash strip will start rotating live pipeline updates.",
                "meta": "Waiting for the next bundle",
                "variant": "idle",
            }
        )
    return items


def infer_stage_from_launch_log(run: Run) -> tuple[str | None, str]:
    if not run.remote_launch_log_path.startswith("/"):
        return None, ""

    command = f"tail -n 80 {shlex.quote(run.remote_launch_log_path)}"
    result = run_shell(default_vm_target(), command, timeout=20)
    stdout = result.stdout or ""
    if result.returncode != 0 or not stdout.strip():
        return None, ""

    log_text = stdout.lower()
    if "traceback" in log_text or "runtimeerror" in log_text or "exception" in log_text:
        return "failed", "The launch log shows a failure signature after the last VM update."
    if "using feature extractor:" in log_text or "generating..." in log_text:
        return "generating_features", "The launch log reached feature generation before the live status path disappeared."
    if "finished tile extraction" in log_text or "extracting (" in log_text:
        return "extracting_tiles", "The launch log reached tile extraction before the live status path disappeared."
    if "downloading" in log_text:
        return "downloading_slides", "The launch log still shows slide downloads as the latest visible stage."
    return None, ""


def hydrate_run(run: Run, *, allow_sync_failure: bool, sync_remote: bool) -> Run | None:
    run.sync_error = ""
    run.sync_detail = ""
    if sync_remote and (run.remote_status_path.startswith("/") or run.bundle_config_path.startswith("/")):
        try:
            sync_run_status(run)
            run.refresh_from_db()
        except Exception as exc:
            run.sync_error = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            inferred_state, inferred_detail = infer_stage_from_launch_log(run)
            if inferred_state:
                run.state = inferred_state
            run.sync_detail = inferred_detail
            if not allow_sync_failure:
                return None

    run.label_counts_msi_h = (run.label_counts or {}).get("MSI-H", 0)
    run.label_counts_mss = (run.label_counts or {}).get("MSS", 0)
    run.state_display = STATE_LABELS.get(run.state, run.state.replace("_", " ").title())
    run.is_complete = run.state == "completed"
    run.selected_slide_display = run.selected_slide_count or run.requested_slide_limit
    run.extractor_display = (run.feature_extractor_used or "pending").replace(",", ", ")
    run.best_link = None
    run.active_link = None
    run.last_sync_display = timezone.localtime(run.updated_at).strftime("%d %b %I:%M:%S %p")
    run.elapsed_display = format_duration(timezone.now() - run.created_at)
    links = list(run.approach_links.select_related("approach_template").all())
    run.display_links = links
    for link in links:
        link.state_display = STATE_LABELS.get(link.state, link.state.replace("_", " ").title())
        link.mean_auroc_display = f"{link.mean_auroc:.3f}" if link.mean_auroc is not None else ""
        link.mean_f1_display = f"{link.mean_f1_macro:.3f}" if link.mean_f1_macro is not None else ""
        link.mean_auprc_display = metric_display(getattr(link, "mean_auprc", None))
        link.mean_balanced_accuracy_display = metric_display(getattr(link, "mean_balanced_accuracy", None))
        link.mean_recall_display = metric_display(getattr(link, "mean_recall_msi_h", None))
        link.mean_specificity_display = metric_display(getattr(link, "mean_specificity", None))
        link.mean_precision_display = metric_display(getattr(link, "mean_precision", None))
        link.mean_brier_display = metric_display(getattr(link, "mean_brier_score", None))
        link.mean_threshold_display = metric_display(getattr(link, "mean_best_threshold", None))
        link.available_bag_slide_count_display = getattr(link, "available_bag_slide_count", None) or "-"
        link.missing_bag_slide_count = len(getattr(link, "missing_bag_slides", []) or [])
        link.chart_score_pct = max(2, min(100, int(round((link.mean_auroc or 0.0) * 100)))) if link.mean_auroc is not None else 6
    completed_links = [link for link in links if link.mean_auroc is not None]
    if completed_links:
        run.best_link = max(completed_links, key=lambda item: item.mean_auroc or 0.0)
        run.best_link.scientific_metrics = build_scientific_metrics(run.best_link)
    else:
        run.best_link = None
    run.active_link = next((link for link in links if link.state in {"training", "spawned"}), None)
    run.total_link_count = len(links)
    run.completed_link_count = sum(1 for link in links if link.state == "completed")
    run.failed_link_count = sum(1 for link in links if link.state == "failed")
    run.running_link_count = sum(1 for link in links if link.state in {"training", "spawned"})
    run.pending_link_count = max(0, run.total_link_count - run.completed_link_count - run.failed_link_count - run.running_link_count)
    run.progress_percent, run.progress_detail = infer_run_progress(run)
    run.eta_display = infer_eta_copy(run)
    run.external_cohort_count = len(getattr(run, "external_cohorts", []) or [])
    run.stage_title, run.stage_detail = build_stage_copy(run)
    return run


def dashboard(request: HttpRequest) -> HttpResponse:
    summary = dashboard_summary()
    live_runs = hydrate_live_runs(sync_remote=False)
    recent_runs = hydrate_recent_runs(sync_remote=False, limit=8)
    milestone_items = build_milestone_items(live_runs)
    initial_tab = "history" if request.GET.get("tab") == "history" else "live"
    history_runs = hydrate_history_runs() if initial_tab == "history" else []
    archive_records = BatchArchive.objects.order_by("-updated_at")[:8] if initial_tab == "history" else []
    context = {
        "summary": summary,
        "approach_slots": build_approach_slots(),
        "chart_json": json.dumps(summary["chart"]),
        "recent_runs": recent_runs,
        "live_runs": live_runs,
        "history_runs": history_runs,
        "archive_records": archive_records,
        "integrations": integration_summary(),
        "initial_tab": initial_tab,
        "milestone_items": milestone_items,
        "milestone_items_json": json.dumps(milestone_items),
    }
    return render(request, "core/dashboard.html", context)


@require_GET
def dashboard_metrics_partial(request: HttpRequest) -> HttpResponse:
    summary = dashboard_summary()
    return render(
        request,
        "core/partials/metrics_panel.html",
        {
            "summary": summary,
            "chart_json": json.dumps(summary["chart"]),
        },
    )


@require_GET
def live_runs_partial(request: HttpRequest) -> HttpResponse:
    live_runs = hydrate_live_runs(sync_remote=True)
    return render(
        request,
        "core/partials/live_runs_panel.html",
        {
            "live_runs": live_runs,
        },
    )


@require_GET
def milestone_ticker_partial(request: HttpRequest) -> HttpResponse:
    live_runs = hydrate_live_runs(sync_remote=False)
    milestone_items = build_milestone_items(live_runs)
    return render(
        request,
        "core/partials/milestone_ticker.html",
        {
            "milestone_items": milestone_items,
            "milestone_items_json": json.dumps(milestone_items),
        },
    )


@require_GET
def history_partial(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "core/partials/history_panel.html",
        {
            "history_runs": hydrate_history_runs(),
            "archive_records": BatchArchive.objects.order_by("-updated_at")[:8],
        },
    )


@require_GET
def history_page(request: HttpRequest) -> HttpResponse:
    summary = dashboard_summary()
    live_runs = hydrate_live_runs(sync_remote=False)
    recent_runs = hydrate_recent_runs(sync_remote=False, limit=12)
    milestone_items = build_milestone_items(live_runs)
    context = {
        "summary": summary,
        "approach_slots": build_approach_slots(),
        "chart_json": json.dumps(summary["chart"]),
        "recent_runs": recent_runs,
        "live_runs": live_runs,
        "history_runs": hydrate_history_runs(limit=30),
        "archive_records": BatchArchive.objects.order_by("-updated_at")[:20],
        "integrations": integration_summary(),
        "initial_tab": "history",
        "milestone_items": milestone_items,
        "milestone_items_json": json.dumps(milestone_items),
    }
    return render(request, "core/dashboard.html", context)


@require_POST
def launch_run(request: HttpRequest) -> HttpResponse:
    external_cohorts_text = request.POST.get("external_cohorts_json", "").strip()
    try:
        external_cohorts = json.loads(external_cohorts_text) if external_cohorts_text else []
    except json.JSONDecodeError:
        external_cohorts = []
    payload = {
        "experiment_name": request.POST.get("experiment_name") or "tcga3-hybrid-02-200x10f-conch15-phikonv2-gigapath-prism-chief-dinov3-midnight-256tiles",
        "source_uri": request.POST.get("source_uri") or "gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD",
        "annotations_csv": request.POST.get("annotations_csv") or "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/tcga3_vm_annotations.csv",
        "requested_slide_limit": int(request.POST.get("requested_slide_limit") or 200),
        "n_folds": int(request.POST.get("n_folds") or 10),
        "n_repeats": int(request.POST.get("n_repeats") or 1),
        "max_tiles_per_slide": int(request.POST.get("max_tiles_per_slide") or 256),
        "feature_extractor_candidates": [
            item.strip()
            for item in (
                request.POST.get("feature_extractors")
                or "conchv1_5,phikon-v2,prov-gigapath,prism-virchow,chief-ctranspath,dinov3,midnight"
            ).split(",")
            if item.strip()
        ],
        "external_cohorts": external_cohorts,
        "approaches": [slot["key"] for slot in settings.MSI_DEFAULT_APPROACHES],
        "n8n_webhook_url": request.POST.get("n8n_webhook_url") or "",
    }
    create_run_from_payload(payload)
    if request.headers.get("HX-Request") == "true":
        return dashboard_metrics_partial(request)
    return redirect("dashboard")


def hydrate_live_runs(sync_remote: bool = True):
    recent_cutoff = timezone.now() - timedelta(hours=8)
    runs = list(
        Run.objects.filter(state__in=LIVE_STATES, updated_at__gte=recent_cutoff)
        .order_by("-updated_at")[:6]
    )
    hydrated_runs = []
    for run in runs:
        hydrated = hydrate_run(run, allow_sync_failure=True, sync_remote=sync_remote)
        if hydrated and (hydrated.state or hydrated.selected_slide_display or hydrated.best_link):
            hydrated_runs.append(hydrated)
    return hydrated_runs


def hydrate_recent_runs(sync_remote: bool = False, limit: int = 8):
    runs = list(Run.objects.order_by("-updated_at")[:limit])
    hydrated_runs = []
    for run in runs:
        hydrated = hydrate_run(run, allow_sync_failure=True, sync_remote=sync_remote)
        if hydrated is not None:
            hydrated_runs.append(hydrated)
    return hydrated_runs


def hydrate_history_runs(limit: int = 16):
    terminal = Q(state__in=["completed", "failed"])
    runs = list(
        Run.objects.filter(terminal)
        .order_by("-updated_at")[:limit]
    )
    hydrated_runs = []
    for run in runs:
        hydrated = hydrate_run(run, allow_sync_failure=True, sync_remote=False)
        if hydrated is not None:
            hydrated_runs.append(hydrated)
    return hydrated_runs
