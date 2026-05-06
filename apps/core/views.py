import json

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from apps.approaches.registry import build_approach_slots
from apps.archives.models import BatchArchive
from apps.runs.models import Run
from apps.runs.services import create_run_from_payload, dashboard_summary
from apps.runs.vm_runtime import sync_run_status
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


def dashboard(request: HttpRequest) -> HttpResponse:
    summary = dashboard_summary()
    live_runs = hydrate_live_runs()
    context = {
        "summary": summary,
        "approach_slots": build_approach_slots(),
        "chart_json": json.dumps(summary["chart"]),
        "recent_runs": live_runs[:6],
        "live_runs": live_runs,
        "history_runs": hydrate_history_runs(),
        "archive_records": BatchArchive.objects.order_by("-updated_at")[:8],
        "integrations": integration_summary(),
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
    return render(
        request,
        "core/partials/live_runs_panel.html",
        {
            "live_runs": hydrate_live_runs(),
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


@require_POST
def launch_run(request: HttpRequest) -> HttpResponse:
    payload = {
        "experiment_name": request.POST.get("experiment_name") or "tcga3-vm-200x10f-virchow-retccl-ctranspath-256tiles",
        "source_uri": request.POST.get("source_uri") or "gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD",
        "annotations_csv": request.POST.get("annotations_csv") or "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/tcga3_vm_annotations.csv",
        "requested_slide_limit": int(request.POST.get("requested_slide_limit") or 200),
        "n_folds": int(request.POST.get("n_folds") or 10),
        "n_repeats": int(request.POST.get("n_repeats") or 1),
        "max_tiles_per_slide": int(request.POST.get("max_tiles_per_slide") or 256),
        "feature_extractor_candidates": [item.strip() for item in (request.POST.get("feature_extractors") or "virchow,retccl,ctranspath").split(",") if item.strip()],
        "approaches": [slot["key"] for slot in settings.MSI_DEFAULT_APPROACHES],
        "n8n_webhook_url": request.POST.get("n8n_webhook_url") or "",
    }
    create_run_from_payload(payload)
    if request.headers.get("HX-Request") == "true":
        return dashboard_metrics_partial(request)
    return redirect("dashboard")


def hydrate_live_runs():
    runs = list(Run.objects.order_by("-created_at")[:4])
    hydrated_runs = []
    for run in runs:
        if run.remote_status_path.startswith("/") or run.bundle_config_path.startswith("/"):
            try:
                sync_run_status(run)
                run.refresh_from_db()
            except Exception:
                continue
        run.label_counts_msi_h = (run.label_counts or {}).get("MSI-H", 0)
        run.label_counts_mss = (run.label_counts or {}).get("MSS", 0)
        run.state_display = STATE_LABELS.get(run.state, run.state.replace("_", " ").title())
        run.is_complete = run.state == "completed"
        run.selected_slide_display = run.selected_slide_count or run.requested_slide_limit
        run.extractor_display = (run.feature_extractor_used or "pending").replace(",", ", ")
        run.best_link = None
        links = list(run.approach_links.select_related("approach_template").all())
        run.display_links = links
        for link in links:
            link.state_display = STATE_LABELS.get(link.state, link.state.replace("_", " ").title())
            link.mean_auroc_display = f"{link.mean_auroc:.3f}" if link.mean_auroc is not None else ""
            link.mean_f1_display = f"{link.mean_f1_macro:.3f}" if link.mean_f1_macro is not None else ""
        completed_links = [link for link in links if link.mean_auroc is not None]
        if completed_links:
            run.best_link = max(completed_links, key=lambda item: item.mean_auroc or 0.0)
        if run.state or run.selected_slide_display or run.best_link:
            hydrated_runs.append(run)
    return hydrated_runs


def hydrate_history_runs():
    runs = list(Run.objects.filter(state__in=["completed", "failed"]).order_by("-updated_at")[:10])
    hydrated_runs = []
    for run in runs:
        run.label_counts_msi_h = (run.label_counts or {}).get("MSI-H", 0)
        run.label_counts_mss = (run.label_counts or {}).get("MSS", 0)
        run.state_display = STATE_LABELS.get(run.state, run.state.replace("_", " ").title())
        run.selected_slide_display = run.selected_slide_count or run.requested_slide_limit
        run.extractor_display = (run.feature_extractor_used or "pending").replace(",", ", ")
        links = list(run.approach_links.select_related("approach_template").all())
        run.display_links = links
        for link in links:
            link.state_display = STATE_LABELS.get(link.state, link.state.replace("_", " ").title())
            link.mean_auroc_display = f"{link.mean_auroc:.3f}" if link.mean_auroc is not None else ""
            link.mean_f1_display = f"{link.mean_f1_macro:.3f}" if link.mean_f1_macro is not None else ""
        hydrated_runs.append(run)
    return hydrated_runs
