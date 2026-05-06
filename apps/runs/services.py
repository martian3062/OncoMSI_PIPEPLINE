from collections import Counter

from django.conf import settings

import plotly.graph_objects as go

from apps.approaches.registry import build_approach_slots, build_launch_slots, sync_default_approaches
from apps.archives.models import BatchArchive

from .models import Run, RunApproachLink, RunState
from .n8n import trigger_n8n_run_created


def create_run_from_payload(payload: dict) -> Run:
    sync_default_approaches()
    slots = build_launch_slots()
    run = Run.objects.create(
        experiment_name=payload["experiment_name"],
        source_uri=payload["source_uri"],
        annotations_csv=payload["annotations_csv"],
        requested_slide_limit=payload["requested_slide_limit"],
        n_folds=payload["n_folds"],
        n_repeats=payload["n_repeats"],
        max_tiles_per_slide=payload["max_tiles_per_slide"],
        feature_extractor_candidates=payload["feature_extractor_candidates"],
        feature_extractor_used=payload["feature_extractor_candidates"][0] if payload["feature_extractor_candidates"] else "",
        label_counts={"MSI-H": 0, "MSS": 0},
        remote_status_path="automation/tcga_slide_triads/<bundle_id>/status.json",
        archive_path="automation/tcga_batch_archives_<date>/<bundle_id>",
    )
    for slot in slots:
        RunApproachLink.objects.create(
            run=run,
            approach_template=slot,
            trainer_params=slot.default_params,
        )
    trigger_n8n_run_created(
        {
            "run_id": run.run_id,
            "experiment_name": run.experiment_name,
            "source_uri": run.source_uri,
            "annotations_csv": run.annotations_csv,
            "requested_slide_limit": run.requested_slide_limit,
            "feature_extractor_candidates": run.feature_extractor_candidates,
            "n_folds": run.n_folds,
            "n_repeats": run.n_repeats,
            "max_tiles_per_slide": run.max_tiles_per_slide,
            "approaches": [slot.key for slot in slots],
            "n8n_webhook_url": payload.get("n8n_webhook_url", ""),
        }
    )
    return run


def dashboard_summary() -> dict:
    sync_default_approaches()
    total_runs = Run.objects.count()
    active_runs = Run.objects.exclude(state__in=[RunState.COMPLETED, RunState.FAILED]).count()
    archive_count = BatchArchive.objects.count()
    approach_slots = build_approach_slots()
    links = list(
        RunApproachLink.objects.select_related("approach_template").order_by("created_at")
    )
    by_approach = Counter(link.approach_template.label for link in links)
    if not by_approach:
        by_approach = Counter({slot.label: 0 for slot in approach_slots})
    figure = go.Figure(
        data=[
            go.Bar(
                x=list(by_approach.keys()),
                y=list(by_approach.values()),
                marker_color=[
                    "#54d7ff",
                    "#ff7c6b",
                    "#ffd166",
                    "#9bffb0",
                    "#89a8ff",
                    "#ff9be7",
                    "#95f0ff",
                    "#ffb86c",
                    "#b0b7ff",
                ][: len(by_approach)],
                text=list(by_approach.values()),
                textposition="outside",
            )
        ]
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef5ff"},
        margin={"l": 24, "r": 24, "t": 10, "b": 24},
        yaxis={"title": "Configured runs", "gridcolor": "rgba(255,255,255,0.08)"},
        xaxis={"title": "Approach slot"},
    )
    return {
        "total_runs": total_runs,
        "active_runs": active_runs,
        "archive_count": archive_count,
        "approach_count": len(approach_slots),
        "chart": figure.to_plotly_json(),
    }
