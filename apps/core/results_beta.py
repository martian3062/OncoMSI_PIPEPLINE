import json
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from apps.runs.models import Run


TERMINAL_STATES = {"completed", "failed"}


def _safe_float(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _summary_roots() -> list[Path]:
    base = Path(settings.BASE_DIR)
    return [
        base / "archive" / "local_cleanup_2026-05-08" / "ten",
        base / "ten",
        base / "archive" / "local_cleanup_2026-05-08" / "final_local_results" / "ten",
    ]


def _find_latest_summary_path() -> Path | None:
    candidates: list[Path] = []
    for root in _summary_roots():
        if not root.exists():
            continue
        candidates.extend(
            path for path in root.rglob("final_summary.json") if "manager_repro_store" not in str(path)
        )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


@lru_cache(maxsize=1)
def load_results_beta_bundle() -> dict:
    summary_path = _find_latest_summary_path()
    if not summary_path:
        return {"summary_path": None, "summary": {}}
    return {
        "summary_path": summary_path,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def _build_summary_leaderboard(summary: dict) -> list[dict]:
    rows = []
    for label, payload in (summary.get("approaches") or {}).items():
        rows.append(
            {
                "label": label,
                "extractor": payload.get("feature_extractor_used") or "-",
                "auroc": payload.get("mean_auroc"),
                "auroc_display": _safe_float(payload.get("mean_auroc")),
                "f1_display": _safe_float(payload.get("mean_f1_macro")),
                "auprc_display": _safe_float(payload.get("mean_auprc")),
                "bal_acc_display": _safe_float(payload.get("mean_balanced_accuracy")),
                "recall_display": _safe_float(payload.get("mean_recall_msi_h")),
                "specificity_display": _safe_float(payload.get("mean_specificity")),
                "threshold_display": _safe_float(payload.get("mean_best_threshold")),
            }
        )
    return sorted(rows, key=lambda item: item["auroc"] if item["auroc"] is not None else -1, reverse=True)


def _build_db_leaderboard() -> list[dict]:
    run = Run.objects.filter(state="completed").order_by("-updated_at").first()
    if not run:
        return []
    rows = []
    for link in run.approach_links.select_related("approach_template").all():
        rows.append(
            {
                "label": link.approach_template.label,
                "extractor": (link.trainer_params or {}).get("feature_extractor", "-"),
                "auroc": link.mean_auroc,
                "auroc_display": _safe_float(link.mean_auroc),
                "f1_display": _safe_float(link.mean_f1_macro),
                "auprc_display": _safe_float(link.mean_auprc),
                "bal_acc_display": _safe_float(link.mean_balanced_accuracy),
                "recall_display": _safe_float(link.mean_recall_msi_h),
                "specificity_display": _safe_float(link.mean_specificity),
                "threshold_display": _safe_float(link.mean_best_threshold),
            }
        )
    return sorted(rows, key=lambda item: item["auroc"] if item["auroc"] is not None else -1, reverse=True)


def _build_live_run_snapshot() -> dict | None:
    run = Run.objects.exclude(state__in=TERMINAL_STATES).order_by("-updated_at").first()
    if not run:
        return None
    running = run.approach_links.filter(state__in=["training", "spawned"]).count()
    completed = run.approach_links.filter(state="completed").count()
    total = run.approach_links.count()
    return {
        "run_id": run.run_id,
        "state": run.state.replace("_", " ").title(),
        "experiment_name": run.experiment_name,
        "updated_at": timezone.localtime(run.updated_at).strftime("%d %b %Y %I:%M %p"),
        "selected_slide_count": run.selected_slide_count or run.requested_slide_limit,
        "n_folds": run.n_folds,
        "n_repeats": run.n_repeats,
        "parallel_running": running,
        "completed_links": completed,
        "total_links": total,
        "feature_extractors": ", ".join(run.feature_extractor_candidates or []) or "-",
    }


def build_results_beta_context() -> dict:
    bundle = load_results_beta_bundle()
    summary = bundle["summary"]
    leaderboard = _build_summary_leaderboard(summary) if summary else _build_db_leaderboard()
    best = leaderboard[0] if leaderboard else None
    labels = summary.get("label_counts") or {}
    total_slides = summary.get("selected_slide_count") or summary.get("requested_slide_limit") or 0
    msi_h = int(labels.get("MSI-H") or 0)
    mss = int(labels.get("MSS") or 0)
    return {
        "summary_available": bool(summary),
        "source_path": str(bundle["summary_path"]) if bundle["summary_path"] else "",
        "bundle_id": summary.get("bundle_id") or "",
        "best_approach": summary.get("best_approach") or (best["label"] if best else "-"),
        "state": summary.get("state") or "-",
        "selected_slide_count": total_slides,
        "msi_h_count": msi_h,
        "mss_count": mss,
        "msi_h_rate_display": _safe_float((msi_h / total_slides) if total_slides else None, digits=3),
        "n_folds": summary.get("n_folds") or 0,
        "n_repeats": summary.get("n_repeats") or 0,
        "approach_count": len(leaderboard),
        "feature_extractors": ", ".join(summary.get("feature_extractor_candidates") or []),
        "leaderboard": leaderboard,
        "top_four": leaderboard[:4],
        "live_run": _build_live_run_snapshot(),
    }
