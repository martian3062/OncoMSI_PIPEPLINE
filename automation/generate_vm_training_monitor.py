from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


TIME_RE = re.compile(r"\[(?P<ts>\d{2}:\d{2}:\d{2})\]")
EXTRACTOR_RE = re.compile(r"Using feature extractor:\s+(?P<name>[A-Za-z0-9_.\-]+)")
APPROACH_PATH_RE = re.compile(
    r"(?P<approach>a\d+_[a-z0-9\-]+)_transmil_repeat_(?P<repeat>\d+)_fold_(?P<fold>\d+)",
    re.IGNORECASE,
)


@dataclass
class Milestone:
    label: str
    start: datetime
    end: datetime
    kind: str

    @property
    def duration_minutes(self) -> float:
        return max((self.end - self.start).total_seconds() / 60.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate live VM training progress PNGs.")
    parser.add_argument("--run-dir", required=True, help="Local synced run directory.")
    parser.add_argument("--output-dir", help="Directory for generated PNGs and summaries.")
    parser.add_argument("--date", default="2026-05-08", help="Date to use for HH:MM:SS log timestamps.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_hms(date_text: str, hhmmss: str) -> datetime:
    return datetime.strptime(f"{date_text} {hhmmss}", "%Y-%m-%d %H:%M:%S")


def build_milestones(run_dir: Path, date_text: str) -> list[Milestone]:
    relaunch_log = run_dir / "relaunch.log"

    lines = relaunch_log.read_text(encoding="utf-8", errors="ignore").splitlines()
    extractor_events: list[tuple[str, datetime]] = []
    extraction_finished: datetime | None = None
    bundle_start: datetime | None = None
    for line in lines:
        ts_match = TIME_RE.search(line)
        if not ts_match:
            continue
        ts = parse_hms(date_text, ts_match.group("ts"))
        if bundle_start is None:
            bundle_start = ts
        if "Finished tile extraction for" in line:
            extraction_finished = ts
        ex_match = EXTRACTOR_RE.search(line)
        if ex_match:
            extractor_events.append((ex_match.group("name"), ts))

    training_start = earliest_runner_timestamp(run_dir, date_text)
    training_observed_end = latest_sample_timestamp(run_dir / "pace_sample_2min.txt", date_text)

    milestones: list[Milestone] = []
    if extraction_finished and bundle_start:
        milestones.append(Milestone("Launch -> extraction complete", bundle_start, extraction_finished, "preprocess"))

    if extractor_events:
        feature_phase_start = extractor_events[0][1]
        if extraction_finished and feature_phase_start > extraction_finished:
            milestones.append(Milestone("Extraction tail / handoff", extraction_finished, feature_phase_start, "handoff"))

        for idx, (name, start) in enumerate(extractor_events):
            if idx + 1 < len(extractor_events):
                end = extractor_events[idx + 1][1]
                label = f"Feature gen: {name}"
                kind = "feature"
            else:
                end = training_start or training_observed_end or start
                label = f"{name} -> recovery -> training resume"
                kind = "recovery"
            milestones.append(Milestone(label, start, end, kind))

    if training_start:
        training_now = training_observed_end or training_start
        milestones.append(Milestone("Observed training window", training_start, training_now, "training"))

    return milestones


def parse_iso_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def earliest_runner_timestamp(run_dir: Path, date_text: str) -> datetime | None:
    candidates: list[datetime] = []
    for runner_log in run_dir.glob("approaches/*/runner.log"):
        for line in runner_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            ts_match = TIME_RE.search(line)
            if ts_match:
                candidates.append(parse_hms(date_text, ts_match.group("ts")))
                break
    return min(candidates) if candidates else None


def latest_sample_timestamp(path: Path, date_text: str) -> datetime | None:
    if not path.exists():
        return None
    timestamps = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if re.match(r"^\d{2}:\d{2}:\d{2}$", line.strip()):
            timestamps.append(parse_hms(date_text, line.strip()))
    return max(timestamps) if timestamps else None


def load_histories(run_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for history_path in sorted(run_dir.glob("mil/**/history.csv")):
        match = APPROACH_PATH_RE.search(str(history_path).replace("\\", "/"))
        if not match:
            continue
        df = pd.read_csv(history_path)
        if df.empty:
            continue
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
        df["valid_loss"] = pd.to_numeric(df["valid_loss"], errors="coerce")
        df["roc_auc_score"] = pd.to_numeric(df["roc_auc_score"], errors="coerce")
        df["epoch_seconds"] = df["time"].map(parse_duration_to_seconds)
        df["elapsed_seconds"] = df["epoch_seconds"].fillna(0).cumsum()
        approach = match.group("approach")
        pretty = approach.split("_", 1)[1].replace("-", " ").title()
        repeat = int(match.group("repeat"))
        fold = int(match.group("fold"))
        mtime = datetime.fromtimestamp(history_path.stat().st_mtime)
        for row in df.to_dict("records"):
            row.update(
                {
                    "approach_key": approach,
                    "approach": pretty,
                    "repeat": repeat,
                    "fold": fold,
                    "history_path": str(history_path),
                    "history_mtime": mtime,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def parse_duration_to_seconds(value: str) -> float:
    if not isinstance(value, str) or ":" not in value:
        return math.nan
    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def load_pace_sample(path: Path, date_text: str) -> pd.DataFrame:
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line == "__SEP__":
            if current:
                blocks.append(current)
            current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    rows: list[dict] = []
    pattern = re.compile(r"^\s*(?P<pid>\d+)\s+(?P<etimes>\d+)\s+(?P<cpu>[0-9.]+)\s+(?P<mem>[0-9.]+)\s+(?P<cmd>.+)$")
    for block in blocks:
        if not block:
            continue
        sample_time = parse_hms(date_text, block[0])
        for raw in block[1:]:
            match = pattern.match(raw)
            if not match:
                continue
            cmd = match.group("cmd")
            if "Approach1-UNI2-h" in cmd:
                bucket = "Approach1-UNI2-h"
            elif "Approach2-Virchow2" in cmd:
                bucket = "Approach2-Virchow2"
            else:
                bucket = "other"
            rows.append(
                {
                    "sample_time": sample_time,
                    "pid": int(match.group("pid")),
                    "etimes": int(match.group("etimes")),
                    "cpu": float(match.group("cpu")),
                    "mem": float(match.group("mem")),
                    "cmd": cmd,
                    "bucket": bucket,
                }
            )
    return pd.DataFrame(rows)


def save_step_durations(milestones: list[Milestone], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [m.label for m in milestones]
    values = [m.duration_minutes for m in milestones]
    colors = [
        "#5B8FF9" if m.kind == "feature" else "#61DDAA" if m.kind == "training" else "#F6BD16" if m.kind == "preprocess" else "#E8684A"
        for m in milestones
    ]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Minutes")
    ax.set_title("Run Step Durations")
    for idx, value in enumerate(values):
        ax.text(value + 0.5, idx, f"{value:.1f}m", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "step_durations.png", dpi=180)
    plt.close(fig)


def save_step_timeline(milestones: list[Milestone], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, milestone in enumerate(milestones):
        ax.barh(
            y=idx,
            width=(milestone.end - milestone.start).total_seconds() / 3600.0,
            left=mdates.date2num(milestone.start),
            height=0.6,
            color="#4E79A7" if milestone.kind != "training" else "#59A14F",
        )
    ax.set_yticks(range(len(milestones)))
    ax.set_yticklabels([m.label for m in milestones])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title("Run Timeline")
    ax.set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(output_dir / "step_timeline.png", dpi=180)
    plt.close(fig)


def save_training_auc(histories: pd.DataFrame, output_dir: Path) -> None:
    approaches = sorted(histories["approach"].unique())
    fig, axes = plt.subplots(len(approaches), 1, figsize=(12, max(4, 4 * len(approaches))), sharex=True)
    if len(approaches) == 1:
        axes = [axes]
    for ax, approach in zip(axes, approaches):
        subset = histories[histories["approach"] == approach]
        for (repeat, fold), grp in subset.groupby(["repeat", "fold"]):
            ax.plot(grp["epoch"], grp["roc_auc_score"], marker="o", linewidth=1.6, label=f"R{repeat} F{fold}")
        ax.set_title(f"{approach}: Validation AUROC by Epoch")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(output_dir / "training_auc_by_epoch.png", dpi=180)
    plt.close(fig)


def save_training_loss(histories: pd.DataFrame, output_dir: Path) -> None:
    approaches = sorted(histories["approach"].unique())
    fig, axes = plt.subplots(len(approaches), 1, figsize=(12, max(4, 4 * len(approaches))), sharex=True)
    if len(approaches) == 1:
        axes = [axes]
    for ax, approach in zip(axes, approaches):
        subset = histories[histories["approach"] == approach]
        for (repeat, fold), grp in subset.groupby(["repeat", "fold"]):
            ax.plot(grp["epoch"], grp["valid_loss"], marker="o", linewidth=1.6, label=f"valid R{repeat} F{fold}")
        ax.set_title(f"{approach}: Validation Loss by Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(output_dir / "training_valid_loss_by_epoch.png", dpi=180)
    plt.close(fig)


def save_fold_progress(histories: pd.DataFrame, output_dir: Path) -> None:
    latest = (
        histories.sort_values(["approach", "repeat", "fold", "epoch"])
        .groupby(["approach", "repeat", "fold"], as_index=False)
        .tail(1)
        .copy()
    )
    latest["label"] = latest.apply(lambda r: f"{r['approach']} R{int(r['repeat'])}F{int(r['fold'])}", axis=1)
    latest = latest.sort_values(["approach", "repeat", "fold"])
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(latest))))
    ax.barh(latest["label"], latest["epoch"] + 1, color="#9C755F")
    ax.set_xlabel("Completed Epochs")
    ax.set_title("Observed Fold Progress")
    for idx, value in enumerate(latest["epoch"] + 1):
        ax.text(value + 0.2, idx, f"{int(value)}/30", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_progress_epochs.png", dpi=180)
    plt.close(fig)


def save_pace_chart(pace_df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        pace_df.groupby(["sample_time", "bucket"], as_index=False)
        .agg(cpu=("cpu", "sum"), process_count=("pid", "count"))
        .sort_values("sample_time")
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for bucket, grp in summary.groupby("bucket"):
        axes[0].plot(grp["sample_time"], grp["cpu"], marker="o", linewidth=2, label=bucket)
        axes[1].plot(grp["sample_time"], grp["process_count"], marker="o", linewidth=2, label=bucket)
    axes[0].set_title("2-Minute CPU Pace Sample")
    axes[0].set_ylabel("Summed CPU %")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].set_title("2-Minute Worker Count Sample")
    axes[1].set_ylabel("Observed processes")
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "pace_cpu_workers_2min.png", dpi=180)
    plt.close(fig)


def write_summary(run_dir: Path, milestones: list[Milestone], histories: pd.DataFrame, pace_df: pd.DataFrame, output_dir: Path) -> None:
    bundle = load_json(run_dir / "bundle_config.json")
    total_requested_folds = len(bundle.get("specs", [])) * bundle.get("request", {}).get("n_folds", 0) * bundle.get("request", {}).get("n_repeats", 0)
    latest = (
        histories.sort_values(["approach", "repeat", "fold", "epoch"])
        .groupby(["approach", "repeat", "fold"], as_index=False)
        .tail(1)
        .copy()
    )
    latest["epochs_completed"] = latest["epoch"] + 1
    latest["best_auc_so_far"] = latest.groupby(["approach", "repeat", "fold"])["roc_auc_score"].transform("max")

    avg_epoch_seconds = histories.groupby("approach")["epoch_seconds"].mean().to_dict()
    mean_cpu = pace_df.groupby("bucket")["cpu"].mean().to_dict() if not pace_df.empty else {}
    training_window = next((m for m in milestones if m.label == "Observed training window"), None)
    fold_equivalents_done = float((latest["epochs_completed"] / 30.0).sum()) if not latest.empty else 0.0
    folds_per_minute = (
        fold_equivalents_done / training_window.duration_minutes
        if training_window and training_window.duration_minutes > 0
        else 0.0
    )
    remaining_fold_equivalents = max(total_requested_folds - fold_equivalents_done, 0.0)
    eta_remaining_minutes = (remaining_fold_equivalents / folds_per_minute) if folds_per_minute > 0 else None

    summary = {
        "bundle_id": bundle.get("bundle_id"),
        "observed_at": datetime.now().isoformat(),
        "milestones": [
            {
                "label": m.label,
                "start": m.start.isoformat(),
                "end": m.end.isoformat(),
                "duration_minutes": round(m.duration_minutes, 2),
                "kind": m.kind,
            }
            for m in milestones
        ],
        "latest_folds": latest[
            ["approach", "repeat", "fold", "epochs_completed", "roc_auc_score", "valid_loss", "best_auc_so_far"]
        ].to_dict("records"),
        "avg_epoch_seconds_by_approach": {k: round(v, 2) for k, v in avg_epoch_seconds.items()},
        "mean_cpu_by_bucket": {k: round(v, 2) for k, v in mean_cpu.items()},
        "fold_equivalents_done": round(fold_equivalents_done, 2),
        "folds_per_minute_estimate": round(folds_per_minute, 3) if folds_per_minute else 0.0,
        "eta_remaining_minutes_estimate": round(eta_remaining_minutes, 1) if eta_remaining_minutes is not None else None,
    }
    (output_dir / "monitor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    observed_folds = latest.shape[0]
    lines = [
        f"Bundle: {bundle.get('bundle_id')}",
        f"Observed fold histories: {observed_folds}",
        f"Requested fold-runs: {total_requested_folds}",
        "",
        "Milestones:",
    ]
    for milestone in milestones:
        lines.append(f"- {milestone.label}: {milestone.start.strftime('%H:%M:%S')} -> {milestone.end.strftime('%H:%M:%S')} ({milestone.duration_minutes:.1f} min)")
    lines.append("")
    lines.append("Latest fold snapshots:")
    for row in latest.sort_values(["approach", "repeat", "fold"]).to_dict("records"):
        lines.append(
            f"- {row['approach']} R{int(row['repeat'])}F{int(row['fold'])}: epoch {int(row['epochs_completed'])}/30, "
            f"AUROC {row['roc_auc_score']:.4f}, valid_loss {row['valid_loss']:.4f}, best AUROC {row['best_auc_so_far']:.4f}"
        )
    lines.append("")
    lines.append("2-minute pace sample:")
    if pace_df.empty:
        lines.append("- No pace rows parsed.")
    else:
        for bucket, value in mean_cpu.items():
            lines.append(f"- {bucket}: mean CPU {value:.1f}")
    if eta_remaining_minutes is not None:
        lines.append("")
        lines.append(
            f"Estimated remaining time at current pace: {eta_remaining_minutes:.1f} minutes "
            f"({eta_remaining_minutes / 60.0:.1f} hours)"
        )

    (output_dir / "monitor_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "monitoring_pngs"
    output_dir.mkdir(parents=True, exist_ok=True)

    milestones = build_milestones(run_dir, args.date)
    histories = load_histories(run_dir)
    pace_df = load_pace_sample(run_dir / "pace_sample_2min.txt", args.date)

    if milestones:
        save_step_durations(milestones, output_dir)
        save_step_timeline(milestones, output_dir)
    if not histories.empty:
        save_training_auc(histories, output_dir)
        save_training_loss(histories, output_dir)
        save_fold_progress(histories, output_dir)
    if not pace_df.empty:
        save_pace_chart(pace_df, output_dir)
    write_summary(run_dir, milestones, histories, pace_df, output_dir)


if __name__ == "__main__":
    main()
