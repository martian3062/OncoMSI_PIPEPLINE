from rest_framework import serializers

from .models import Run, RunApproachLink


class RunApproachLinkSerializer(serializers.ModelSerializer):
    approach = serializers.CharField(source="approach_template.label", read_only=True)

    class Meta:
        model = RunApproachLink
        fields = [
            "approach",
            "state",
            "trainer_params",
            "mean_auroc",
            "mean_f1_macro",
            "mean_f1_macro_default_threshold",
            "mean_auprc",
            "mean_balanced_accuracy",
            "mean_precision",
            "mean_recall_msi_h",
            "mean_specificity",
            "mean_best_threshold",
            "mean_brier_score",
            "auroc_std",
            "auroc_ci_low",
            "auroc_ci_high",
            "auroc_per_fold",
            "fold_metrics",
            "aggregate_confusion_matrix",
            "available_bag_slide_count",
            "missing_bag_slides",
            "external_metrics",
            "metrics_path",
        ]


class RunSerializer(serializers.ModelSerializer):
    approaches = RunApproachLinkSerializer(source="approach_links", many=True, read_only=True)

    class Meta:
        model = Run
        fields = [
            "run_id",
            "experiment_name",
            "source_type",
            "source_uri",
            "annotations_csv",
            "requested_slide_limit",
            "selected_slide_count",
            "label_counts",
            "feature_extractor_candidates",
            "feature_extractor_used",
            "tile_px",
            "tile_um",
            "max_tiles_per_slide",
            "n_folds",
            "n_repeats",
            "repeat_seeds",
            "external_cohorts",
            "state",
            "remote_status_path",
            "archive_path",
            "bundle_config_path",
            "remote_launch_log_path",
            "remote_pid",
            "approaches",
        ]
