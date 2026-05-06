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
            "state",
            "remote_status_path",
            "archive_path",
            "bundle_config_path",
            "remote_launch_log_path",
            "remote_pid",
            "approaches",
        ]
