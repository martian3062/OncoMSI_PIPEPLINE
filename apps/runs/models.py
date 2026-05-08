import uuid

from django.db import models

from apps.approaches.models import ApproachTemplate
from apps.vm.models import VMTarget


class RunState(models.TextChoices):
    MATCHING_ANNOTATIONS = "matching_annotations", "Matching annotations"
    DOWNLOADING_SLIDES = "downloading_slides", "Downloading slides"
    EXTRACTING_TILES = "extracting_tiles", "Extracting tiles"
    RETRYING_TILES = "retrying_tiles", "Retrying tiles"
    GENERATING_FEATURES = "generating_features", "Generating features"
    PREPARED = "prepared", "Prepared"
    TRAINING_PARALLEL = "training_parallel", "Training parallel"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class Run(models.Model):
    run_id = models.CharField(max_length=64, unique=True, default="", editable=False)
    experiment_name = models.CharField(max_length=255)
    source_type = models.CharField(max_length=64, default="gcs_bucket")
    source_uri = models.CharField(max_length=255)
    annotations_csv = models.CharField(max_length=255)
    requested_slide_limit = models.PositiveIntegerField(default=0)
    selected_slide_count = models.PositiveIntegerField(default=0)
    label_counts = models.JSONField(default=dict, blank=True)
    feature_extractor_candidates = models.JSONField(default=list, blank=True)
    feature_extractor_used = models.CharField(max_length=120, blank=True)
    tile_px = models.PositiveIntegerField(default=256)
    tile_um = models.PositiveIntegerField(default=128)
    max_tiles_per_slide = models.PositiveIntegerField(default=96)
    n_folds = models.PositiveIntegerField(default=2)
    n_repeats = models.PositiveIntegerField(default=2)
    repeat_seeds = models.JSONField(default=list, blank=True)
    external_cohorts = models.JSONField(default=list, blank=True)
    state = models.CharField(max_length=64, choices=RunState.choices, default=RunState.MATCHING_ANNOTATIONS)
    remote_status_path = models.CharField(max_length=255, blank=True)
    archive_path = models.CharField(max_length=255, blank=True)
    bundle_config_path = models.CharField(max_length=255, blank=True)
    remote_launch_log_path = models.CharField(max_length=255, blank=True)
    remote_pid = models.CharField(max_length=64, blank=True)
    vm_target = models.ForeignKey(VMTarget, null=True, blank=True, on_delete=models.SET_NULL, related_name="runs")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.run_id:
            self.run_id = f"run-{uuid.uuid4().hex[:12]}"
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return self.run_id


class RunApproachLink(models.Model):
    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="approach_links")
    approach_template = models.ForeignKey(ApproachTemplate, on_delete=models.PROTECT, related_name="run_links")
    state = models.CharField(max_length=64, default="pending")
    trainer_params = models.JSONField(default=dict, blank=True)
    mean_auroc = models.FloatField(null=True, blank=True)
    mean_f1_macro = models.FloatField(null=True, blank=True)
    mean_f1_macro_default_threshold = models.FloatField(null=True, blank=True)
    mean_auprc = models.FloatField(null=True, blank=True)
    mean_balanced_accuracy = models.FloatField(null=True, blank=True)
    mean_precision = models.FloatField(null=True, blank=True)
    mean_recall_msi_h = models.FloatField(null=True, blank=True)
    mean_specificity = models.FloatField(null=True, blank=True)
    mean_best_threshold = models.FloatField(null=True, blank=True)
    mean_brier_score = models.FloatField(null=True, blank=True)
    auroc_std = models.FloatField(null=True, blank=True)
    auroc_ci_low = models.FloatField(null=True, blank=True)
    auroc_ci_high = models.FloatField(null=True, blank=True)
    auroc_per_fold = models.JSONField(default=list, blank=True)
    fold_metrics = models.JSONField(default=list, blank=True)
    aggregate_confusion_matrix = models.JSONField(default=dict, blank=True)
    available_bag_slide_count = models.PositiveIntegerField(null=True, blank=True)
    missing_bag_slides = models.JSONField(default=list, blank=True)
    external_metrics = models.JSONField(default=dict, blank=True)
    prediction_artifacts = models.JSONField(default=dict, blank=True)
    metrics_path = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("run", "approach_template")

    def __str__(self) -> str:
        return f"{self.run.run_id}:{self.approach_template.key}"
