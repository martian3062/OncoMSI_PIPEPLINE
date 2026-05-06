from django.db import models


class BatchArchive(models.Model):
    archive_root = models.CharField(max_length=255, unique=True)
    state = models.CharField(max_length=64, default="pending")
    completed_batches = models.PositiveIntegerField(default=0)
    aggregate_label_counts = models.JSONField(default=dict, blank=True)
    aggregate_approaches = models.JSONField(default=dict, blank=True)
    orchestration_status_path = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return self.archive_root
