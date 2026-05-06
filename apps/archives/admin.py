from django.contrib import admin

from .models import BatchArchive


@admin.register(BatchArchive)
class BatchArchiveAdmin(admin.ModelAdmin):
    list_display = ("archive_root", "state", "completed_batches", "updated_at")
    list_filter = ("state",)
    search_fields = ("archive_root", "orchestration_status_path")
