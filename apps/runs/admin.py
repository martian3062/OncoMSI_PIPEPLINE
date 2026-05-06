from django.contrib import admin

from .models import Run, RunApproachLink


class RunApproachInline(admin.TabularInline):
    model = RunApproachLink
    extra = 0


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    list_display = ("run_id", "experiment_name", "state", "requested_slide_limit", "feature_extractor_used", "created_at")
    list_filter = ("state", "feature_extractor_used")
    search_fields = ("run_id", "experiment_name", "source_uri", "annotations_csv")
    inlines = [RunApproachInline]
