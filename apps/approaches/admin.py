from django.contrib import admin

from .models import ApproachTemplate


@admin.register(ApproachTemplate)
class ApproachTemplateAdmin(admin.ModelAdmin):
    list_display = ("key", "label", "model_family", "is_active")
    list_filter = ("is_active",)
    search_fields = ("key", "label", "model_family")
