from django.conf import settings

from .models import ApproachTemplate


def sync_default_approaches() -> None:
    for slot in settings.MSI_DEFAULT_APPROACHES:
        ApproachTemplate.objects.update_or_create(
            key=slot["key"],
            defaults={
                "label": slot["label"],
                "model_family": slot["model_family"],
                "color_token": slot["color_token"],
                "default_params": slot["default_params"],
                "is_active": True,
            },
        )


def build_approach_slots():
    slots = list(ApproachTemplate.objects.filter(is_active=True).order_by("key"))
    if slots:
        return slots
    return settings.MSI_DEFAULT_APPROACHES
