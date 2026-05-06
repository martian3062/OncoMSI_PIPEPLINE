from django.conf import settings

from .models import ApproachTemplate


def sync_default_approaches() -> None:
    configured_keys = [slot["key"] for slot in settings.MSI_DEFAULT_APPROACHES]
    existing = {
        template.key: template
        for template in ApproachTemplate.objects.filter(key__in=configured_keys)
    }
    for slot in settings.MSI_DEFAULT_APPROACHES:
        defaults = {
            "label": slot["label"],
            "model_family": slot["model_family"],
            "color_token": slot["color_token"],
            "default_params": slot["default_params"],
            "is_active": True,
        }
        current = existing.get(slot["key"])
        if current is None:
            ApproachTemplate.objects.create(key=slot["key"], **defaults)
            continue

        changed_fields: list[str] = []
        for field, value in defaults.items():
            if getattr(current, field) != value:
                setattr(current, field, value)
                changed_fields.append(field)
        if changed_fields:
            current.save(update_fields=[*changed_fields, "updated_at"])

    stale_templates = ApproachTemplate.objects.exclude(key__in=configured_keys).filter(is_active=True)
    stale_templates.update(is_active=False)


def build_approach_slots():
    slots = list(ApproachTemplate.objects.filter(is_active=True).order_by("key"))
    if slots:
        return slots
    return settings.MSI_DEFAULT_APPROACHES


def build_launch_slots():
    slots = build_approach_slots()
    launch_slots = []
    for slot in slots:
        params = getattr(slot, "default_params", {}) or {}
        if params.get("launch_enabled", True):
            launch_slots.append(slot)
    return launch_slots
