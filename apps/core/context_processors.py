from pathlib import Path

from django.conf import settings

from apps.approaches.registry import build_approach_slots


def _asset_version() -> str:
    css_path = Path(settings.BASE_DIR) / "apps" / "core" / "static" / "core" / "app.css"
    try:
        return str(int(css_path.stat().st_mtime))
    except OSError:
        return "dev"


def app_shell(request):
    return {
        "app_name": "MSI Control Center",
        "approach_slots": build_approach_slots(),
        "asset_version": _asset_version(),
    }
