from apps.approaches.registry import build_approach_slots


def app_shell(request):
    return {
        "app_name": "MSI Control Center",
        "approach_slots": build_approach_slots(),
    }
