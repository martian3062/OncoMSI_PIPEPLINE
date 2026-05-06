from django.urls import path

from .api_views import health, launch_run, runs_list, sync_archive, sync_run, vm_status


urlpatterns = [
    path("health/", health, name="api-health"),
    path("runs/", runs_list, name="api-runs-list"),
    path("vm/status/", vm_status, name="api-vm-status"),
    path("runs/<str:run_id>/launch-vm/", launch_run, name="api-run-launch-vm"),
    path("runs/<str:run_id>/sync-status/", sync_run, name="api-run-sync-status"),
    path("archives/sync-latest/", sync_archive, name="api-archive-sync-latest"),
]
