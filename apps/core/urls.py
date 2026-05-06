from django.urls import path

from .views import dashboard, dashboard_metrics_partial, launch_run, live_runs_partial


urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("partials/metrics/", dashboard_metrics_partial, name="dashboard-metrics-partial"),
    path("partials/live-runs/", live_runs_partial, name="live-runs-partial"),
    path("runs/launch/", launch_run, name="launch-run"),
]
