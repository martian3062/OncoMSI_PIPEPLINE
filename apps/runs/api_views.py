from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Run
from .serializers import RunSerializer
from .vm_runtime import launch_run_on_vm, sync_latest_archive, sync_run_status
from apps.vm.services import vm_health


@api_view(["GET"])
def health(request):
    return Response({"status": "ok", "service": "django-msi"})


@api_view(["GET"])
def runs_list(request):
    runs = Run.objects.order_by("-created_at")
    return Response(RunSerializer(runs, many=True).data)


@api_view(["GET"])
def vm_status(request):
    return Response(vm_health())


@api_view(["POST"])
def launch_run(request, run_id: str):
    run = get_object_or_404(Run, run_id=run_id)
    return Response(launch_run_on_vm(run))


@api_view(["POST"])
def sync_run(request, run_id: str):
    run = get_object_or_404(Run, run_id=run_id)
    return Response(sync_run_status(run))


@api_view(["POST"])
def sync_archive(request):
    return Response(sync_latest_archive())
