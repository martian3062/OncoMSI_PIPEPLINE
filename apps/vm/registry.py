from django.conf import settings

from .models import VMTarget


def ensure_default_vm_target() -> VMTarget:
    target, _ = VMTarget.objects.update_or_create(
        name=settings.VM_NAME,
        defaults={
            "execution_mode": settings.VM_EXECUTION_MODE,
            "ssh_user": settings.VM_SSH_USER,
            "ssh_host": settings.VM_SSH_HOST,
            "ssh_key_path": settings.VM_SSH_KEY_PATH,
            "conda_env": settings.VM_CONDA_ENV,
            "runner_python": settings.VM_RUNNER_PYTHON,
            "project_root": settings.VM_PROJECT_ROOT,
            "is_default": True,
        },
    )
    VMTarget.objects.exclude(pk=target.pk).filter(is_default=True).update(is_default=False)
    return target
