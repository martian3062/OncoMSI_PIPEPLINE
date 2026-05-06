from django.conf import settings

from .models import VMTarget


def get_default_vm_target() -> VMTarget | None:
    target = VMTarget.objects.filter(is_default=True).first()
    if target is not None:
        return target
    return VMTarget.objects.filter(name=settings.VM_NAME).first()


def ensure_default_vm_target() -> VMTarget:
    defaults = {
        "execution_mode": settings.VM_EXECUTION_MODE,
        "ssh_user": settings.VM_SSH_USER,
        "ssh_host": settings.VM_SSH_HOST,
        "ssh_key_path": settings.VM_SSH_KEY_PATH,
        "conda_env": settings.VM_CONDA_ENV,
        "runner_python": settings.VM_RUNNER_PYTHON,
        "project_root": settings.VM_PROJECT_ROOT,
        "is_default": True,
    }
    target = get_default_vm_target()
    if target is None:
        target = VMTarget.objects.create(name=settings.VM_NAME, **defaults)
    else:
        changed_fields: list[str] = []
        if target.name != settings.VM_NAME:
            target.name = settings.VM_NAME
            changed_fields.append("name")
        for field, value in defaults.items():
            if getattr(target, field) != value:
                setattr(target, field, value)
                changed_fields.append(field)
        if changed_fields:
            target.save(update_fields=[*changed_fields, "updated_at"])
    VMTarget.objects.exclude(pk=target.pk).filter(is_default=True).update(is_default=False)
    return target
