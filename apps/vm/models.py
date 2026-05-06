from django.db import models


class VMTarget(models.Model):
    name = models.CharField(max_length=120, unique=True)
    execution_mode = models.CharField(max_length=32, default="ssh")
    ssh_user = models.CharField(max_length=120)
    ssh_host = models.CharField(max_length=120)
    ssh_key_path = models.CharField(max_length=255, blank=True)
    conda_env = models.CharField(max_length=120, blank=True)
    runner_python = models.CharField(max_length=255, blank=True)
    project_root = models.CharField(max_length=255)
    is_default = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return self.name
