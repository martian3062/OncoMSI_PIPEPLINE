from django.contrib import admin

from .models import VMTarget


@admin.register(VMTarget)
class VMTargetAdmin(admin.ModelAdmin):
    list_display = ("name", "execution_mode", "ssh_user", "ssh_host", "conda_env", "is_default")
    list_filter = ("is_default",)
    search_fields = ("name", "ssh_user", "ssh_host", "project_root")
