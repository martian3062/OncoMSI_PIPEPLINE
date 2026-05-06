from django.db import models


class ApproachTemplate(models.Model):
    key = models.SlugField(unique=True)
    label = models.CharField(max_length=120)
    model_family = models.CharField(max_length=120)
    color_token = models.CharField(max_length=64, blank=True)
    default_params = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return self.label
