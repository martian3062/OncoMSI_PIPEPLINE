from django.conf import settings


def integration_summary() -> dict:
    return {
        "hf_configured": bool(settings.HF_TOKEN),
        "n8n_enabled": settings.N8N_ENABLED,
        "n8n_base_url": settings.N8N_BASE_URL,
        "n8n_workflow_path": settings.N8N_WORKFLOW_PATH,
    }
