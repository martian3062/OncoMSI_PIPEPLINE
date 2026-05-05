# Django MSI Rebuild

This workspace is a greenfield Django rebuild of the TCGA MSI control system
described in [generic.md](./generic.md).

## Stack

- Django 5
- Django REST Framework
- HTMX
- Alpine.js
- Plotly
- Plain CSS with gradient/glass styling
- n8n-ready webhook integration

## Architecture

```text
apps/
  core/         # Django shell, HTMX pages, Alpine-driven interactions
  vm/           # VM target definitions and remote execution settings
  runs/         # Run/bundle lifecycle, API, orchestration entry points
  approaches/   # Pluggable approach templates and two default slots
  archives/     # Archived batch summaries and long-lived comparisons
  vm/           # VM target registry plus local/ssh execution helpers
msi_platform/   # Django project config
```

## Why this shape

The rebuild keeps the UI inside Django instead of a separate frontend. That
means:

- HTMX handles partial reloads and launch actions
- Alpine.js handles tiny client-state toggles
- Plotly turns Python-generated chart payloads into dashboard visuals
- the system remains modular enough to swap in real VM orchestration later

## Default experiment surface

The scaffold comes with two approach slots:

- `approach1` -> `transmil`
- `approach2` -> `attention_mil`

These act as extension spaces for the two-approach parallel contract from the
previous system.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py createsuperuser
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/admin/`
- `http://127.0.0.1:8000/api/health/`

## Environment and n8n

The app now reads `.env` values directly at startup.

- `HF_TOKEN` supports gated Hugging Face extractor access
- `N8N_ENABLED` toggles webhook triggering
- `N8N_BASE_URL` documents the intended local/remote n8n endpoint
- `N8N_WEBHOOK_SECRET` can be used to sign webhook payloads
- `N8N_WORKFLOW_PATH` points to the default importable workflow JSON
- `VM_EXECUTION_MODE` chooses `ssh` or `local`
- `VM_PROJECT_ROOT`, `VM_RUNNER_SCRIPT`, and `VM_RUNNER_PYTHON` point Django at the real pathology runner

Starter n8n assets live in [automation/n8n](./automation/n8n/).

## Real VM API endpoints

- `GET /api/vm/status/`
- `POST /api/runs/<run_id>/launch-vm/`
- `POST /api/runs/<run_id>/sync-status/`
- `POST /api/archives/sync-latest/`

# OncoMSI_PIPEPLINE

## Next recommended build steps

1. Add real VM orchestration services for bundle launch/status sync.
2. Persist archive summaries from remote `orchestration_status.json`.
3. Add per-run detail pages with phase timelines and approach metrics.
4. Replace the placeholder run-creation flow with real remote execution hooks.
