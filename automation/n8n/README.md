# n8n Integration

This folder holds import-ready n8n assets for the Django MSI rebuild.

## Current flow

- Django creates a run blueprint
- optional webhook trigger posts the run payload to n8n
- n8n can fan out into VM launch, archive sync, or notification workflows

## Environment contracts

- `HF_TOKEN` should live in the Django `.env` and on the VM `.env`
- `N8N_ENABLED=true`
- `N8N_BASE_URL=http://127.0.0.1:5678`
- `N8N_WEBHOOK_SECRET=<shared secret>`

## Suggested next uses

1. Launch the VM runner from n8n when a run is created.
2. Poll `status.json` on the VM and push summaries back into Django.
3. Trigger archive import or notification branches after completion.

## VM helper scripts

- `import_workflow_vm.sh <django_rebuild_root>`
- `start_n8n_vm.sh <django_rebuild_root>`

These are intended for the pathology VM where the Django rebuild lives under:

- `/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi`
