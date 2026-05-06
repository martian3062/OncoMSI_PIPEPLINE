from apps.approaches.registry import sync_default_approaches
from apps.approaches.models import ApproachTemplate
from apps.runs.services import create_run_from_payload
from apps.runs.vm_runtime import launch_run_on_vm

sync_default_approaches()
print('approaches', list(ApproachTemplate.objects.order_by('key').values_list('key', 'label')))

payload = {
    'experiment_name': 'hybrid-smoke-24x2f-7enc-96tiles',
    'source_uri': 'gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD',
    'annotations_csv': '/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/tcga3_vm_annotations.csv',
    'requested_slide_limit': 24,
    'n_folds': 2,
    'n_repeats': 1,
    'max_tiles_per_slide': 96,
    'feature_extractor_candidates': ['virchow', 'retccl', 'ctranspath', 'conch', 'virchow2', 'uni2-h', 'h-optimus-0'],
    'n8n_webhook_url': '',
}
run = create_run_from_payload(payload)
print('run_id', run.run_id)
print(launch_run_on_vm(run))
