from apps.approaches.registry import sync_default_approaches
from apps.runs.services import create_run_from_payload
from apps.runs.vm_runtime import launch_run_on_vm
from apps.vm.registry import ensure_default_vm_target

sync_default_approaches()

target = ensure_default_vm_target()
target.runner_python = '/home/pardeep/.venvs/pathology310-hybrid/bin/python'
target.save(update_fields=['runner_python', 'updated_at'])

payload = {
    'experiment_name': 'hybrid-full-150x10f-7enc-256tiles',
    'source_uri': 'gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD',
    'annotations_csv': '/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/tcga3_vm_annotations.csv',
    'requested_slide_limit': 150,
    'n_folds': 10,
    'n_repeats': 1,
    'max_tiles_per_slide': 256,
    'feature_extractor_candidates': ['virchow', 'retccl', 'ctranspath', 'conch', 'virchow2', 'uni2-h', 'h-optimus-0'],
    'n8n_webhook_url': '',
}
run = create_run_from_payload(payload)

per_key = {
    'approach1': {'feature_extractor': 'virchow', 'extractor_backend': 'slideflow', 'epochs': 30, 'learning_rate': 2.5e-5, 'weight_decay': 8e-5, 'bag_size': 160, 'max_val_bag_size': 160, 'mil_batch_size': 10, 'weighted_loss': True, 'fit_one_cycle': True, 'seed': 310, 'launch_enabled': True},
    'approach2': {'feature_extractor': 'retccl', 'extractor_backend': 'slideflow', 'epochs': 30, 'learning_rate': 5e-5, 'weight_decay': 1e-4, 'bag_size': 160, 'max_val_bag_size': 160, 'mil_batch_size': 12, 'weighted_loss': True, 'fit_one_cycle': True, 'seed': 310, 'launch_enabled': True},
    'approach3': {'feature_extractor': 'ctranspath', 'extractor_backend': 'slideflow', 'epochs': 30, 'learning_rate': 4e-5, 'weight_decay': 1e-4, 'bag_size': 160, 'max_val_bag_size': 160, 'mil_batch_size': 12, 'weighted_loss': True, 'fit_one_cycle': True, 'seed': 310, 'launch_enabled': True},
    'approach4': {'feature_extractor': 'conch', 'extractor_backend': 'hybrid', 'epochs': 30, 'learning_rate': 3e-5, 'weight_decay': 8e-5, 'bag_size': 160, 'max_val_bag_size': 160, 'mil_batch_size': 10, 'weighted_loss': True, 'fit_one_cycle': True, 'seed': 310, 'launch_enabled': True},
    'approach5': {'feature_extractor': 'virchow2', 'extractor_backend': 'hybrid', 'epochs': 30, 'learning_rate': 2e-5, 'weight_decay': 6e-5, 'bag_size': 160, 'max_val_bag_size': 160, 'mil_batch_size': 8, 'weighted_loss': True, 'fit_one_cycle': True, 'seed': 310, 'launch_enabled': True},
    'approach6': {'feature_extractor': 'uni2-h', 'extractor_backend': 'hybrid', 'epochs': 30, 'learning_rate': 3e-5, 'weight_decay': 8e-5, 'bag_size': 160, 'max_val_bag_size': 160, 'mil_batch_size': 10, 'weighted_loss': True, 'fit_one_cycle': True, 'seed': 310, 'launch_enabled': True},
    'approach7': {'feature_extractor': 'h-optimus-0', 'extractor_backend': 'hybrid', 'epochs': 30, 'learning_rate': 3e-5, 'weight_decay': 8e-5, 'bag_size': 160, 'max_val_bag_size': 160, 'mil_batch_size': 10, 'weighted_loss': True, 'fit_one_cycle': True, 'seed': 310, 'launch_enabled': True},
}

for link in run.approach_links.select_related('approach_template').all():
    if link.approach_template.key in per_key:
        params = dict(link.trainer_params or {})
        params.update(per_key[link.approach_template.key])
        link.trainer_params = params
        link.save(update_fields=['trainer_params', 'updated_at'])

result = launch_run_on_vm(run)
print(run.run_id)
print(result)
