from __future__ import annotations

from apps.approaches.registry import sync_default_approaches
from apps.runs.services import create_run_from_payload
from apps.runs.vm_runtime import launch_run_on_vm
from apps.vm.registry import ensure_default_vm_target


TOP4_KEYS = {
    "approach1": {
        "feature_extractor": "uni2-h",
        "extractor_backend": "hybrid",
        "epochs": 30,
        "learning_rate": 3e-5,
        "weight_decay": 8e-5,
        "bag_size": 160,
        "max_val_bag_size": 160,
        "mil_batch_size": 10,
        "weighted_loss": True,
        "fit_one_cycle": True,
        "seed": 310,
        "launch_enabled": True,
        "strict_feature_extractor": True,
    },
    "approach2": {
        "feature_extractor": "virchow2",
        "extractor_backend": "hybrid",
        "epochs": 30,
        "learning_rate": 3e-5,
        "weight_decay": 8e-5,
        "bag_size": 160,
        "max_val_bag_size": 160,
        "mil_batch_size": 10,
        "weighted_loss": True,
        "fit_one_cycle": True,
        "seed": 310,
        "launch_enabled": True,
        "strict_feature_extractor": True,
    },
    "approach5": {
        "feature_extractor": "h-optimus-0",
        "extractor_backend": "hybrid",
        "epochs": 30,
        "learning_rate": 4e-5,
        "weight_decay": 1e-4,
        "bag_size": 160,
        "max_val_bag_size": 160,
        "mil_batch_size": 12,
        "weighted_loss": True,
        "fit_one_cycle": True,
        "seed": 310,
        "launch_enabled": True,
        "strict_feature_extractor": True,
    },
    "approach6": {
        "feature_extractor": "midnight",
        "extractor_backend": "hybrid",
        "epochs": 30,
        "learning_rate": 4e-5,
        "weight_decay": 1e-4,
        "bag_size": 160,
        "max_val_bag_size": 160,
        "mil_batch_size": 12,
        "weighted_loss": True,
        "fit_one_cycle": True,
        "seed": 310,
        "launch_enabled": True,
        "strict_feature_extractor": True,
    },
}


sync_default_approaches()

target = ensure_default_vm_target()
target.runner_python = "/home/pardeep/.venvs/pathology310-hybrid/bin/python"
target.save(update_fields=["runner_python", "updated_at"])

payload = {
    "experiment_name": "final-top4-200x10f-seed310",
    "source_uri": "gs://wsi_aiml_repo/TCGA/TCGA_COAD/TCGA_COAD",
    "annotations_csv": "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/tcga3_vm_annotations.csv",
    "requested_slide_limit": 200,
    "n_folds": 10,
    "n_repeats": 1,
    "max_tiles_per_slide": 256,
    "feature_extractor_candidates": ["virchow2", "midnight", "uni2-h", "h-optimus-0"],
    "external_cohorts": [
        {
            "name": "CPTAC-COAD",
            "annotations_csv": "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/annotations/cptac_coad_annotations.csv",
            "source_uri": "gs://wsi_aiml_repo/CPTAC/CPTAC_COAD",
        }
    ],
    "n8n_webhook_url": "",
}

run = create_run_from_payload(payload)

for link in list(run.approach_links.select_related("approach_template").all()):
    params = TOP4_KEYS.get(link.approach_template.key)
    if params is None:
        link.delete()
        continue
    merged = dict(link.trainer_params or {})
    merged.update(params)
    link.trainer_params = merged
    link.save(update_fields=["trainer_params", "updated_at"])

result = launch_run_on_vm(run)
print(run.run_id)
print(result)
