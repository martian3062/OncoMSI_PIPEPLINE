import os
import subprocess
from pathlib import Path

CFG = '/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/bundle_configs/run-7808c90045e9.json'
ROOT = Path('/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/automation/tcga_slide_triads/run-7808c90045e9')
PY = '/opt/miniforge3/envs/pathology310/bin/python'
SCRIPT = '/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/scripts/run_tcga_coad_automated_triad.py'
labels = [
    'Approach1-Virchow',
    'Approach2-RetCCL',
    'Approach3-CTransPath',
    'Approach4-CONCH',
    'Approach5-Virchow2',
    'Approach6-UNI2-H',
    'Approach7-H-Optimus-0',
]
for label in labels:
    outdir = ROOT / 'approaches' / label
    outdir.mkdir(parents=True, exist_ok=True)
    log = open(outdir / 'runner.log', 'ab')
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env.pop('PYTHONPATH', None)
    env.pop('VIRTUAL_ENV', None)
    proc = subprocess.Popen(
        [PY, SCRIPT, '--bundle-config', CFG, '--stage', 'train-approach', '--approach-label', label],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    print(label, proc.pid)
