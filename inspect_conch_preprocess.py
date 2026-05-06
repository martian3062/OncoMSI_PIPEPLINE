import os, sys
from pathlib import Path
SCRIPT_DIR = Path('/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    raise RuntimeError('Set HF_TOKEN before running this script.')
from conch.open_clip_custom import create_model_from_pretrained
model, preprocess = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/CONCH', hf_auth_token=HF_TOKEN)
print(type(preprocess))
print(preprocess)
