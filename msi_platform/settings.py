import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

def _load_env_file() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

SECRET_KEY = "django-insecure-change-me-for-production"
DEBUG = True
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver,34.59.145.240").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "apps.core",
    "apps.vm",
    "apps.runs",
    "apps.approaches",
    "apps.archives",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "msi_platform.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "apps.core.context_processors.app_shell",
            ],
        },
    },
]

WSGI_APPLICATION = "msi_platform.wsgi.application"
ASGI_APPLICATION = "msi_platform.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {
            "timeout": 30,
            "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=30000;",
        },
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
}

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MSI_LOCAL_ENCODER_DIR = os.environ.get("MSI_LOCAL_ENCODER_DIR", "")
MSI_INFERENCE_BUNDLE_DIR = os.environ.get("MSI_INFERENCE_BUNDLE_DIR", "")
MSI_PREFER_DEVICE = os.environ.get("MSI_PREFER_DEVICE", "auto")
MSI_PIPELINE_MODE = os.environ.get("MSI_PIPELINE_MODE", "manager1")
MSI_MAX_INFERENCE_TILES = int(os.environ.get("MSI_MAX_INFERENCE_TILES", "24"))
MSI_FAST_MAX_TILES = int(os.environ.get("MSI_FAST_MAX_TILES", "64"))
MSI_FAST_CHECKPOINTS = int(os.environ.get("MSI_FAST_CHECKPOINTS", "4"))
MSI_EXACT_MAX_TILES = int(os.environ.get("MSI_EXACT_MAX_TILES", "0"))
MSI_EXACT_PREVIEW_TILES = int(os.environ.get("MSI_EXACT_PREVIEW_TILES", "6"))
MSI_EXACT_TILE_THREADS = int(os.environ.get("MSI_EXACT_TILE_THREADS", "4"))
MSI_FAST_TILE_READ_WORKERS = int(os.environ.get("MSI_FAST_TILE_READ_WORKERS", "8"))
MSI_IMAGE_TILE_WORKERS = int(os.environ.get("MSI_IMAGE_TILE_WORKERS", "8"))
MSI_ENCODE_PREPROCESS_WORKERS = int(os.environ.get("MSI_ENCODE_PREPROCESS_WORKERS", "8"))
MSI_ENCODE_BATCH_SIZE = int(os.environ.get("MSI_ENCODE_BATCH_SIZE", "0"))
MSI_SCORE_WORKERS = int(os.environ.get("MSI_SCORE_WORKERS", "8"))
MSI_JOB_CAPACITY_LIMIT = int(os.environ.get("MSI_JOB_CAPACITY_LIMIT", "2"))
MSI_EXACT_JOB_WEIGHT = int(os.environ.get("MSI_EXACT_JOB_WEIGHT", "2"))
MSI_FAST_JOB_WEIGHT = int(os.environ.get("MSI_FAST_JOB_WEIGHT", "1"))
NEXT_APP_URL = os.environ.get("NEXT_APP_URL", "http://127.0.0.1:3000")
N8N_ENABLED = os.environ.get("N8N_ENABLED", "true").lower() == "true"
N8N_BASE_URL = os.environ.get("N8N_BASE_URL", "http://127.0.0.1:5678")
N8N_WEBHOOK_SECRET = os.environ.get("N8N_WEBHOOK_SECRET", "")
N8N_WORKFLOW_PATH = os.environ.get("N8N_WORKFLOW_PATH", "automation/n8n/msi_django_launch.json")
VM_EXECUTION_MODE = os.environ.get("VM_EXECUTION_MODE", "ssh" if os.name == "nt" else "local")
VM_NAME = os.environ.get("VM_NAME", "pathology310-primary")
VM_SSH_USER = os.environ.get("VM_SSH_USER", "pardeep")
VM_SSH_HOST = os.environ.get("VM_SSH_HOST", "34.59.145.240")
VM_SSH_KEY_PATH = os.environ.get("VM_SSH_KEY_PATH", str(Path.home() / ".ssh" / "evolet_rsa") if os.name == "nt" else "")
VM_CONDA_ENV = os.environ.get("VM_CONDA_ENV", "pathology310")
VM_RUNNER_PYTHON = os.environ.get(
    "VM_RUNNER_PYTHON",
    "/home/pardeep/.venvs/pathology310-hybrid/bin/python",
)
VM_PROJECT_ROOT = os.environ.get(
    "VM_PROJECT_ROOT",
    "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc",
)
VM_RUNNER_SCRIPT = os.environ.get(
    "VM_RUNNER_SCRIPT",
    "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/scripts/run_tcga_coad_automated_triad.py",
)
VM_ARCHIVE_GLOB = os.environ.get(
    "VM_ARCHIVE_GLOB",
    "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/automation/tcga_batch_archives*",
)
VM_LIVE_BUNDLE_GLOB = os.environ.get(
    "VM_LIVE_BUNDLE_GLOB",
    "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/automation/tcga_slide_triads/*",
)
VM_DEFAULT_ANNOTATIONS = os.environ.get(
    "VM_DEFAULT_ANNOTATIONS",
    "annotations/tcga_coad_bucket_annotations_final_all3_live_dx1.csv",
)
VM_VIRCHOW_WEIGHTS = os.environ.get(
    "VM_VIRCHOW_WEIGHTS",
    "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/models/virchow/pytorch_model.bin",
)
VM_STUDENT_ENCODER_DIR = os.environ.get("VM_STUDENT_ENCODER_DIR", "")

MSI_DEFAULT_APPROACHES = [
    {
        "key": "approach1",
        "label": "Approach 1 - Virchow",
        "model_family": "transmil",
        "color_token": "var(--accent-cyan)",
        "default_params": {
            "feature_extractor": "virchow",
            "epochs": 30,
            "learning_rate": 0.00003,
            "weight_decay": 0.00008,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 10,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
        },
    },
    {
        "key": "approach2",
        "label": "Approach 2 - RetCCL",
        "model_family": "transmil",
        "color_token": "var(--accent-coral)",
        "default_params": {
            "feature_extractor": "retccl",
            "epochs": 30,
            "learning_rate": 0.00005,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
        },
    },
    {
        "key": "approach3",
        "label": "Approach 3 - CTransPath",
        "model_family": "transmil",
        "color_token": "var(--accent-amber)",
        "default_params": {
            "feature_extractor": "ctranspath",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
        },
    },
]

# hybrid-02 replaces the previous live roster while preserving the older seven
# inside results-history/roster_snapshots/hybrid_01_legacy_seven_snapshot.json.
MSI_DEFAULT_APPROACHES = [
    {
        "key": "approach1",
        "label": "Approach 1 - CONCHv1.5",
        "model_family": "transmil",
        "color_token": "var(--accent-cyan)",
        "default_params": {
            "feature_extractor": "conchv1_5",
            "epochs": 30,
            "learning_rate": 0.00003,
            "weight_decay": 0.00008,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 10,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
        },
    },
    {
        "key": "approach2",
        "label": "Approach 2 - Phikon-v2",
        "model_family": "transmil",
        "color_token": "var(--accent-coral)",
        "default_params": {
            "feature_extractor": "phikon-v2",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
        },
    },
    {
        "key": "approach3",
        "label": "Approach 3 - Prov-GigaPath",
        "model_family": "transmil",
        "color_token": "var(--accent-amber)",
        "default_params": {
            "feature_extractor": "prov-gigapath",
            "epochs": 30,
            "learning_rate": 0.00003,
            "weight_decay": 0.00008,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 10,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
        },
    },
    {
        "key": "approach4",
        "label": "Approach 4 - PRISM",
        "model_family": "transmil",
        "color_token": "var(--accent-lime)",
        "default_params": {
            "feature_extractor": "prism-virchow",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "proxy_source": "paige-ai/Virchow",
        },
    },
    {
        "key": "approach5",
        "label": "Approach 5 - CHIEF",
        "model_family": "transmil",
        "color_token": "var(--accent-cyan)",
        "default_params": {
            "feature_extractor": "chief-ctranspath",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "slideflow",
            "proxy_source": "ctranspath",
        },
    },
    {
        "key": "approach6",
        "label": "Approach 6 - DINOv2-Large",
        "model_family": "transmil",
        "color_token": "var(--accent-coral)",
        "default_params": {
            "feature_extractor": "dinov2-large",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
        },
    },
    {
        "key": "approach7",
        "label": "Approach 7 - Midnight-12k",
        "model_family": "transmil",
        "color_token": "var(--accent-amber)",
        "default_params": {
            "feature_extractor": "midnight",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
        },
    },
]

# Override the early scaffold defaults with the active semi-final experiment catalog.
MSI_DEFAULT_APPROACHES = [
    {
        "key": "approach1",
        "label": "Approach 1 - UNI2-h",
        "model_family": "transmil",
        "color_token": "var(--accent-cyan)",
        "default_params": {
            "feature_extractor": "uni2-h",
            "epochs": 30,
            "learning_rate": 0.00003,
            "weight_decay": 0.00008,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 10,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach2",
        "label": "Approach 2 - Virchow2",
        "model_family": "transmil",
        "color_token": "var(--accent-coral)",
        "default_params": {
            "feature_extractor": "virchow2",
            "epochs": 30,
            "learning_rate": 0.00003,
            "weight_decay": 0.00008,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 10,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach3",
        "label": "Approach 3 - Prov-GigaPath",
        "model_family": "transmil",
        "color_token": "var(--accent-amber)",
        "default_params": {
            "feature_extractor": "prov-gigapath",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach4",
        "label": "Approach 4 - CONCHv1.5",
        "model_family": "transmil",
        "color_token": "var(--accent-lime)",
        "default_params": {
            "feature_extractor": "conchv1_5",
            "epochs": 30,
            "learning_rate": 0.00003,
            "weight_decay": 0.00008,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 10,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach5",
        "label": "Approach 5 - H-Optimus-0",
        "model_family": "transmil",
        "color_token": "var(--accent-cyan)",
        "default_params": {
            "feature_extractor": "h-optimus-0",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach6",
        "label": "Approach 6 - Midnight-12k",
        "model_family": "transmil",
        "color_token": "var(--accent-coral)",
        "default_params": {
            "feature_extractor": "midnight",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach7",
        "label": "Approach 7 - DINOv2-Large",
        "model_family": "transmil",
        "color_token": "var(--accent-amber)",
        "default_params": {
            "feature_extractor": "dinov2-large",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach8",
        "label": "Approach 8 - DINOv3 ViT-B/16",
        "model_family": "transmil",
        "color_token": "var(--accent-lime)",
        "default_params": {
            "feature_extractor": "dinov3-vitb16",
            "epochs": 30,
            "learning_rate": 0.00004,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach9",
        "label": "Approach 9 - CHIEF",
        "model_family": "transmil",
        "color_token": "var(--accent-cyan)",
        "default_params": {
            "feature_extractor": "chief",
            "epochs": 30,
            "learning_rate": 0.00003,
            "weight_decay": 0.00008,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 10,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "hybrid",
            "strict_feature_extractor": True,
        },
    },
    {
        "key": "approach10",
        "label": "Approach 10 - RetCCL",
        "model_family": "transmil",
        "color_token": "var(--accent-coral)",
        "default_params": {
            "feature_extractor": "retccl",
            "epochs": 30,
            "learning_rate": 0.00005,
            "weight_decay": 0.0001,
            "bag_size": 160,
            "max_val_bag_size": 160,
            "mil_batch_size": 12,
            "weighted_loss": True,
            "fit_one_cycle": True,
            "seed": 310,
            "launch_enabled": True,
            "extractor_backend": "slideflow",
            "strict_feature_extractor": True,
        },
    },
]
VM_MAX_PARALLEL_APPROACHES = int(os.getenv("VM_MAX_PARALLEL_APPROACHES", "2"))
