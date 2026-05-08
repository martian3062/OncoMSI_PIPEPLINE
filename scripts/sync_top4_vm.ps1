$ErrorActionPreference = "Stop"

$sshKey = "$env:USERPROFILE\.ssh\evolet_rsa"
$hostName = "34.126.112.227"
$userAtHost = "pardeep@$hostName"
$remoteRoot = "/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc"
$staging = Join-Path $env:TEMP "cleaned_msi_top4_sync"
$payload = Join-Path $staging "top4-sync.tar.gz"

if (Test-Path $staging) {
    Remove-Item -Recurse -Force $staging
}
New-Item -ItemType Directory -Force -Path $staging | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $staging "django_rebuild_cleaned_msi") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $staging "scripts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $staging "tools") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $staging "models\virchow") | Out-Null

Copy-Item -Recurse -Force "E:\Cleaned_MSI\apps" (Join-Path $staging "django_rebuild_cleaned_msi\apps")
Copy-Item -Recurse -Force "E:\Cleaned_MSI\msi_platform" (Join-Path $staging "django_rebuild_cleaned_msi\msi_platform")
Copy-Item -Recurse -Force "E:\Cleaned_MSI\static" (Join-Path $staging "django_rebuild_cleaned_msi\static")
Copy-Item -Recurse -Force "E:\Cleaned_MSI\runtime\annotations" (Join-Path $staging "django_rebuild_cleaned_msi\runtime\annotations")
Copy-Item -Force "E:\Cleaned_MSI\manage.py" (Join-Path $staging "django_rebuild_cleaned_msi\manage.py")
Copy-Item -Force "E:\Cleaned_MSI\requirements.txt" (Join-Path $staging "django_rebuild_cleaned_msi\requirements.txt")
Copy-Item -Force "E:\Cleaned_MSI\.env.example" (Join-Path $staging "django_rebuild_cleaned_msi\.env.example")
if (Test-Path "E:\Cleaned_MSI\.env") {
    Copy-Item -Force "E:\Cleaned_MSI\.env" (Join-Path $staging "django_rebuild_cleaned_msi\.env")
}
Copy-Item -Force "E:\Cleaned_MSI\launch_top4_montecarlo.py" (Join-Path $staging "django_rebuild_cleaned_msi\launch_top4_montecarlo.py")
Copy-Item -Force "E:\Cleaned_MSI\tcga3_vm_annotations.csv" (Join-Path $staging "django_rebuild_cleaned_msi\runtime\annotations\tcga3_vm_annotations.csv")
Copy-Item -Force "E:\Cleaned_MSI\vm_patch\run_tcga_coad_automated_triad.py" (Join-Path $staging "scripts\run_tcga_coad_automated_triad.py")
Copy-Item -Force "E:\Cleaned_MSI\vm_patch\hybrid_extractors.py" (Join-Path $staging "scripts\hybrid_extractors.py")
Copy-Item -Force "E:\Cleaned_MSI\scripts\run_top4_hybrid_ensemble.sh" (Join-Path $staging "scripts\run_top4_hybrid_ensemble.sh")
Copy-Item -Recurse -Force "E:\Cleaned_MSI\tools\*" (Join-Path $staging "tools")
Copy-Item -Force "E:\Cleaned_MSI\new3\vm_weights\virchow\pytorch_model.bin" (Join-Path $staging "models\virchow\pytorch_model.bin")
Copy-Item -Force "E:\Cleaned_MSI\scripts\bootstrap_top4_vm.sh" (Join-Path $staging "bootstrap_top4_vm.sh")

tar -czf $payload -C $staging .
scp -i $sshKey $payload "${userAtHost}:/tmp/top4-sync.tar.gz"
ssh -i $sshKey $userAtHost "mkdir -p '$remoteRoot' && tar -xzf /tmp/top4-sync.tar.gz -C '$remoteRoot' && chmod +x '$remoteRoot/bootstrap_top4_vm.sh' '$remoteRoot/scripts/run_top4_hybrid_ensemble.sh' && '$remoteRoot/bootstrap_top4_vm.sh'"

Write-Host "Top-4 project synced to ${userAtHost}:$remoteRoot"
