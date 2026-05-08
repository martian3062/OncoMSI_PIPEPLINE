$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Join-Path $root "data\\download_5_svs_manifest.csv"
$outputDir = Join-Path $root "downloads\\svs_5"

if (-not (Test-Path $manifest)) {
    throw "Manifest not found: $manifest"
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$rows = Import-Csv $manifest
$uris = $rows | ForEach-Object { $_.bucket_uri }

if (Get-Command gsutil -ErrorAction SilentlyContinue) {
    Write-Host "Using gsutil to download 5 SVS files into $outputDir"
    & gsutil -m cp @uris $outputDir
    exit $LASTEXITCODE
}

if (Get-Command gcloud -ErrorAction SilentlyContinue) {
    Write-Host "Using gcloud storage to download 5 SVS files into $outputDir"
    & gcloud storage cp @uris $outputDir
    exit $LASTEXITCODE
}

throw "Neither gsutil nor gcloud is installed. Install Google Cloud SDK or gsutil, authenticate, then rerun this script."
