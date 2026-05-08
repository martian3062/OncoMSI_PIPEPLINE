$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Join-Path $root "data\\download_5_svs_gdc_manifest.csv"
$outputDir = Join-Path $root "downloads\\gdc_5_exact"

if (-not (Test-Path $manifest)) {
    throw "Manifest not found: $manifest"
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$rows = Import-Csv $manifest

foreach ($row in $rows) {
    $target = Join-Path $outputDir $row.slide
    if (Test-Path $target) {
        Write-Host "Skipping existing file $($row.slide)"
        continue
    }

    Write-Host "Downloading $($row.slide) [$($row.msi_status)]"
    Invoke-WebRequest -Uri $row.gdc_url -OutFile $target
}

Write-Host "Done. Files saved to $outputDir"
