# MSI-H

This is a separate clean local frontend focused only on preserved H-Optimus
trained-output presentation.

## What it does

- shows saved H-Optimus run metrics
- shows fold-level output table
- lets you search a known TCGA slide or patient from the preserved cohort
- stays lightweight and local

## What it does not do

- no Slideflow
- no heavy WSI inference
- no fake fresh prediction without the missing saved inference head

## Run locally

```powershell
.\.venv\Scripts\python.exe .\msi-h\serve.py
```

Then open:

`http://127.0.0.1:8011`
