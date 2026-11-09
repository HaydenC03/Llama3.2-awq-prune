# Render the figures + report from the PREVIEW dataset, without Docker.
# Needs only python + matplotlib + pyyaml (no torch, no GPU).
#   powershell -File scripts/preview.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

python -m pip install --quiet matplotlib numpy pyyaml
python -m src.synthetic_metrics
python -m src.make_plots --metrics results/metrics.preview.json
python -c "from src.make_report import write_report; write_report('results/metrics.preview.json','results/REPORT.preview.md')"
Write-Host "`nopen: results/plots/  and  results/REPORT.preview.md" -ForegroundColor Green
