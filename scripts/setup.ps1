# One-time setup: verify Docker + GPU, build the image.
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== docker ==" -ForegroundColor Cyan
docker version --format '{{.Server.Version}}'
docker compose version

Write-Host "`n== GPU visible inside containers ==" -ForegroundColor Cyan
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv
if ($LASTEXITCODE -ne 0) {
    throw "GPU not reachable from Docker. Enable WSL2 backend + GPU support in Docker Desktop."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "created .env (fill in HF_TOKEN if you use the gated meta-llama repo)" -ForegroundColor Yellow
}

Write-Host "`n== building image (first build: 15-25 min, ~7 GB) ==" -ForegroundColor Cyan
docker compose build

Write-Host "`n== smoke test ==" -ForegroundColor Cyan
docker compose run --rm pipeline --stage prepare
docker compose run --rm pipeline --stage all --quick

Write-Host "`nDone. Full run:  powershell -File scripts/run.ps1 --stage all" -ForegroundColor Green
