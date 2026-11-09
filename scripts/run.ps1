# Run the pipeline in the container, forwarding all arguments.
#   powershell -File scripts/run.ps1 --stage all
#   powershell -File scripts/run.ps1 --stage eval --limit 100
#   powershell -File scripts/run.ps1 --stage plots
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Rest
)
Set-Location (Split-Path $PSScriptRoot -Parent)
docker compose run --rm pipeline @Rest
exit $LASTEXITCODE
