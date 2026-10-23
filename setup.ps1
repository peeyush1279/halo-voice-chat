#!/usr/bin/env pwsh
# One-time setup: virtualenv, dependencies, model files.
[CmdletBinding()]
param(
    [switch]$Clean   # wipe .venv first (fresh start)
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

Write-Host "`n  voice-chat setup" -ForegroundColor Cyan
Write-Host "  ----------------------------" -ForegroundColor DarkGray

$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if ($Clean -and (Test-Path $venv)) {
    Write-Host "  removing existing virtualenv" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $py)) {
    Write-Host "  creating virtualenv (.venv)" -ForegroundColor Cyan
    python -m venv $venv
}

Write-Host "  installing dependencies" -ForegroundColor Cyan
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

Write-Host "  downloading model files (Kokoro ~330 MB on first run)" -ForegroundColor Cyan
& $py -m server.download_models
if ($LASTEXITCODE -ne 0) { throw "model download failed" }

Write-Host "`n  Setup complete." -ForegroundColor Green
Write-Host "  Start the app with:  .\run.ps1" -ForegroundColor Green
Write-Host "  (the first run also downloads Whisper weights, ~150 MB)`n" -ForegroundColor DarkGray
