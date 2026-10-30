#!/usr/bin/env pwsh
# One-time setup: virtualenv, dependencies, model files.
[CmdletBinding()]
param(
    [switch]$Clean,   # wipe .venv first (fresh start)
    [switch]$Local    # install the offline models (faster-whisper + Kokoro)
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
$req = if ($Local) { "requirements-local.txt" } else { "requirements.txt" }
& $py -m pip install -r (Join-Path $root $req)
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

Write-Host "  downloading the VAD model (~2 MB)" -ForegroundColor Cyan
$dl = @("-m", "server.download_models")
if ($Local) { $dl += "--kokoro" }
& $py @dl
if ($LASTEXITCODE -ne 0) { throw "model download failed" }

Write-Host "`n  Setup complete." -ForegroundColor Green
Write-Host "  Start the app with:  .\run.ps1" -ForegroundColor Green
if ($Local) {
    Write-Host "  Local mode: set V2V_STT_BACKEND=local, V2V_LLM_BACKEND=ollama, V2V_TTS_ENGINE=kokoro"
    Write-Host "  (the first run also downloads Whisper weights, ~470 MB)`n" -ForegroundColor DarkGray
} else {
    Write-Host "  Add your free Groq key to .env:  GROQ_API_KEY=...`n" -ForegroundColor DarkGray
}
