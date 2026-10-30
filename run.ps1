#!/usr/bin/env pwsh
# Start the voice chat server.
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$Model = "",
    [string]$Voice = "",
    [switch]$Check,      # run the offline component self-test instead of the server
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "  No virtualenv found. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$env:V2V_PORT = "$Port"
if ($Model) { $env:V2V_LLM_MODEL = $Model }
if ($Voice) { $env:V2V_TTS_VOICE = $Voice }

if ($Check) {
    # The self-test deliberately measures the offline stack (it round-trips
    # Kokoro audio through the VAD and Whisper), so force it even if .env says
    # otherwise. A real environment variable wins over .env in config.py.
    $env:V2V_STT_BACKEND = "local"
    $env:V2V_LLM_BACKEND = "ollama"
    $env:V2V_TTS_ENGINE = "kokoro"
}

# Ollama is only needed for the fully-offline stack. Start it here (the desktop
# app does the same thing) so the local mode does not fail with a confusing
# "cannot reach Ollama" several seconds into the first conversation.
function Test-Ollama {
    try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null; return $true }
    catch { return $false }
}

if ($env:V2V_LLM_BACKEND -and $env:V2V_LLM_BACKEND -ne "ollama") {
    Write-Host "  Chat backend: $($env:V2V_LLM_BACKEND) (no local model needed)" -ForegroundColor DarkGray
} elseif (-not (Test-Ollama)) {
    Write-Host "  Ollama is not running - starting it..." -ForegroundColor Yellow
    $ollama = (Get-Command ollama.exe -ErrorAction SilentlyContinue).Source
    if (-not $ollama) { $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
    if (Test-Path $ollama) {
        Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 40 -and -not (Test-Ollama); $i++) { Start-Sleep -Milliseconds 500 }
    }
    if (Test-Ollama) {
        Write-Host "  Ollama is up." -ForegroundColor Green
    } else {
        Write-Host "  Could not reach Ollama - chat will not work until it is started:" -ForegroundColor Red
        Write-Host "    ollama serve" -ForegroundColor DarkGray
    }
}

if (-not (Test-Path (Join-Path $root ".env")) -and -not $env:GROQ_API_KEY) {
    Write-Host "  Note: no .env and no GROQ_API_KEY - speech and chat will report an error." -ForegroundColor Yellow
    Write-Host "        Create one free key at https://console.groq.com/keys" -ForegroundColor DarkGray
}

if ($Check) {
    & $py -m server.selftest
    exit $LASTEXITCODE
}

Write-Host "`n  Halo voice chat" -ForegroundColor Cyan
Write-Host "  http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "  (Ctrl+C to stop)`n" -ForegroundColor DarkGray

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($p)
        Start-Sleep -Seconds 6
        Start-Process "http://127.0.0.1:$p"
    } -ArgumentList $Port | Out-Null
}

& $py -m server.main
