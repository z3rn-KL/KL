# Starts the web app in public mode on this machine and, with -Tunnel, opens a
# temporary public address through Cloudflare. Stop with Ctrl+C.
#
#   $env:APP_PASSWORD = "choose-a-password"
#   .\scripts\start_public.ps1            # http://localhost:8000 only
#   .\scripts\start_public.ps1 -Tunnel    # also a public https address

param([switch]$Tunnel)

$ErrorActionPreference = "Stop"

if (-not $env:APP_PASSWORD) {
    throw "Set a password first: `$env:APP_PASSWORD = '...'"
}

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Missing .venv. See README, section 1."
}

if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) {
    throw "Frontend is not built. Run: cd frontend; npm ci; npm run build"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "."

if ($Tunnel -and -not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "Missing cloudflared. Run: winget install --id Cloudflare.cloudflared"
}

# One worker only: dispatch sessions are kept in this process's memory.
$server = Start-Process -FilePath $python -PassThru -NoNewWindow `
    -WorkingDirectory (Join-Path $root "python") `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"

try {
    if ($Tunnel) {
        cloudflared tunnel --url http://127.0.0.1:8000
    } else {
        Wait-Process -Id $server.Id
    }
} finally {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
