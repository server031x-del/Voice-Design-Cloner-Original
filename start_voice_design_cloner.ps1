$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:VDC_SERVER_NAME = if ($env:VDC_SERVER_NAME) { $env:VDC_SERVER_NAME } else { "0.0.0.0" }
$env:VDC_SERVER_PORT = if ($env:VDC_SERVER_PORT) { $env:VDC_SERVER_PORT } else { "7860" }

& "$root\venv\Scripts\python.exe" "$root\app.py"
