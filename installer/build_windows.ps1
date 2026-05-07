# Build WhisperApp installer for Windows
# Prerequisites:
#   pip install pyinstaller
#   NSIS installed at C:\Program Files (x86)\NSIS\makensis.exe
# Usage: powershell -ExecutionPolicy Bypass -File installer\build_windows.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "==> Building PyInstaller bundle..."
pyinstaller whisperapp.spec --noconfirm

Write-Host "==> Building NSIS installer..."
$makensis = "C:\Program Files (x86)\NSIS\makensis.exe"
if (-not (Test-Path $makensis)) {
    $makensis = "C:\Program Files\NSIS\makensis.exe"
}
if (-not (Test-Path $makensis)) {
    Write-Error "NSIS not found. Install from https://nsis.sourceforge.io"
}

& $makensis installer\windows.nsi

Write-Host "==> Done: WhisperApp-1.1.0-Setup.exe"
Pop-Location
