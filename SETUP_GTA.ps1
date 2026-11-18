$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$python = Join-Path $appRoot '.venv\Scripts\python.exe'
Get-Command uv -ErrorAction Stop | Out-Null
$env:UV_CACHE_DIR = Join-Path $appRoot '.vice-local\uv-cache'
if (!(Test-Path -LiteralPath $python)) {
    uv venv (Join-Path $appRoot '.venv') --python 3.13
    if ($LASTEXITCODE -ne 0) { throw 'Creazione ambiente Python non riuscita.' }
}
uv pip install --python $python -r (Join-Path $appRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installazione librerie non riuscita.' }
Write-Output 'Setup completato. Avvia AVVIA_GTA.bat. Per VICE Locale: SETUP_VICE_LOCALE.ps1.'
