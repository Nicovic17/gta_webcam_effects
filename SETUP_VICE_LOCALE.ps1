$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$localRoot = Join-Path $appRoot '.vice-local'
$upstream = Join-Path $localRoot 'StreamDiffusion'
$revision = 'b623251dc055e1fd858d53509aa43e09dfc5cdc0'
New-Item -ItemType Directory -Path $localRoot -Force | Out-Null
if (!(Test-Path -LiteralPath (Join-Path $localRoot '.gitignore'))) {
    Set-Content -LiteralPath (Join-Path $localRoot '.gitignore') -Value '*' -Encoding ASCII
}
Get-Command uv -ErrorAction Stop | Out-Null
Get-Command git -ErrorAction Stop | Out-Null
$env:UV_CACHE_DIR = Join-Path $localRoot 'uv-cache'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
uv python install 3.10.22 --install-dir (Join-Path $localRoot 'python')
if ($LASTEXITCODE -ne 0) { throw 'Download Python non riuscito.' }
$pythonBase = Join-Path $localRoot 'python\cpython-3.10.22-windows-x86_64-none\python.exe'
$python = Join-Path $localRoot 'venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) {
    uv venv (Join-Path $localRoot 'venv') --python $pythonBase
    if ($LASTEXITCODE -ne 0) { throw 'Creazione virtualenv non riuscita.' }
}
if (!(Test-Path -LiteralPath $upstream)) {
    git clone --no-checkout https://github.com/cumulo-autumn/StreamDiffusion.git $upstream
    if ($LASTEXITCODE -ne 0) { throw 'Download StreamDiffusion non riuscito.' }
    git -C $upstream checkout --detach $revision
    if ($LASTEXITCODE -ne 0) { throw 'Revisione StreamDiffusion non disponibile.' }
}
$actual = git -C $upstream rev-parse HEAD
if ($actual -ne $revision) { throw 'Revisione StreamDiffusion diversa da quella verificata.' }
uv pip install --python $python --index https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match -r (Join-Path $appRoot 'requirements-vice-local.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installazione librerie locali non riuscita.' }
& $python -B (Join-Path $appRoot 'benchmark_vice_local.py') --runtime-root $localRoot --input (Join-Path $appRoot 'assets\benchmark_input.jpg') --output (Join-Path $localRoot 'benchmark') --download
if ($LASTEXITCODE -ne 0) { throw 'Download modello o benchmark non riuscito.' }
Write-Output 'Setup completato. Avvia gta.py e scegli 4 - VICE Locale.'
