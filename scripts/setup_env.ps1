# One-command setup for a teammate: Python environment + libraries + model code and weights.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1          # NVIDIA GPU (CUDA 13.0 build)
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -Cpu     # no NVIDIA GPU: CPU stages only
#
# Creates .venv/ in the repo, installs pinned torch + requirements.txt, then runs
# setup_models.ps1 (model repos at pinned commits + ~520 MB of weights), and
# finally checks that everything imports. Safe to run again.
param([switch]$Cpu)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Windows' 260-character path limit breaks some packages (hydra) in deep folders
if ($root.Length -gt 80) {
    Write-Warning "Repo path is $($root.Length) characters long. If pip fails with 'filename too long' (WinError 206), clone into a short folder such as C:\work\ or enable Windows long paths."
}

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "Python not found. Install Python 3.11 from python.org and tick 'Add to PATH'." }
$ver = & python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ($ver -ne "3.11") { Write-Warning "Python $ver found; the project is tested on 3.11." }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
}
$vpy = Join-Path $root ".venv\Scripts\python.exe"
& $vpy -m pip install -q --upgrade pip

if ($Cpu) {
    Write-Host "Installing PyTorch (CPU build) ..."
    & $vpy -m pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
} else {
    Write-Host "Installing PyTorch (CUDA 13.0 build, about 3 GB) ..."
    & $vpy -m pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
}
Write-Host "Installing requirements.txt ..."
& $vpy -m pip install -q -r requirements.txt
& $vpy -m pip check

Write-Host "Getting model code and weights ..."
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "setup_models.ps1")

Write-Host "Checking ..."
& $vpy -c "import torch, cv2, OpenEXR, yaml; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU only)')"
if ($LASTEXITCODE -ne 0) {
    throw "Import failed. 'WinError 4551' = Windows Smart App Control is blocking PyTorch: turn it off (Windows Security > App & browser control)."
}
Write-Host ""
Write-Host "Done. Activate the environment with:  .venv\Scripts\activate"
Write-Host "Then get data (samples\README.md) and run, e.g.:  python run_pipeline.py --config configs/tos_01_2a_from_mp4.yaml"
