# One-command setup: Python environment (.venv) + libraries + model code and weights.
# Easiest: double-click setup.bat in the repo folder. Or:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1          # GPU/CPU detected automatically
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -Cpu     # force the CPU build
# Safe to run again: finished steps are skipped.
param([switch]$Cpu, [switch]$Gpu)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Step($t) { Write-Host ""; Write-Host "== $t" -ForegroundColor Cyan }

# Windows' 260-character path limit breaks some packages (hydra) in deep folders
if ($root.Length -gt 80) {
    Write-Warning "This folder path is $($root.Length) characters long. If the install fails with 'filename too long', move the project to a shorter folder (for example C:\work\roto) and run setup.bat again."
}

Step "1/5 Finding Python 3.11"
function Find-Python311 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $p = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    }
    foreach ($c in @("python", "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")) {
        if (Get-Command $c -ErrorAction SilentlyContinue) {
            $v = & $c -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($v -eq "3.11") { return (& $c -c "import sys; print(sys.executable)").Trim() }
        }
    }
    return $null
}
$python = Find-Python311
if (-not $python) {
    Write-Host "Python 3.11 not found: installing it with winget (a few minutes) ..."
    winget install --id Python.Python.3.11 -e --silent --accept-source-agreements --accept-package-agreements
    $python = Find-Python311
    if (-not $python) { throw "Could not install Python 3.11. Install it from python.org (tick 'Add python.exe to PATH') and run setup.bat again." }
}
Write-Host "Using $python"

Step "2/5 Creating the project environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $python -m venv .venv }
$vpy = Join-Path $root ".venv\Scripts\python.exe"
& $vpy -m pip install -q --upgrade pip

Step "3/5 Installing PyTorch"
if (-not $Cpu -and -not $Gpu) { $Gpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue) }
if ($Gpu) {
    Write-Host "NVIDIA GPU found: installing the CUDA 13.0 build (about 3 GB, 5-15 minutes) ..."
    & $vpy -m pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
} else {
    Write-Host "No NVIDIA GPU: installing the CPU build (the AI stages will be slow; align/fuse/exr are fine) ..."
    & $vpy -m pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed (internet? disk space?). Run setup.bat again: it continues where it stopped." }

Step "4/5 Installing the other libraries (requirements.txt)"
& $vpy -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Library install failed. Run setup.bat again." }

Step "5/5 Getting the AI models (code + ~520 MB of weights)"
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "setup_models.ps1")

Step "Checking"
& $vpy -c "import torch, cv2, OpenEXR, yaml; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU only)')"
if ($LASTEXITCODE -ne 0) {
    throw "Import failed. If you see 'WinError 4551': Windows Smart App Control blocks PyTorch. Turn it off (Windows Security > App & browser control > Smart App Control > Off), restart, run setup.bat again."
}
Write-Host ""
Write-Host "SETUP DONE. Next: double-click run_demo.bat" -ForegroundColor Green
