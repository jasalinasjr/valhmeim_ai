#Requires -RunAsAdministrator
# Valheim AI Setup Script for Windows (requires Administrator rights)

Write-Host "`n=== Valheim AI Environment Setup (2026 Edition) ===" -ForegroundColor Cyan
Write-Host "This script requires Administrator privileges.`n" -ForegroundColor Yellow

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
$PythonVersion     = "Python.Python.3.12"              # 3.12 is stable & widely supported in 2026
$ModelUrl          = "https://your-link-here.com/valheim_custom_v3.pt"   # ← CHANGE THIS
$ModelFilename     = "valheim_custom_v3.pt"
$TrainerScript     = "valheim_ai.py"                   # ← Update to match your actual trainer filename
$ProjectDir        = Get-Location
$VenvName          = "valheim_ai"
$VenvPath          = Join-Path $ProjectDir $VenvName

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
function Test-CommandExists {
    param ([string]$command)
    $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
}

function Refresh-EnvironmentPath {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
    Write-Host "Environment Path refreshed." -ForegroundColor DarkGreen
}

# ─── MAIN STEPS ─────────────────────────────────────────────────────────────────
try {
    # 1. Install Python 3.12 + Git (if missing)
    Write-Host "`n[1/7] Installing/Verifying Python 3.12 and Git..." -ForegroundColor Yellow
    if (-not (Test-CommandExists python)) {
        winget install --id $PythonVersion --scope machine --silent --accept-source-agreements --accept-package-agreements
        Refresh-EnvironmentPath
    } else {
        $pyVer = python --version
        Write-Host "Python already installed: $pyVer" -ForegroundColor Green
    }

    if (-not (Test-CommandExists git)) {
        winget install --id Git.Git --scope machine --silent --accept-source-agreements --accept-package-agreements
        Refresh-EnvironmentPath
    }

    # 2. Create & activate virtual environment
    Write-Host "`n[2/7] Setting up virtual environment..." -ForegroundColor Yellow
    if (Test-Path $VenvPath) {
        Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Green
    } else {
        python -m venv $VenvName
        Write-Host "Virtual environment created: $VenvName" -ForegroundColor Green
    }

    # Activate venv in current session
    & "$VenvPath\Scripts\Activate.ps1"
    Write-Host "Activated virtual environment." -ForegroundColor DarkGreen

    # 3. Install GPU-enabled PyTorch (CUDA 12.1 – safe for GTX 960 series in 2026)
    Write-Host "`n[3/7] Installing PyTorch + TorchVision + Torchaudio (CUDA 12.1)..." -ForegroundColor Yellow
    pip install --upgrade pip setuptools wheel
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Quick verification
    $torchCheck = python -c "import torch; print('CUDA available:' + str(torch.cuda.is_available())); print('Device count:' + str(torch.cuda.device_count()))"
    Write-Host $torchCheck -ForegroundColor DarkCyan

    # 4. Install project dependencies
    Write-Host "`n[4/7] Installing remaining dependencies..." -ForegroundColor Yellow
    pip install --upgrade `
        gymnasium `
        ultralytics `
        stable-baselines3 `
        mss `
        pydirectinput `
        pymem `
        pynvml `
        opencv-python `
        numpy

    # Optional: tensorboard if you use SB3 logging
    pip install tensorboard

    # 5. Download custom YOLO model weights
    Write-Host "`n[5/7] Downloading YOLO model weights..." -ForegroundColor Yellow
    $modelPath = Join-Path $ProjectDir $ModelFilename

    if (Test-Path $modelPath) {
        Write-Host "Model file already exists: $ModelFilename" -ForegroundColor Green
    } else {
        Write-Host "Downloading from: $ModelUrl"
        Invoke-WebRequest -Uri $ModelUrl -OutFile $modelPath -UseBasicParsing

        if ((Get-Item $modelPath).Length -gt 1000) {
            Write-Host "Model downloaded successfully." -ForegroundColor Green
        } else {
            throw "Downloaded file is too small or empty. Check URL or network."
        }
    }

    # 6. Create Desktop Shortcut
    Write-Host "`n[6/7] Creating Desktop shortcut..." -ForegroundColor Yellow
    $Desktop = [Environment]::GetFolderPath("Desktop")
    $ShortcutPath = Join-Path $Desktop "Start Valheim AI.lnk"

    $WShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WShell.CreateShortcut($ShortcutPath)

    $PythonExe   = Join-Path $VenvPath "Scripts\python.exe"
    $ScriptPath  = Join-Path $ProjectDir $TrainerScript

    $Shortcut.TargetPath     = "cmd.exe"
    $Shortcut.Arguments      = "/K `"$PythonExe`" `"$ScriptPath`""
    $Shortcut.WorkingDirectory = $ProjectDir
    $Shortcut.Description    = "Launch Valheim AI Trainer (venv activated)"
    $Shortcut.WindowStyle    = 1  # Normal window (change to 7 for minimized)
    # Optional: $Shortcut.IconLocation = "path\to\icon.ico"

    $Shortcut.Save()
    Write-Host "Shortcut created on Desktop: 'Start Valheim AI'" -ForegroundColor Green

    Write-Host "`n=== SETUP COMPLETED SUCCESSFULLY ===" -ForegroundColor Cyan
    Write-Host "You can now double-click the desktop shortcut to start training.`n" -ForegroundColor White

} catch {
    Write-Host "`nERROR during setup: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkRed
    Write-Host "`nSetup failed. Please check the error above and try again." -ForegroundColor Yellow
    exit 1
} finally {
    # Deactivate venv if it was activated
    if (Get-Command deactivate -ErrorAction SilentlyContinue) {
        deactivate
    }
}