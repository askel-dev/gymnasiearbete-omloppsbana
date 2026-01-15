# OrbitLab - Windows Installation Script
# Run this in PowerShell to automatically install OrbitLab

Write-Host "╔═══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                       ║" -ForegroundColor Cyan
Write-Host "║           O r b i t L a b             ║" -ForegroundColor Cyan
Write-Host "║                                       ║" -ForegroundColor Cyan
Write-Host "║   Windows Installation Script         ║" -ForegroundColor Cyan
Write-Host "║                                       ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
    
    # Verify version is 3.10+
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($matches) {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "✗ Python 3.10+ required. Found $major.$minor" -ForegroundColor Red
            Write-Host "Please download Python 3.10+ from https://python.org/downloads/" -ForegroundColor Yellow
            pause
            exit 1
        }
    }
} catch {
    Write-Host "✗ Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host ""

# Check if requirements.txt exists
Write-Host "[2/4] Checking requirements file..." -ForegroundColor Yellow

if (-Not (Test-Path "requirements.txt")) {
    Write-Host "✗ requirements.txt not found!" -ForegroundColor Red
    Write-Host "Make sure you're running this script from the OrbitLab directory." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✓ requirements.txt found" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "[3/4] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ pip upgraded" -ForegroundColor Green
} else {
    Write-Host "⚠ pip upgrade failed (continuing anyway)" -ForegroundColor Yellow
}

Write-Host ""

# Install dependencies
Write-Host "[4/4] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Installation failed!" -ForegroundColor Red
    Write-Host "Try running manually: pip install -r requirements.txt" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host ""
Write-Host "════════════════════════════════════════" -ForegroundColor Green
Write-Host "    Installation Complete! ✓" -ForegroundColor Green
Write-Host "════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "To run OrbitLab:" -ForegroundColor Cyan
Write-Host "  python src\main.py" -ForegroundColor White
Write-Host ""
Write-Host "Controls:" -ForegroundColor Cyan
Write-Host "  P        - Cycle through planets" -ForegroundColor White
Write-Host "  1-5      - Select scenario" -ForegroundColor White
Write-Host "  SPACE    - Pause/Resume" -ForegroundColor White
Write-Host "  ESC      - Exit" -ForegroundColor White
Write-Host ""

$response = Read-Host "Launch OrbitLab now? (Y/N)"

if ($response -eq "Y" -or $response -eq "y") {
    Write-Host "Launching OrbitLab..." -ForegroundColor Green
    python src\main.py
} else {
    Write-Host "Run 'python src\main.py' when ready!" -ForegroundColor Cyan
}

pause
