# OrbitLab – Installation Guide

Complete installation instructions for all platforms.

---

## 🚀 Quick Install (All Platforms)

**Prerequisites**: Python 3.10 or newer

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/OrbitLab.git
cd OrbitLab

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the simulator
python src/main.py
```

That's it! OrbitLab should launch.

---

## 📦 Detailed Installation by Platform

### Windows

#### Method 1: Standard Installation

1. **Install Python**
   - Download from https://python.org/downloads/
   - ✅ Check "Add Python to PATH" during installation
   - Verify: Open Command Prompt and type `python --version`

2. **Clone or Download OrbitLab**
   ```cmd
   git clone https://github.com/yourusername/OrbitLab.git
   cd OrbitLab
   ```
   
   *OR* download ZIP from GitHub and extract it

3. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run OrbitLab**
   ```cmd
   python src\main.py
   ```

#### Method 2: Using PowerShell Script (Automated)

```powershell
# Run this in PowerShell (as Administrator)
.\install_windows.ps1
```

#### Method 3: Standalone Executable (No Python Required)

*Coming soon: Pre-built .exe for Windows*

Download `OrbitLab_v1.5_win64.zip` from Releases, extract, and run `OrbitLab.exe`.

---

### macOS

#### Prerequisites

Install Python 3.10+ via Homebrew:

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
```

#### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/OrbitLab.git
cd OrbitLab

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python src/main.py
```

#### macOS-Specific Notes

- If you get SDL/Pygame display issues, install SDL2:
  ```bash
  brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
  ```

- For Apple Silicon (M1/M2), use:
  ```bash
  arch -arm64 pip install -r requirements.txt
  ```

---

### Linux

#### Ubuntu/Debian

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Clone repository
git clone https://github.com/yourusername/OrbitLab.git
cd OrbitLab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python src/main.py
```

#### Fedora/RHEL

```bash
# Install Python
sudo dnf install python3 python3-pip

# Clone and install
git clone https://github.com/yourusername/OrbitLab.git
cd OrbitLab
pip install -r requirements.txt
python src/main.py
```

#### Arch Linux

```bash
# Install Python
sudo pacman -S python python-pip

# Clone and install
git clone https://github.com/yourusername/OrbitLab.git
cd OrbitLab
pip install -r requirements.txt
python src/main.py
```

---

## 🐳 Docker Installation (Advanced)

Run OrbitLab in an isolated container:

```bash
# Build image
docker build -t orbitlab .

# Run with GUI support
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  orbitlab
```

*Note: Dockerfile coming soon*

---

## 🔍 Troubleshooting

### Problem: `pygame` installation fails

**Windows:**
```cmd
pip install pygame --user
```

**Linux:** Install SDL development libraries first:
```bash
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
pip install pygame
```

**macOS:**
```bash
brew install sdl2
pip install pygame
```

### Problem: `ModuleNotFoundError: No module named 'pygame'`

You're not in the correct environment. Make sure you:
1. Activated your virtual environment (if using one)
2. Installed dependencies with `pip install -r requirements.txt`
3. Are running Python 3.10+

Verify:
```bash
python --version  # Should be 3.10+
pip list | grep pygame  # Should show pygame 2.5.0+
```

### Problem: Window doesn't open / Black screen

**Check 1:** Verify display is available
```bash
# Linux
echo $DISPLAY  # Should output something like :0

# Windows/macOS: Should work by default
```

**Check 2:** Update graphics drivers
- Windows: Update via Device Manager
- macOS: Update via System Preferences
- Linux: `sudo apt install mesa-utils` or your GPU-specific drivers

**Check 3:** Try software rendering
```bash
export SDL_VIDEODRIVER=x11  # Linux
python src/main.py
```

### Problem: ImportError related to NumPy

```bash
# Upgrade NumPy
pip install --upgrade numpy

# If that fails, reinstall
pip uninstall numpy
pip install numpy
```

### Problem: Permission denied

**Windows:** Run Command Prompt as Administrator

**macOS/Linux:** Use `sudo` or install in user directory:
```bash
pip install --user -r requirements.txt
```

### Problem: Slow performance

1. **Close other applications** to free RAM
2. **Reduce timestep:** In simulator, decrease simulation speed
3. **Lower zoom level:** Render fewer pixels
4. **Disable anti-aliasing** (code modification needed)

---

## 🧪 Verify Installation

Run this quick test:

```bash
python -c "import pygame; import numpy; import matplotlib; print('✓ All dependencies installed!')"
```

If you see "✓ All dependencies installed!", you're good to go!

---

## 📚 Next Steps

After installation:

1. **Launch OrbitLab:**
   ```bash
   python src/main.py
   ```

2. **Try keyboard controls:**
   - `P` – Cycle through planets
   - `1-5` – Select scenarios
   - `SPACE` – Pause/resume
   - `ESC` – Exit

3. **Explore documentation:**
   - `README.md` – Overview and features
   - `LOGGING_OVERVIEW.md` – Data format details
   - `src/analyze_run.py` – Analysis tools

---

## 🆘 Still Having Issues?

If you encounter problems:

1. **Check Python version:** `python --version` (must be 3.10+)
2. **Update pip:** `pip install --upgrade pip`
3. **Clear pip cache:** `pip cache purge`
4. **Try fresh install:**
   ```bash
   pip uninstall pygame numpy matplotlib
   pip install -r requirements.txt
   ```

5. **Report bug:** Open an issue on GitHub with:
   - Your OS and version
   - Python version (`python --version`)
   - Error message (full traceback)
   - Output of `pip list`

---

## 🚀 Alternative: No Installation Required

**Try OrbitLab Online** (coming soon):
- Browser-based version using WebAssembly
- No Python installation needed
- Limited features but instant access

---

## 📦 For Developers

If you want to modify OrbitLab:

```bash
# Clone with development dependencies
git clone https://github.com/yourusername/OrbitLab.git
cd OrbitLab

# Install in editable mode with dev tools
pip install -e .
pip install pytest black pylint mypy

# Run tests
pytest tests/

# Format code
black src/

# Lint
pylint src/
```

---

**Installation Time:** ~2-5 minutes  
**Disk Space:** ~50 MB (dependencies + code)  
**Internet Required:** Only for initial download

*Happy orbiting! 🛰️*
