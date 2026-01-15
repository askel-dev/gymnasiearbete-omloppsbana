#!/bin/bash
# OrbitLab - Unix/Linux/macOS Installation Script

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔═══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                       ║${NC}"
echo -e "${CYAN}║           O r b i t L a b             ║${NC}"
echo -e "${CYAN}║                                       ║${NC}"
echo -e "${CYAN}║   Unix/Linux/macOS Install Script     ║${NC}"
echo -e "${CYAN}║                                       ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════╝${NC}"
echo ""

# Check Python
echo -e "${YELLOW}[1/5] Checking Python installation...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found!${NC}"
    echo -e "${YELLOW}Install Python 3.10+ from your package manager:${NC}"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS:         brew install python@3.11"
    echo "  Fedora:        sudo dnf install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"

# Verify version
MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo -e "${RED}✗ Python 3.10+ required. Found $MAJOR.$MINOR${NC}"
    echo -e "${YELLOW}Please upgrade Python to 3.10 or newer${NC}"
    exit 1
fi

echo ""

# Check pip
echo -e "${YELLOW}[2/5] Checking pip...${NC}"

if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}✗ pip3 not found!${NC}"
    echo -e "${YELLOW}Installing pip...${NC}"
    
    if [ "$(uname)" == "Darwin" ]; then
        python3 -m ensurepip --upgrade
    else
        sudo apt install python3-pip
    fi
fi

echo -e "${GREEN}✓ pip3 found${NC}"
echo ""

# Check requirements.txt
echo -e "${YELLOW}[3/5] Checking requirements file...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗ requirements.txt not found!${NC}"
    echo -e "${YELLOW}Make sure you're running this from the OrbitLab directory.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ requirements.txt found${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}[4/5] Upgrading pip...${NC}"
python3 -m pip install --upgrade pip --quiet

echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}[5/5] Installing dependencies...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"

python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully!${NC}"
else
    echo -e "${RED}✗ Installation failed!${NC}"
    echo -e "${YELLOW}Try running manually: pip3 install -r requirements.txt${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}    Installation Complete! ✓${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}To run OrbitLab:${NC}"
echo "  python3 src/main.py"
echo ""
echo -e "${CYAN}Controls:${NC}"
echo "  P        - Cycle through planets"
echo "  1-5      - Select scenario"
echo "  SPACE    - Pause/Resume"
echo "  ESC      - Exit"
echo ""

read -p "Launch OrbitLab now? (Y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Launching OrbitLab...${NC}"
    python3 src/main.py
else
    echo -e "${CYAN}Run 'python3 src/main.py' when ready!${NC}"
fi
