#!/bin/bash
# scripts/setup_rpi4.sh
# The big red button script to get the Pi ready.
# Sets up the virtual env, forces the UART open, and creates a background service.
# Just run this and go get a coffee.
set -e

echo "============================================="
echo "  INS Drone Setup — Raspberry Pi 4"
echo "  Pixhawk Cube Orange via UART"
echo "============================================="

# ── 1. System packages ────────────────────────────────────────
sudo apt-get update -y
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    git build-essential \
    python3-numpy python3-yaml

# ── 2. Python virtualenv ──────────────────────────────────────
VENV_DIR="$HOME/ins-venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install pymavlink numpy pyyaml pytest

echo "Python dependencies installed."

# ── 3. Enable UART for Pixhawk (ttyAMA0 = GPIO 14/15) ────────
echo ""
echo "Configuring UART (/dev/ttyAMA0) for Pixhawk..."

# Disable Bluetooth to free up the full UART
if ! grep -q "dtoverlay=disable-bt" /boot/config.txt; then
    echo "dtoverlay=disable-bt"       | sudo tee -a /boot/config.txt
    echo "enable_uart=1"              | sudo tee -a /boot/config.txt
    echo "dtparam=uart0=on"           | sudo tee -a /boot/config.txt
fi

# Remove serial console (would conflict with MAVLink)
sudo sed -i 's/console=serial0,[0-9]* //' /boot/cmdline.txt || true

# Add user to dialout group (serial port access)
sudo usermod -aG dialout "$USER"
echo "User $USER added to dialout group."

# ── 4. Create logs directory ──────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_DIR/logs"
echo "Log directory: $REPO_DIR/logs"

# ── 5. Install systemd service ────────────────────────────────
SERVICE_FILE="/etc/systemd/system/ins-drone.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=INS Drone Navigation System (Pixhawk Cube Orange)
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR/src
ExecStart=$VENV_DIR/bin/python core/main.py \\
    --connection /dev/ttyAMA0 \\
    --baud 921600 \\
    --hz 100
Restart=on-failure
RestartSec=5
StandardOutput=append:$REPO_DIR/logs/service.log
StandardError=append:$REPO_DIR/logs/service_error.log
Environment="PYTHONPATH=$REPO_DIR/src"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ins-drone.service
echo "Systemd service installed: ins-drone.service"
echo "  Start : sudo systemctl start ins-drone"
echo "  Stop  : sudo systemctl stop  ins-drone"
echo "  Logs  : journalctl -u ins-drone -f"

# ── 6. Run benchmark (no hardware) ───────────────────────────
echo ""
echo "Running the drag race bench to see if it works..."
cd "$REPO_DIR/src"
PYTHONPATH="$REPO_DIR/src" python "$REPO_DIR/tests/bench.py"

echo ""
echo "============================================="
echo " Setup complete!"
echo " REBOOT required for UART changes to take effect."
echo ""
echo " After reboot, connect Pixhawk TELEM2:"
echo "   TX  → RPi GPIO 15  (pin 10)"
echo "   RX  → RPi GPIO 14  (pin 8)"
echo "   GND → RPi GND      (pin 6)"
echo "   Baud: 921600"
echo ""
echo " In Mission Planner set:"
echo "   SERIAL2_BAUD     = 921"
echo "   SERIAL2_PROTOCOL = 2   (MAVLink2)"
echo ""
echo " Run manually:"
echo "   source $VENV_DIR/bin/activate"
echo "   cd $REPO_DIR/src"
echo "   python core/main.py"
echo "============================================="
