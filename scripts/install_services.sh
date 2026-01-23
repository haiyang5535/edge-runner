#!/bin/bash
# EdgeRunner Service Installer
# Installs systemd services for llama-server and edge-runner

set -e

SERVICES_DIR="$(dirname "$0")/../services"

echo "=== EdgeRunner Service Installer ==="
echo ""

# Copy service files
echo "Installing services..."
if [ -f "$SERVICES_DIR/llama-server.service" ]; then
    sudo cp "$SERVICES_DIR/llama-server.service" /etc/systemd/system/
    echo "   ✅ llama-server.service"
fi

if [ -f "$SERVICES_DIR/edge-runner.service" ]; then
    sudo cp "$SERVICES_DIR/edge-runner.service" /etc/systemd/system/
    echo "   ✅ edge-runner.service"
fi

# Reload and enable
sudo systemctl daemon-reload
echo "   ✅ daemon-reload"

sudo systemctl enable llama-server.service
sudo systemctl enable edge-runner.service
echo "   ✅ Services enabled"

echo ""
echo "=== Quick Reference ==="
echo "   Start:   sudo systemctl start llama-server edge-runner"
echo "   Stop:    sudo systemctl stop edge-runner llama-server"
echo "   Logs:    sudo journalctl -u edge-runner -f"
echo "   Status:  sudo systemctl status edge-runner llama-server"
echo "   Restart: sudo systemctl restart edge-runner"
