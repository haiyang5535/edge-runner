#!/bin/bash
# Disconnect from internet for offline testing
# Bug fix: Delete ALL default routes, not just one

# Save current gateway if not already saved
if [ ! -f /tmp/.last_gateway ]; then
    ip route show default | awk '{print $3}' | head -1 > /tmp/.last_gateway
fi

# Delete ALL default routes (there may be multiple)
while ip route show default 2>/dev/null | grep -q default; do
    sudo ip route del default 2>/dev/null
done

# Verify disconnection
if ping -c 1 -W 1 8.8.8.8 >/dev/null 2>&1; then
    echo -e "\033[31mError: Failed to disconnect. Still online.\033[0m"
    echo "Try running: sudo ip route del default"
    exit 1
else
    echo -e "\033[31mInternet DISCONNECTED\033[0m"
fi
