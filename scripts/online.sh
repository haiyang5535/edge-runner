#!/bin/bash
# Read saved gateway or default to known IP
GW=$(cat /tmp/.last_gateway 2>/dev/null || echo "10.0.0.1")

# Restore default route
sudo ip route add default via $GW

echo -e "\033[32mInternet RESTORED\033[0m"
