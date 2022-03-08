#!/bin/bash
set -euo pipefail
IP_ADDR="$(ip route get 8.8.8.8 | awk '{print $7}')"

echo "$IP_ADDR"
