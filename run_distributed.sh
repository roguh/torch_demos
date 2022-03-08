#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"
IP_ADDR="${IP_ADDR-$("$SCRIPT_DIR/getip.sh")}"
REMOTE_MACHINE=192.168.50.159
REMOTE_USER=hugo
SSH="$REMOTE_USER@$REMOTE_MACHINE"

rsync -arhP "$SCRIPT_DIR/" "$SSH:src/torch_tests"
ssh "$SSH" conda activate torch \; PYTHONUNBUFFERED=true python src/torch_tests/distributed.py --rank 1 --world 2 --ip "$IP_ADDR"
