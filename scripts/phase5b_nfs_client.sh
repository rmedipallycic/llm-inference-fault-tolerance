#!/usr/bin/env bash
# Phase 5 Option B, on the FIRST machine: mount the standby's /srv/kv at the same
# path, with 4 TCP connections (one connection is limited to 5 Gbit/s, Phase 4).
#   bash scripts/phase5b_nfs_client.sh <standby private IP>
set -euo pipefail
IP="${1:?usage: phase5b_nfs_client.sh <standby private IP>}"
sudo apt-get install -y -q nfs-common >/dev/null
sudo mkdir -p /srv/kv
mountpoint -q /srv/kv || sudo mount -t nfs4 -o nconnect=4,rsize=1048576,wsize=1048576 "$IP":/srv/kv /srv/kv
touch /srv/kv/.probe && rm /srv/kv/.probe && echo "write test OK"
mount | grep /srv/kv
