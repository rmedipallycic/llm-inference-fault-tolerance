#!/usr/bin/env bash
# Phase 5 Option B, on the STANDBY machine: a RAM disk at /srv/kv, exported over
# NFS to the private network. Needs inbound TCP 2049 from the same security group.
#   bash scripts/phase5b_nfs_server.sh
set -euo pipefail
SIZE="${SIZE:-6G}"
sudo apt-get install -y -q nfs-kernel-server >/dev/null
sudo mkdir -p /srv/kv
mountpoint -q /srv/kv || sudo mount -t tmpfs -o size="$SIZE",mode=1777 tmpfs /srv/kv
echo "/srv/kv 172.31.0.0/16(rw,async,no_subtree_check,no_root_squash,fsid=1)" | sudo tee /etc/exports.d/kv.exports >/dev/null
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
df -h /srv/kv
sudo exportfs -v | grep /srv/kv
