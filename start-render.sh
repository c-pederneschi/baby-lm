#!/usr/bin/env bash
set -euo pipefail

TS_VER="1.82.5"
ARCH="amd64"
PKG="tailscale_${TS_VER}_${ARCH}.tgz"
URL="https://pkgs.tailscale.com/stable/${PKG}"

mkdir -p /tmp/ts /tmp/tailscale-state
curl -fsSL "$URL" -o /tmp/ts.tgz
tar -xzf /tmp/ts.tgz -C /tmp/ts

TS_DIR="/tmp/ts/tailscale_${TS_VER}_${ARCH}"
TAILSCALED="${TS_DIR}/tailscaled"
TAILSCALE="${TS_DIR}/tailscale"

"$TAILSCALED" \
  --tun=userspace-networking \
  --state=/tmp/tailscale-state/tailscaled.state \
  --socket=/tmp/tailscale.sock \
  --socks5-server=localhost:1055 \
  >/tmp/tailscaled.log 2>&1 &

sleep 5

"$TAILSCALE" --socket=/tmp/tailscale.sock up \
  --authkey="${TAILSCALE_AUTHKEY}" \
  --hostname="${TS_HOSTNAME:-render-babylm}"

"$TAILSCALE" --socket=/tmp/tailscale.sock ip -4
"$TAILSCALE" --socket=/tmp/tailscale.sock status || true

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
