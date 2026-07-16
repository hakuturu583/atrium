#!/usr/bin/env bash
#
# Provision the gateway-minted **sandbox JWT** signing bundle a local OpenShell
# gateway needs before it will create Docker sandboxes.
#
# Why this is needed: OpenShell's Docker compute driver refuses to create
# sandboxes unless the gateway can mint per-sandbox JWTs for the sandbox->gateway
# callback ("docker sandboxes require gateway JWT auth"). The gateway auto-loads
# that bundle from `$OPENSHELL_LOCAL_TLS_DIR/jwt/{signing.pem,public.pem,kid}`.
# `generate-certs` writes a full TLS+JWT bundle; we keep only the JWT half so the
# gateway stays plaintext-HTTP (disable_tls = true in gateway.toml) while still
# minting sandbox JWTs.
#
# Idempotent: re-running regenerates the bundle. Run it once before the first
# `docker compose up`, then register the gateway with the CLI:
#
#   bash deploy/openshell/init-jwt.sh
#   docker compose -f deploy/openshell/docker-compose.yml up -d
#   openshell gateway add http://localhost:8080 --name atrium-local
#   openshell gateway select atrium-local
#
set -euo pipefail

GATEWAY_IMAGE="${GATEWAY_IMAGE:-ghcr.io/nvidia/openshell/gateway:latest}"
# Must match OPENSHELL_LOCAL_TLS_DIR in docker-compose.yml (both sides bind-mount
# /var/lib/openshell at the same absolute path).
DATA_DIR="${OPENSHELL_DATA_DIR:-/var/lib/openshell}"
TLS_DIR="${DATA_DIR}/tls"

echo "generating sandbox JWT bundle into ${TLS_DIR} (via ${GATEWAY_IMAGE})..."
docker run --rm --user 0 \
  -e "OPENSHELL_LOCAL_TLS_DIR=${TLS_DIR}" \
  -v "${DATA_DIR}:${DATA_DIR}" \
  "${GATEWAY_IMAGE}" \
  generate-certs --output-dir "${TLS_DIR}"

# Keep only jwt/*; drop the TLS PEMs so the gateway does not auto-enable HTTPS
# (complete_local_tls_paths() treats 0 TLS files as "no local TLS", any partial
# set as an error). A distroless gateway image has no shell, so do the pruning
# from a small image that shares the same bind-mount.
echo "pruning TLS PEMs (keeping only the JWT bundle)..."
docker run --rm --user 0 -v "${DATA_DIR}:${DATA_DIR}" busybox sh -c "
  rm -f '${TLS_DIR}'/ca.crt '${TLS_DIR}'/ca.key \
        '${TLS_DIR}'/server/tls.crt '${TLS_DIR}'/server/tls.key \
        '${TLS_DIR}'/client/tls.crt '${TLS_DIR}'/client/tls.key
  rmdir '${TLS_DIR}'/server '${TLS_DIR}'/client 2>/dev/null || true
  echo 'remaining:'; find '${TLS_DIR}' -type f | sort
"
echo "done. Now: docker compose -f deploy/openshell/docker-compose.yml up -d"
