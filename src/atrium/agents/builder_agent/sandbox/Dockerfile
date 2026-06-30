# Immutable, rootless builder image for the (fixed, non-evolving) BuilderAgent.
#
# It ships the Kaniko executor on a bash-capable base so OpenShell's `bash -lc`
# exec can drive userspace builds. There is deliberately NO Docker daemon, NO
# /var/run/docker.sock and NO --privileged anywhere: Kaniko builds entirely in
# user space and pushes only to the internal local-registry.
#
# Built (by this very agent) and pushed to local-registry/builder_agent:<version>.

# Kaniko's published image is scratch-based (executor binary only, no shell),
# so we copy /kaniko onto a minimal base that has bash for OpenShell exec.
FROM gcr.io/kaniko-project/executor:v1.23.2 AS kaniko

FROM debian:stable-slim

# Bring in the kaniko executor + its warmer/credential helpers.
COPY --from=kaniko /kaniko /kaniko

ENV PATH="/kaniko:${PATH}" \
    DOCKER_CONFIG="/kaniko/.docker" \
    SSL_CERT_DIR="/kaniko/ssl/certs"

# Non-root user: the build runs rootless. /workspace is the build context and
# /kaniko is kaniko's writable scratch (see sandbox/policy.yaml).
RUN useradd --create-home --uid 10001 builder \
    && mkdir -p /workspace \
    && chown -R builder:builder /workspace /kaniko

USER builder
WORKDIR /workspace
