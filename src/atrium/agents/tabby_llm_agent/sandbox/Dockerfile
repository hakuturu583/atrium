# TabbyLLMAgent sandbox image.
#
# Built by the (fixed, non-evolving) BuilderAgent with a rootless builder
# (Kaniko / Buildah) and pushed to local-registry/tabby_llm_agent:<version>.
#
# Bundles tabbyAPI (exllamav3 backend) + the A2A↔tabby bridge so the whole
# inference unit is one self-contained, version-pinned artifact.
#
#   docker build -t local-registry/tabby_llm_agent:0.1.0 \
#       -f src/atrium/agents/tabby_llm_agent/sandbox/Dockerfile .

ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TABBY_HOST=127.0.0.1 \
    TABBY_PORT=5000 \
    BRIDGE_PORT=8730

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Inference backend: tabbyAPI (exllamav3). Pinned by tag at build time.
ARG TABBY_REF=main
RUN git clone --depth 1 --branch ${TABBY_REF} https://github.com/theroyallab/tabbyAPI.git \
    && pip3 install --upgrade pip

# Container-side python deps (httpx, a2a-sdk[http-server], uvicorn, OTel, ...).
COPY src/atrium/agents/tabby_llm_agent/sandbox/requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt \
    && pip3 install -r /opt/tabbyAPI/requirements.txt || true

# Install the atrium package (for the bridge: protocol + telemetry helpers).
COPY . /opt/atrium
RUN pip3 install /opt/atrium

EXPOSE 8730
WORKDIR /workspace

# Entrypoint: start tabbyAPI (exllamav3) and the A2A bridge in the same container.
# tabby loads its model lazily / on demand (the model may still be quantizing).
CMD ["bash", "-lc", "\
    python3 /opt/tabbyAPI/main.py --host ${TABBY_HOST} --port ${TABBY_PORT} & \
    python3 -m atrium.agents.tabby_llm_agent.bridge.server \
"]
