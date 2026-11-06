# syntax=docker/dockerfile:1
#
# The agent image for the proving ground. One file, two arms:
#
#   docker build -f docker/proving-ground.Dockerfile --build-arg ARM=skill \
#     --build-arg VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)" \
#     --build-arg REVISION="$(git rev-parse HEAD)" -t weights-lab-proving:skill .
#
# ARM=control is the same image without the lab and the skill. Both arms get the same
# Python, the same CPU torch, and the same transformers and numpy, pinned from uv.lock.
#
# What the agent can see is the point of this file. The image never holds the proving
# package (task builders, planted constructions, the scorer), its documentation or its
# skill recipe, tests, keys or secrets. The source stage below removes the package from
# the lab it installs and replaces the recipe with a short stub, so links to the recipe
# from other recipes still land somewhere. The .dockerignore keeps the real recipe out of
# the build context too.

ARG UV_VERSION=0.10.12
ARG ARM=skill

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# The lab's source with the proving package taken out and its recipe replaced by a stub.
# Only this stage touches the build context; the arms copy from it.
FROM node:22-bookworm-slim AS source
ARG VERSION
ARG REVISION
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY skills ./skills
COPY docker/proving-ground-stub.md /stub.md
RUN test -n "$REVISION" \
    && grep -q "^version = \"$VERSION\"\$" pyproject.toml \
    && rm -rf src/oumi_weights_lab/proving \
    && cp /stub.md skills/oumi-weights-lab/references/proving-ground.md \
    && cmp /stub.md skills/oumi-weights-lab/references/proving-ground.md \
    && test ! -e src/oumi_weights_lab/proving

FROM node:22-bookworm-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN uv python install 3.12 && uv venv --python 3.12 /opt/venv
# The agent command-line tool. Anything that takes its prompt on the command line works.
ARG AGENT_INSTALL="npm install -g @anthropic-ai/claude-code"
RUN sh -c "$AGENT_INSTALL"
# Claude Code refuses to skip permission prompts as root.
RUN useradd --create-home --uid 10001 agent \
    && mkdir /work \
    && chown agent:agent /work

# Exact versions from uv.lock, minus the CUDA wheels (torch comes from the CPU index).
# The extras to install, in one place: the ones bin/oumi-weights-lab installs on Linux.
FROM base AS pins
ARG LAB_EXTRAS="adapters datasets multi-gpu quantized"
COPY --from=source /src/pyproject.toml /src/uv.lock /src/README.md /pins/
RUN cd /pins \
    && uv export --frozen --no-dev --no-hashes --no-emit-project \
        $(for extra in $LAB_EXTRAS; do printf -- '--extra %s ' "$extra"; done) \
        --output-file all.txt \
    && grep -vE '^(nvidia-|triton==)' all.txt > pinned.txt \
    && grep -q '^torch==' pinned.txt

FROM base AS arm-skill
ARG SKILL_DIR=/home/agent/.claude/skills/oumi-weights-lab
COPY --from=pins /pins/pinned.txt /pins/pinned.txt
COPY --from=source /src /opt/lab-source
RUN uv pip install --python /opt/venv/bin/python --no-deps --torch-backend cpu \
        -r /pins/pinned.txt \
    && uv pip install --python /opt/venv/bin/python --no-deps /opt/lab-source \
    && rm -rf /opt/lab-source \
    && ln -s /opt/venv/bin/oumi-weights-lab /usr/local/bin/oumi-weights-lab
COPY --from=source /src/skills/oumi-weights-lab ${SKILL_DIR}

FROM base AS arm-control
COPY --from=pins /pins/pinned.txt /pins/pinned.txt
RUN uv pip install --python /opt/venv/bin/python --torch-backend cpu \
        -c /pins/pinned.txt torch transformers numpy

FROM arm-${ARM} AS final
ARG ARM
ARG VERSION
ARG REVISION
LABEL org.oumi.weights-lab.arm="${ARM}" \
      org.oumi.weights-lab.version="${VERSION}" \
      org.oumi.weights-lab.revision="${REVISION}"
RUN mkdir -p /home/agent/.claude && chown -R agent:agent /home/agent /work
ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1
USER agent
WORKDIR /work
