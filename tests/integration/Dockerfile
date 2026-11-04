FROM node:22-bookworm-slim

ARG CODEX_VERSION=0.166.0
ARG CLAUDE_VERSION=2.1.318
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN npm install --global \
    @openai/codex@${CODEX_VERSION} @anthropic-ai/claude-code@${CLAUDE_VERSION}
RUN python3 -m venv /opt/oumi-weights-lab \
    && /opt/oumi-weights-lab/bin/pip install hatchling
ENV PATH="/opt/oumi-weights-lab/bin:${PATH}"
COPY pyproject.toml README.md LICENSE /opt/source/
COPY src /opt/source/src
# Help and plugin discovery do not import the model stack. No torch/model downloads.
RUN pip install --no-deps --no-build-isolation /opt/source \
    && ln -s /opt/oumi-weights-lab/bin/oumi-weights-lab /usr/local/bin/oumi-weights-lab \
    && useradd --create-home --uid 10001 harness \
    && mkdir /work && chown harness:harness /work
USER harness
WORKDIR /work
