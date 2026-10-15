FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
COPY pyproject.toml uv.lock README.md ./
COPY trading_lab ./trading_lab
COPY configs ./configs
COPY deploy/state-templates ./state-templates
RUN uv sync
ENV LAB_MODE=paper
ENV PYTHONUNBUFFERED=1

# Seed volumes on first boot (evolution: /app/state, signals: /app/live-state).
# Template names live in deploy/state-templates/: state-template | jpy-state-template
#   | smc-state-template | orb-state-template | session-range-state-template
#   | mean-reversion-state-template | orb2r-state-template
# The legacy HERMES_STATE_TEMPLATE variable is still honoured.
CMD ["sh", "-c", "mkdir -p /app/state /app/live-state && TPL=${LAB_STATE_TEMPLATE:-${HERMES_STATE_TEMPLATE:-state-template}} && if [ ! -f /app/state/goal.yaml ]; then cp -r /app/state-templates/${TPL}/* /app/state/; fi && if [ ! -f /app/live-state/hf_baseline.yaml ]; then cp -r /app/state-templates/live-state-template/* /app/live-state/; fi && uv run python -m trading_lab.run"]
