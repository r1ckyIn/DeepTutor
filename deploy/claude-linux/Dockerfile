# DeepTutor for the owner's Linux host: the published upstream image plus this
# fork's backend packages and the Claude Agent SDK. The frontend stays upstream's.
# Plain legacy-builder syntax (the host has no buildx). Build from the repo root:
#   docker build -f deploy/claude-linux/Dockerfile -t deeptutor-claude:local .
# Move the tag along with the fork's upstream syncs so requirements keep matching.
FROM ghcr.io/hkuds/deeptutor:1.6.15

# The wheel bundles the Claude Code CLI under claude_agent_sdk/_bundled and puts
# nothing on PATH, which is intended. Fail here if a wheel without it was picked.
RUN pip install --no-cache-dir "claude-agent-sdk==0.2.174" \
    && python -c "import os, claude_agent_sdk as m; p = os.path.join(os.path.dirname(m.__file__), '_bundled', 'claude'); assert os.access(p, os.X_OK), p"

# Replace rather than merge, so modules deleted in the fork do not linger. Same
# paths and root ownership as the upstream Dockerfile's COPY of these files;
# pyproject.toml carries the fork's extras (scripts/install_extras.py reads it).
RUN rm -rf /app/deeptutor /app/deeptutor_cli
COPY deeptutor/ /app/deeptutor/
COPY deeptutor_cli/ /app/deeptutor_cli/
COPY pyproject.toml /app/pyproject.toml
