# Speech for DeepTutor through Microsoft Edge's online neural voices (edge-tts,
# no account or key). Build from the repo root:
#   docker build -t deeptutor-edge-tts:local deploy/claude-linux/edge-tts
# When Microsoft changes the protocol, bump edge-tts and rebuild.
FROM python:3.11-slim
RUN pip install --no-cache-dir "edge-tts==7.3.1"
COPY server.py /app/server.py
USER nobody
CMD ["python", "/app/server.py"]
