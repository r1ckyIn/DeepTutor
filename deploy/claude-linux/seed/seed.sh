#!/usr/bin/env bash
# Idempotent setup after up.sh: download the local models, then configure
# DeepTutor through its own CLI and settings services. Re-run any time. Needs
# no token: it never calls docker compose.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)

for c in deeptutor deeptutor-ollama deeptutor-speaches; do
  for _ in $(seq 60); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' "$c" 2>/dev/null)" = healthy ] && continue 2
    sleep 5
  done
  echo "seed: $c is not healthy after 5 minutes; see: docker logs $c" >&2
  exit 1
done

# Same model ids as setup.json (embedding) and catalog.py (voice).
docker exec deeptutor-ollama ollama pull qwen3-embedding:0.6b
for m in speaches-ai/Kokoro-82M-v1.0-ONNX Systran/faster-whisper-small; do
  docker exec deeptutor-speaches curl -fsS -X POST "http://localhost:8000/v1/models/$m"
  echo
done

# As the app user, so the settings files stay readable by the backend. The JSON
# goes in by docker cp because `config apply` opens a path, and a non-root exec
# may not be allowed to reopen /dev/stdin (a root-owned pipe). `python -` only
# reads the fd it was given, so piping catalog.py is fine.
docker cp "$here/setup.json" deeptutor:/tmp/deeptutor-setup.json
docker exec -u deeptutor -w /app deeptutor python -m deeptutor_cli config apply /tmp/deeptutor-setup.json
out=$(docker exec -i -u deeptutor -w /app deeptutor python - < "$here/catalog.py")
echo "$out"
if [[ $out == *"restart required"* ]]; then
  docker restart deeptutor > /dev/null
  echo "seed: restarted deeptutor so login and the exec policy take effect"
fi
