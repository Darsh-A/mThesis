#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source .venv/bin/activate

export OPENCODE_SERVER_PASSWORD=28264

opencode serve --port 4096 --hostname 0.0.0.0 &
OPENCODE_PID=$!

sleep 2

ngrok http 4096 --log=stdout > /tmp/ngrok_thesis.log 2>&1 &
NGROK_PID=$!

sleep 3

NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])")

echo ""
echo "=== opencode serve running (pid $OPENCODE_PID) ==="
echo "=== ngrok tunnel (pid $NGROK_PID) ==="
echo "=== Public URL: $NGROK_URL ==="
echo "=== Password: 28264 ==="
echo ""
