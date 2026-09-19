#!/usr/bin/env bash
# Start an MCP server and a Cloudflare tunnel in front of it, then print the
# URL to paste into Claude.
#
#   PORT=8787 START_CMD="python3 -m yourpkg serve-http" scripts/tunnel-up.sh
#   TUNNEL_HOSTNAME=mcp.example.com scripts/tunnel-up.sh   # permanent URL
#
# START_CMD is optional: leave it unset if the server is already running.
set -euo pipefail

PORT="${PORT:-8787}"
ENDPOINT="${ENDPOINT:-/mcp}"
LOG_DIR="${LOG_DIR:-$HOME/.mcp-tunnel}"
mkdir -p "$LOG_DIR"

SERVER_PID=""
if [ -n "${START_CMD:-}" ]; then
    # shellcheck disable=SC2086
    $START_CMD >"$LOG_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$LOG_DIR/server.pid"
fi

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for the server rather than racing it: a tunnel that starts first just
# reports 502 until the origin appears, which reads like a tunnel problem.
for _ in $(seq 1 40); do
    curl -fsS "http://127.0.0.1:$PORT$ENDPOINT" -X POST -d '' >/dev/null 2>&1 && break
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 0.25
done
echo "server responding on port $PORT"

if [ -n "${TUNNEL_HOSTNAME:-}" ]; then
    CRED_ARGS=()
    [ -n "${TUNNEL_CREDENTIALS:-}" ] && CRED_ARGS=(--credentials-file "$TUNNEL_CREDENTIALS")
    cloudflared tunnel --url "http://localhost:$PORT" \
        "${CRED_ARGS[@]}" run "${TUNNEL_NAME:-mcp}" \
        >"$LOG_DIR/tunnel.log" 2>&1 &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > "$LOG_DIR/tunnel.pid"
    echo
    echo "  MCP server URL:  https://$TUNNEL_HOSTNAME$ENDPOINT"
    echo
else
    cloudflared tunnel --url "http://localhost:$PORT" \
        >"$LOG_DIR/tunnel.log" 2>&1 &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > "$LOG_DIR/tunnel.pid"
    printf 'waiting for the tunnel URL'
    URL=""
    for _ in $(seq 1 60); do
        URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" \
              2>/dev/null | head -1 || true)
        [ -n "$URL" ] && break
        printf '.'; sleep 1
    done
    echo
    [ -z "$URL" ] && { echo "no URL; see $LOG_DIR/tunnel.log" >&2; exit 1; }
    echo
    echo "  MCP server URL:  $URL$ENDPOINT"
    echo "  (a quick tunnel gets a new URL every run — see README, Path B)"
    echo
fi

echo "logs: $LOG_DIR  ·  Ctrl-C to stop"
wait
