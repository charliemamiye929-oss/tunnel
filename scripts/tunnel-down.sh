#!/usr/bin/env bash
# Stop whatever tunnel-up.sh started.
set -uo pipefail
LOG_DIR="${LOG_DIR:-$HOME/.mcp-tunnel}"
for name in server tunnel; do
    pidfile="$LOG_DIR/$name.pid"
    if [ -f "$pidfile" ] && kill "$(cat "$pidfile")" 2>/dev/null; then
        echo "stopped $name"
    else
        echo "$name was not running"
    fi
    rm -f "$pidfile"
done
