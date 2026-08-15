#!/usr/bin/env bash
# ==============================================================================
# Script: db_tunnel.sh
# Purpose: Manage an SSH tunnel to PostgreSQL/TimescaleDB on the Cloud VM
# Host: cloud@[IP_ADDRESS] -> 127.0.0.1:5432
# ==============================================================================

SSH_USER="${REMOTE_SSH_USER:-cloud}"
SSH_HOST="${REMOTE_SSH_HOST:[IP_ADDRESS]}"
LOCAL_PORT="${LOCAL_DB_PORT:-5432}"
REMOTE_PORT="${REMOTE_DB_PORT:-5432}"
PID_FILE="/tmp/crypto_trading_db_tunnel.pid"

start_tunnel() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Tunnel is already running (PID: $(cat "$PID_FILE")). Local port: ${LOCAL_PORT}"
        return 0
    fi

    # Check if port is already occupied
    if ss -tulpn | grep -q ":${LOCAL_PORT} "; then
        echo "Warning: Port ${LOCAL_PORT} is already in use by another process."
        echo "Run: ss -tulpn | grep :${LOCAL_PORT}"
        return 1
    fi

    echo "Starting SSH tunnel to ${SSH_USER}@${SSH_HOST} (Local port ${LOCAL_PORT} -> Remote 127.0.0.1:${REMOTE_PORT})..."
    ssh -N -f -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${SSH_USER}@${SSH_HOST}"
    
    # Save PID of SSH tunnel
    TUNNEL_PID=$(pgrep -f "ssh.*-L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" | tail -n 1)
    if [ -n "$TUNNEL_PID" ]; then
        echo "$TUNNEL_PID" > "$PID_FILE"
        echo "Tunnel established successfully! (PID: $TUNNEL_PID)"
        echo "Connection URI: postgresql://postgres:postgres@localhost:${LOCAL_PORT}/crypto_trading"
    else
        echo "Failed to start SSH tunnel."
        return 1
    fi
}

stop_tunnel() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Killed SSH tunnel process (PID: $PID)"
        fi
        rm -f "$PID_FILE"
    fi

    # Also cleanup any matching background ssh process on this port
    PIDS=$(pgrep -f "ssh.*-L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}")
    if [ -n "$PIDS" ]; then
        kill $PIDS 2>/dev/null || true
        echo "Stopped background SSH tunnel processes."
    else
        echo "Tunnel is not running."
    fi
}

status_tunnel() {
    PIDS=$(pgrep -f "ssh.*-L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}")
    if [ -n "$PIDS" ]; then
        echo "Tunnel is RUNNING (PID: $PIDS)"
        echo "Listening on: 127.0.0.1:${LOCAL_PORT}"
        echo "Connection URI: postgresql://postgres:postgres@localhost:${LOCAL_PORT}/crypto_trading"
    else
        echo "Tunnel is NOT running."
    fi
}

check_connection() {
    echo "Testing connection to PostgreSQL on 127.0.0.1:${LOCAL_PORT}..."
    python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(
        dbname='crypto_trading',
        user='postgres',
        password='${POSTGRES_REMOTE_PASSWORD:-postgres}',
        host='127.0.0.1',
        port=${LOCAL_PORT},
        connect_timeout=3
    )
    cur = conn.cursor()
    cur.execute('SELECT version(), current_database();')
    print('Connection SUCCESSFUL!')
    print('Result:', cur.fetchone())
    conn.close()
except Exception as e:
    print('Connection FAILED:', e)
"
}

case "$1" in
    start)
        start_tunnel
        ;;
    stop)
        stop_tunnel
        ;;
    restart)
        stop_tunnel
        sleep 1
        start_tunnel
        ;;
    status)
        status_tunnel
        ;;
    check)
        check_connection
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|check}"
        echo ""
        echo "Commands:"
        echo "  start   - Launch background SSH tunnel (localhost:5432 -> cloud:5432)"
        echo "  stop    - Terminate running SSH tunnel"
        echo "  restart - Restart SSH tunnel"
        echo "  status  - Show tunnel status"
        echo "  check   - Test database query over local port 5432"
        exit 1
        ;;
esac
