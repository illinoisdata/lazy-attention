#!/bin/bash
# Start server in background and redirect logs so it won't be stopped by job control
LOGFILE="$(pwd)/logs/block_generate_server.log"
mkdir -p "$(dirname "$LOGFILE")"
nohup env CUDA_VISIBLE_DEVICES=0 python3 server/block_generate_server.py \
	--model ldsjmdy/Tulu3-Block-FT --port 9898 --dtype bfloat16 \
	&>"$LOGFILE" &

# Wait for server to come up (adjust timeout if your model takes longer)
timeout=60
echo "Waiting up to ${timeout}s for server to start..."
while ! curl -sS --max-time 1 http://127.0.0.1:9898/generate >/dev/null 2>&1; do
	sleep 1
	timeout=$((timeout-1))
	if [ "$timeout" -le 0 ]; then
		echo "Server did not start in time; see $LOGFILE"
		echo "--- last 200 lines of server log ---"
		tail -n 200 "$LOGFILE" || true
		break
	fi
done

python3 example.py