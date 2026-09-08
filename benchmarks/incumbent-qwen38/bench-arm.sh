#!/usr/bin/env bash
# bench-arm.sh <tag> <server-bin> <model> [server args...]
# Runs the fixed workload set against a llama-server instance, 3 runs each,
# greedy, seed 42, 384 tokens. Server log per run is parsed later by parse.py.
set -uo pipefail
TAG="$1"; shift
BIN="$1"; shift
MODEL="$1"; shift
PORT=$(( 9500 + RANDOM % 300 ))
CTX="${CTX:-16384}"
RUNS="${RUNS:-3}"
WLS="${WLS:-chat.txt code.txt reason.txt}"
KVK="${KVK:-q4_0}"
KVV="${KVV:-q4_0}"
RES="results/${TAG}"
mkdir -p "$RES"

"$BIN" -m "$MODEL" -c "$CTX" -ngl 999 -fa on --jinja \
  --cache-type-k "$KVK" --cache-type-v "$KVV" \
  --host 127.0.0.1 --port "$PORT" "$@" > "$RES/server.log" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
trap cleanup EXIT

up=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 'ok'; then up=1; break; fi
  sleep 1
done
if [ "$up" != 1 ]; then echo "$TAG: server failed to start" >&2; tail -20 "$RES/server.log" >&2; exit 1; fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$RES/vram.txt"

for name in $WLS; do
  wl="workloads/$name"
  [ -f "$wl" ] || { echo "$wl: missing" >&2; continue; }
  name=${name%.txt}
  python3 - "$wl" "$PORT" "$name" "$RES" "$RUNS" <<'EOF'
import json, sys, time, urllib.request
wl, port, name, res, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
prompt = open(wl).read()
for run in range(1, runs + 1):
    t0 = time.time()
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 384, "temperature": 0, "seed": 42,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.load(r)
    except Exception as e:
        print(f"{name} run{run}: ERROR {e}")
        continue
    wall = time.time() - t0
    msg = out["choices"][0]["message"]
    text = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or msg.get("reasoning", "") or ""
    open(f"{res}/{name}.run{run}.json", "w").write(json.dumps(out, indent=2))
    open(f"{res}/{name}.run{run}.out", "w").write(reasoning + "\n" + text)
    open(f"{res}/{name}.run{run}.wall", "w").write(f"{wall:.2f}\n")
    print(f"{name} run{run}: wall={wall:.1f}s chars={len(text)}")
EOF
done
echo "$TAG done"
