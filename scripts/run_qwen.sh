#!/usr/bin/env bash
# run_qwen.sh <tag> <seed> [extra main.py flags...]
#
# One Qwen2.5-0.5B W3 run. Idempotent: a run whose JSON already exists is skipped,
# so the runbook can be replayed on a fresh instance after an instance death.
# On completion writes results/qwen05b/<tag>_w3_s<seed>.json, mirrors it to the
# persistent volume, then commits+pushes under a lock. Git failure never fails
# the run -- the result is already on disk and mirrored.
set -uo pipefail

TAG=$1; SEED=$2; shift 2
REPO=/home/boa-kronecker-gap
MIRROR=/home/jl_fs/results/qwen05b_mirror
NAME="${TAG}_w3_s${SEED}"
OUT="$REPO/results/qwen05b/${NAME}.json"
LOG="$MIRROR/${NAME}.log"
mkdir -p "$REPO/results/qwen05b" "$MIRROR"

if [ -f "$OUT" ]; then echo "[skip] $NAME (json exists)"; exit 0; fi

EXTRA="$*"
ARGS="--llm_path /home/models/qwen2.5-0.5b --w_bits 3 --block_v --qparam_comput Hessian --seed $SEED --cache_dir /home/jl_fs/calib $EXTRA"
GITHASH=$(cd "$REPO" && git rev-parse --short HEAD)

echo "[start] $NAME :: $ARGS"
T0=$(date +%s)
HF_HOME=/home/jl_fs/hf PYTHONUNBUFFERED=1 /home/venv_boa/bin/python -u "$REPO/main.py" $ARGS > "$LOG" 2>&1
RC=$?
T1=$(date +%s)
WALL=$((T1-T0))

TAG="$TAG" SEED="$SEED" ARGS="$ARGS" GITHASH="$GITHASH" RC="$RC" WALL="$WALL" \
LOG="$LOG" OUT="$OUT" /home/venv_boa/bin/python - <<'PY'
import ast, json, os
log, out = os.environ["LOG"], os.environ["OUT"]
res = {}
for line in reversed(open(log, errors="replace").read().splitlines()):
    t = line.strip()
    if t.startswith("{") and "wikitext2" in t:
        try:
            res = ast.literal_eval(t); break
        except Exception:
            continue
rec = {
    "tag": os.environ["TAG"], "seed": int(os.environ["SEED"]),
    "model": "Qwen/Qwen2.5-0.5B", "w_bits": 3,
    "git_commit": os.environ["GITHASH"], "rc": int(os.environ["RC"]),
    "wall_s": int(os.environ["WALL"]), "args": os.environ["ARGS"],
    "wikitext2": res.get("wikitext2"), "c4-new": res.get("c4-new"),
    "quant_time_s": res.get("time"), "results": res,
}
json.dump(rec, open(out, "w"), indent=2)
print(f"[done] {rec['tag']} s{rec['seed']} rc={rec['rc']} "
      f"wiki2={rec['wikitext2']} c4={rec['c4-new']} wall={rec['wall_s']}s")
PY

cp -f "$OUT" "$MIRROR/" 2>/dev/null

# --- bank it: serialise pushes, never block the pipeline on git ---
# NO_PUSH=1 holds the result on disk without publishing it -- used when a run is
# launched speculatively, before its configuration has been confirmed.
if [ "${NO_PUSH:-0}" = "1" ]; then echo "[git] NO_PUSH=1, holding $NAME locally"; exit 0; fi
(
  flock -w 300 9 || { echo "[git] lock timeout for $NAME"; exit 0; }
  cd "$REPO" || exit 0
  for i in 1 2 3; do
    git add "results/qwen05b/${NAME}.json" >/dev/null 2>&1
    git diff --cached --quiet && { echo "[git] nothing to commit for $NAME"; break; }
    WIKI=$(/home/venv_boa/bin/python -c "import json;print(json.load(open('$OUT'))['wikitext2'])" 2>/dev/null)
    git -c user.name="Claude Opus 5 (1M context)" -c user.email="noreply@anthropic.com" \
        commit -q -m "$(printf '%s s%s: wiki2 %s\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01WLrtcYZdtuyRLQkUWr7f6F' "$TAG" "$SEED" "$WIKI")" >/dev/null 2>&1
    git pull --rebase --autostash -q origin main >/dev/null 2>&1
    if git push -q origin main >/dev/null 2>&1; then echo "[git] pushed $NAME"; break; fi
    echo "[git] push attempt $i failed for $NAME"; sleep $((i*5))
  done
) 9>/home/jl_fs/.gitpush.lock

exit 0
