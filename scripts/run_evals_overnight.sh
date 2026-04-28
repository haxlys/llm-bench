#!/usr/bin/env bash
#
# Overnight wrapper: stops production launchd agents, runs the full eval matrix
# across all 6 model variants, then restores agents (always — even on failure).
#
# Usage:
#   bash scripts/run_evals_overnight.sh
#
# Or, fully detached (recommended for actual overnight):
#   nohup bash scripts/run_evals_overnight.sh > /tmp/llm-evals-overnight.log 2>&1 &
#   disown
#   tail -f /tmp/llm-evals-overnight.log
#
# Environment overrides:
#   SUITE=full|smoke         # default: full
#   LIMIT=<int>              # override per-task sample limit
#   VARIANTS="key1 key2"     # space-separated variant keys; empty = --all-variants
#   LAUNCH_AGENTS="com.you.x com.you.y"
#                            # space-separated launchd labels to bootout before
#                            # the run and bootstrap back on EXIT (always).
#                            # Empty default = no launchd management.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/results/overnight_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_LOG="$LOG_DIR/$TS.log"

SUITE="${SUITE:-full}"
LIMIT="${LIMIT:-}"
VARIANTS="${VARIANTS:-}"

# launchd agents to stop before benchmarking (Metal contention).
# Override via env: LAUNCH_AGENTS="com.you.foo com.you.bar"
# Empty = no launchd management — kill GPU-using processes manually instead.
UID_=$(id -u)
read -ra LAUNCH_AGENTS <<< "${LAUNCH_AGENTS:-}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

# --- bootstrap agents back; called from EXIT trap ---
restore_agents() {
    if [[ ${#LAUNCH_AGENTS[@]} -eq 0 ]]; then return; fi
    log "==== Restoring launchd agents ===="
    for agent in "${LAUNCH_AGENTS[@]}"; do
        local plist="$HOME/Library/LaunchAgents/${agent}.plist"
        if [[ -f "$plist" ]]; then
            log "  bootstrap $agent"
            launchctl bootstrap "gui/$UID_" "$plist" 2>>"$RUN_LOG" || true
        else
            log "  (skipped: $plist not found)"
        fi
    done
    log "==== Agents restored ===="
}
trap restore_agents EXIT

# --- 1. stop production agents ---
log "=== STARTING overnight eval run ($TS) ==="
log "suite=$SUITE limit='${LIMIT}' variants='${VARIANTS:-all}'"
log "launchd agents to manage: ${LAUNCH_AGENTS[*]:-none}"
log
if [[ ${#LAUNCH_AGENTS[@]} -gt 0 ]]; then
    log "==== Stopping launchd agents ===="
    for agent in "${LAUNCH_AGENTS[@]}"; do
        log "  bootout $agent"
        launchctl bootout "gui/$UID_/$agent" 2>>"$RUN_LOG" || true
    done
else
    log "==== No launchd agents configured (set LAUNCH_AGENTS env var if needed) ===="
fi

# Verify nothing on the production ports
sleep 3
if lsof -nP -iTCP:8080 -iTCP:8081 -iTCP:8082 -sTCP:LISTEN >/dev/null 2>&1; then
    log "WARNING: ports 8080/8081/8082 still listening — Metal contention possible"
fi
log "free memory: $(vm_stat | awk '/Pages free/ {printf "%.1fGB", $3*16384/1024/1024/1024}')"

# --- 2. run the full matrix ---
log
log "==== Running run_evals.py ===="
ARGS=(--suite "$SUITE")
if [[ -n "$LIMIT" ]]; then ARGS+=(--limit "$LIMIT"); fi
if [[ -n "$VARIANTS" ]]; then
    for v in $VARIANTS; do ARGS+=(--variant "$v"); done
else
    ARGS+=(--all-variants)
fi
ARGS+=(--skip-existing)
log "  cmd: uv run python scripts/run_evals.py ${ARGS[*]}"

uv run python scripts/run_evals.py "${ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"
RC=${PIPESTATUS[0]}
log
log "==== run_evals.py exited rc=$RC ===="

# --- 3. aggregate eval scores into eval_summary_*.csv for dashboard ---
log
log "==== Aggregating eval scores ===="
uv run python scripts/aggregate_evals.py 2>&1 | tee -a "$RUN_LOG" || true

# trap restore_agents will fire on exit
log "=== ENDING overnight eval run ($(date +%Y%m%dT%H%M%S)) ==="
exit "$RC"
