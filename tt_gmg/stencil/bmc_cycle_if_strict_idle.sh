#!/usr/bin/env bash
set -euo pipefail

EVIDENCE_DIR=${1:-/home/ttuser/ttgmg/evidence/device_v1}
RUN_TAG=${2:-canonical_dynamic_bmc_preflight}
LOG="${EVIDENCE_DIR}/${RUN_TAG}.log"
PORTAL_URL=${TT_FOLD_PORTAL_URL:-http://100.117.137.85:8099}
CONTROLLER_DB=${TT_FOLD_CONTROLLER_DB:-/home/ttuser/ttbio/tt-bio/webportal/state/controller/controller.sqlite3}
PORTAL_DB=${TT_FOLD_PORTAL_DB:-/home/ttuser/ttbio/tt-bio/webportal/state/portal.db}
EXPECTED_FOLD_HOLDERS=${TT_FOLD_EXPECTED_DEVICE_HOLDERS:-3}
mkdir -p "${EVIDENCE_DIR}"
exec > >(tee "${LOG}") 2>&1

global_portal_state() {
    python3 - "${CONTROLLER_DB}" "${PORTAL_DB}" <<'PY'
import json
import sqlite3
import sys

controller_path, portal_path = sys.argv[1:]


def grouped_nonterminal(connection, table, terminal):
    placeholders = ",".join("?" for _ in terminal)
    query = (
        f"SELECT lower(status), count(*) FROM {table} "
        f"WHERE lower(status) NOT IN ({placeholders}) "
        "GROUP BY lower(status) ORDER BY lower(status)"
    )
    return {status: count for status, count in connection.execute(query, terminal)}


try:
    controller = sqlite3.connect(f"file:{controller_path}?mode=ro", uri=True)
    runs = grouped_nonterminal(
        controller, "runs", ("ok", "failed", "canceled", "cancelled")
    )
    jobs = grouped_nonterminal(
        controller, "jobs", ("ok", "failed", "canceled", "cancelled")
    )
    controller.close()
    portal = sqlite3.connect(f"file:{portal_path}?mode=ro", uri=True)
    submissions = grouped_nonterminal(
        portal, "submissions", ("completed", "failed", "canceled", "cancelled")
    )
    portal.close()
except Exception as exc:
    print(json.dumps({"idle": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
    raise SystemExit(2)

state = {
    "controller_runs_nonterminal": runs,
    "controller_jobs_nonterminal": jobs,
    "portal_submissions_nonterminal": submissions,
}
state["idle"] = not any(state.values())
print(json.dumps(state, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if state["idle"] else 1)
PY
}

session_jobs_empty() {
    local jobs
    if ! jobs=$(curl -fsS --max-time 5 "${PORTAL_URL}/api/jobs"); then
        return 1
    fi
    if ! python3 -c \
        'import json,sys; p=json.load(sys.stdin); raise SystemExit(0 if p.get("jobs") == [] else 1)' \
        <<<"${jobs}"; then
        return 1
    fi
    printf '%s\n' "${jobs}"
}

conflicting_workload() {
    pgrep -a -f \
        "[m]etal_example_.*(brick_spmv|canonical_packed_spmv|canonical_dynamic_spmv)|[r]un_.*_hw.sh" \
        || true
}

dstate_tasks() {
    ps -eo stat=,pid=,comm= | awk '$1 ~ /^D/ {print}'
}

holder_pids() {
    sudo -n fuser /dev/tenstorrent/0 /dev/tenstorrent/1 \
        /dev/tenstorrent/2 /dev/tenstorrent/3 2>/dev/null | \
        tr " " "\n" | awk 'NF && !seen[$1]++ {print $1}' | sort -n
}

cycle_sent=0
restart_on_failure() {
    if ((cycle_sent == 0)); then
        echo "RECOVERY cycle_not_acknowledged restarting_tt_fold=1"
        sudo -n systemctl reset-failed tt-fold || true
        sudo -n systemctl start tt-fold || true
    fi
}
trap restart_on_failure EXIT

echo "PREFLIGHT start=$(date --iso-8601=seconds)"
if ! jobs=$(session_jobs_empty); then
    echo "PREFLIGHT portal_jobs=unavailable_or_nonempty"
    exit 1
fi
echo "PREFLIGHT portal_jobs=${jobs}"
if ! global_state=$(global_portal_state); then
    echo "PREFLIGHT global_portal_state=${global_state:-unreadable}"
    exit 1
fi
echo "PREFLIGHT global_portal_state=${global_state}"
systemd_jobs=$(systemctl list-jobs --no-legend --no-pager)
echo "PREFLIGHT systemd_jobs=${systemd_jobs:-none}"
[[ -z "${systemd_jobs}" ]]
users=$(who)
echo "PREFLIGHT interactive_users=${users:-none}"
[[ -z "${users}" ]]
dstate=$(dstate_tasks)
echo "PREFLIGHT dstate=${dstate:-none}"
[[ -z "${dstate}" ]]
workload=$(conflicting_workload)
echo "PREFLIGHT conflicting_workload=${workload:-none}"
[[ -z "${workload}" ]]
[[ "$(systemctl is-active tt-fold)" == "active" ]]

mapfile -t holders < <(holder_pids)
echo "PREFLIGHT holder_pids=${holders[*]:-none}"
if ((${#holders[@]} != EXPECTED_FOLD_HOLDERS)); then
    echo "PREFLIGHT unexpected_holder_count=${#holders[@]}"
    exit 1
fi
for pid in "${holders[@]}"; do
    cgroup=$(cat "/proc/${pid}/cgroup")
    if [[ "${cgroup}" != *"/system.slice/tt-fold.service"* ]]; then
        echo "PREFLIGHT foreign_holder_pid=${pid} cgroup=${cgroup}"
        exit 1
    fi
done
echo "PREFLIGHT holder_ownership=tt-fold-only"

sudo -n systemctl stop tt-fold
echo "PREFLIGHT tt_fold_after_stop=$(systemctl is-active tt-fold 2>/dev/null || true)"
[[ "$(systemctl is-active tt-fold 2>/dev/null || true)" != "active" ]]

if ! global_state=$(global_portal_state); then
    echo "PREFLIGHT post_stop_global_portal_state=${global_state:-unreadable}"
    exit 1
fi
systemd_jobs=$(systemctl list-jobs --no-legend --no-pager)
users=$(who)
dstate=$(dstate_tasks)
workload=$(conflicting_workload)
mapfile -t holders < <(holder_pids)
# The session-scoped HTTP API is owned by tt-fold and is expected to be down
# here.  The read-only controller/portal databases remain the global authority.
echo "PREFLIGHT post_stop_portal_jobs=service_stopped_global_db_authority"
echo "PREFLIGHT post_stop_global_portal_state=${global_state}"
echo "PREFLIGHT post_stop_systemd_jobs=${systemd_jobs:-none}"
echo "PREFLIGHT post_stop_interactive_users=${users:-none}"
echo "PREFLIGHT post_stop_dstate=${dstate:-none}"
echo "PREFLIGHT post_stop_workload=${workload:-none}"
echo "PREFLIGHT post_stop_holders=${holders[*]:-none}"
[[ -z "${systemd_jobs}" && -z "${users}" && -z "${dstate}" && \
   -z "${workload}" && ${#holders[@]} -eq 0 ]]

old_boot_id=$(cat /proc/sys/kernel/random/boot_id)
echo "POWER_CYCLE old_boot_id=${old_boot_id}"
cycle_output=$(sudo -n ipmitool chassis power cycle)
echo "${cycle_output}"
if [[ "${cycle_output}" != *"Chassis Power Control: Cycle"* ]]; then
    echo "POWER_CYCLE command_acknowledged=0"
    exit 1
fi
cycle_sent=1
echo "POWER_CYCLE command_acknowledged=1"
