#!/usr/bin/env bash
set -uo pipefail

EVIDENCE_DIR=${1:-/home/ttuser/ttgmg/evidence/device_v1}
RUN_TAG=${2:-canonical_dynamic_page_spmv_run1}
LOG="${EVIDENCE_DIR}/${RUN_TAG}.log"
PORTAL_URL=${TT_FOLD_PORTAL_URL:-http://100.117.137.85:8099}
CONTROLLER_DB=${TT_FOLD_CONTROLLER_DB:-/home/ttuser/ttbio/tt-bio/webportal/state/controller/controller.sqlite3}
PORTAL_DB=${TT_FOLD_PORTAL_DB:-/home/ttuser/ttbio/tt-bio/webportal/state/portal.db}
MIN_FOLD_SETTLE_SECONDS=${TT_GMG_MIN_FOLD_SETTLE_SECONDS:-600}
POST_FOLD_STOP_SETTLE_SECONDS=${TT_GMG_POST_FOLD_STOP_SETTLE_SECONDS:-10}
WORKLOAD_TIMEOUT_SECONDS=${TT_GMG_WORKLOAD_TIMEOUT_SECONDS:-300}
VECTOR_TAG=${TT_GMG_VECTOR_TAG:-random_seed351}
PACKED_ROOT=${TT_GMG_CANONICAL_PACKED_ROOT:-/home/ttuser/ttgmg/canonical_packed_v1}
DYNAMIC_ROOT=${TT_GMG_CANONICAL_DYNAMIC_ROOT:-/home/ttuser/ttgmg/canonical_pcg_dynamic_v1}
EXPECTED_FOLD_WORKERS=${TT_FOLD_EXPECTED_ONLINE_WORKERS:-3}
EXPECTED_FOLD_HOLDERS=${TT_FOLD_EXPECTED_DEVICE_HOLDERS:-3}
BINARY=/home/ttuser/src/tt-metal-073/build_Release/programming_examples/metal_example_canonical_dynamic_spmv
# One TT workload per boot across every brick/canonical candidate.
BOOT_MARKER=/home/ttuser/ttgmg/brick_v1/.last_brick_workload_boot_id
mkdir -p "${EVIDENCE_DIR}"
exec > >(tee "${LOG}") 2>&1

jobs_are_empty() {
    python3 -c 'import json, sys; payload = json.load(sys.stdin); sys.exit(0 if payload.get("jobs") == [] else 1)'
}

# /api/jobs is session scoped.  The controller and portal databases are the
# global authority; unknown/nonterminal state fails closed.
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

restart_fold() {
    echo "SERVICE restarting tt-fold"
    sudo -n systemctl reset-failed tt-fold || true
    sudo -n systemctl start tt-fold || true
    local state jobs cluster online holders
    for _ in $(seq 1 90); do
        state=$(systemctl is-active tt-fold 2>/dev/null || true)
        jobs=$(curl -fsS --max-time 3 "${PORTAL_URL}/api/jobs" 2>/dev/null || true)
        cluster=$(curl -fsS --max-time 3 "${PORTAL_URL}/api/cluster" 2>/dev/null || true)
        online=$(python3 -c \
            'import json,sys; print(int(json.load(sys.stdin).get("online_workers") or 0))' \
            <<<"${cluster}" 2>/dev/null || echo 0)
        holders=$(sudo -n fuser /dev/tenstorrent/0 /dev/tenstorrent/1 \
            /dev/tenstorrent/2 /dev/tenstorrent/3 2>/dev/null | tr " " "\n" | \
            awk 'NF {seen[$1]=1} END {for (pid in seen) n++; print n+0}')
        if [[ "${state}" == "active" && -n "${jobs}" ]] && \
                ((online >= EXPECTED_FOLD_WORKERS)) && \
                ((holders >= EXPECTED_FOLD_HOLDERS)); then
            echo "SERVICE tt-fold=active api=ready online_workers=${online} device_holders=${holders} jobs=${jobs}"
            return
        fi
        sleep 2
    done
    echo "SERVICE tt-fold restoration incomplete state=${state:-unknown} online_workers=${online:-0} device_holders=${holders:-0} jobs=${jobs:-unavailable} cluster=${cluster:-unavailable}"
}

echo "RUN start=$(date --iso-8601=seconds)"
python3 - "${PACKED_ROOT}" "${DYNAMIC_ROOT}" "${VECTOR_TAG}" <<'PY'
import hashlib
import json
import pathlib
import sys

packed_root = pathlib.Path(sys.argv[1]).resolve()
dynamic_root = pathlib.Path(sys.argv[2]).resolve()
tag = sys.argv[3]
layout_path = packed_root / "canonical_packed_layout.json"
vector_manifest_path = packed_root / f"canonical_packed_vector_{tag}.json"
report_path = dynamic_root / "canonical_pcg_gather_analysis.json"
layout = json.loads(layout_path.read_text())
vector = json.loads(vector_manifest_path.read_text())
report = json.loads(report_path.read_text())

compact = report.get("compact_gather_plan") or {}
bundle = report.get("page_reader_vector_bundle") or {}
order = compact.get("device_stream_order") or {}
if not (
    compact.get("passing") is True
    and compact.get("device_page_table_mismatches") == 0
    and compact.get("device_page_gather_plan_bytes") == 70330180
    and order.get("runtime_order_mode") == 3
    and order.get("is_permutation") is True
    and order.get("each_publish_batch_single_component") is True
    and order.get("each_publish_batch_descending_offsets") is True
    and bundle.get("passing") is True
    and bundle.get("verified_bf16_words") == 314523648
    and bundle.get("bit_mismatches_by_split") == [0, 0, 0]
):
    raise SystemExit("dynamic plan report is not acceptance-green")
if pathlib.Path(report["canonical_layout"]["path"]).resolve() != layout_path:
    raise SystemExit("dynamic report points at another canonical layout")
if pathlib.Path(bundle["vector_manifest"]["path"]).resolve() != vector_manifest_path:
    raise SystemExit("dynamic report points at another vector manifest")

records = []
for key in (
    "palette_a0", "palette_a1", "palette_a2", "fallback_a0",
    "fallback_a1", "fallback_a2", "group_class",
):
    records.append(layout["files"][key])
records.append(vector["files"]["packed_reference"])
for key in ("page_vector_b0", "page_vector_b1", "page_vector_b2"):
    records.append(bundle["files"][key])
records.append(compact["files"]["device_group_vector_page_table"])
records.append(compact["files"]["neighbor_group_page_lane"])

checked = 0
for record in records:
    path = pathlib.Path(record["path"])
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise SystemExit(f"artifact size/missing failure: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(32 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise SystemExit(f"artifact hash failure: {path}")
    checked += 1
print(f"ARTIFACT_PREFLIGHT pass=1 files={checked} vector={tag} dynamic_plan=1")
PY
artifact_rc=$?
if ((artifact_rc != 0)); then
    echo "SAFETY canonical dynamic artifact preflight failed"
    exit 103
fi

CANONICAL_DYNAMIC_HOST_PREFLIGHT_ONLY=1 \
    "${BINARY}" "${PACKED_ROOT}" "${DYNAMIC_ROOT}" 3 "${VECTOR_TAG}"
host_preflight_rc=$?
if ((host_preflight_rc != 0)); then
    echo "SAFETY canonical dynamic host preflight failed"
    exit 110
fi

jobs=$(curl -fsS --max-time 5 "${PORTAL_URL}/api/jobs" 2>/dev/null || true)
if [[ -z "${jobs}" ]]; then
    echo "SAFETY portal jobs endpoint unavailable"
    exit 92
fi
if ! jobs_are_empty <<<"${jobs}"; then
    echo "SAFETY portal has live or unparseable jobs: ${jobs}"
    exit 93
fi
echo "SAFETY portal_jobs=${jobs}"
global_state=$(global_portal_state 2>&1)
global_rc=$?
echo "SAFETY global_portal_state=${global_state}"
if ((global_rc != 0)); then
    echo "SAFETY global portal/controller state is busy or unreadable"
    exit 104
fi

boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)
if [[ -z "${boot_id}" ]]; then
    echo "SAFETY cannot determine host boot id"
    exit 99
fi
previous_boot_id=$(cat "${BOOT_MARKER}" 2>/dev/null || true)
if [[ "${previous_boot_id}" == "${boot_id}" ]]; then
    echo "SAFETY fresh_boot_required boot_id=${boot_id}"
    exit 100
fi
echo "SAFETY fresh_boot_id=${boot_id}"

active_enter_us=$(systemctl show tt-fold \
    --property=ActiveEnterTimestampMonotonic \
    --value 2>/dev/null || true)
uptime_seconds=$(awk '{print int($1)}' /proc/uptime)
if [[ ! "${active_enter_us}" =~ ^[0-9]+$ || "${active_enter_us}" == 0 ]]; then
    echo "SAFETY cannot determine tt-fold active age"
    exit 97
fi
active_age_seconds=$((uptime_seconds - active_enter_us / 1000000))
if ((active_age_seconds < MIN_FOLD_SETTLE_SECONDS)); then
    remaining=$((MIN_FOLD_SETTLE_SECONDS - active_age_seconds))
    echo "SAFETY waiting_for_tt_fold_settle_seconds=${remaining}"
    while ((remaining > 0)); do
        sleep 2
        jobs=$(curl -fsS --max-time 5 "${PORTAL_URL}/api/jobs" 2>/dev/null || true)
        if [[ -z "${jobs}" ]] || ! jobs_are_empty <<<"${jobs}"; then
            echo "SAFETY portal changed while waiting to settle: ${jobs:-unavailable}"
            exit 98
        fi
        global_state=$(global_portal_state 2>&1)
        global_rc=$?
        if ((global_rc != 0)); then
            echo "SAFETY global portal/controller state changed while waiting: ${global_state}"
            exit 105
        fi
        active_age_seconds=$((active_age_seconds + 2))
        remaining=$((MIN_FOLD_SETTLE_SECONDS - active_age_seconds))
    done
fi
echo "SAFETY tt_fold_active_age_seconds=${active_age_seconds}"

if pgrep -a -f "[m]etal_example_.*(brick_spmv|canonical_packed_spmv|canonical_dynamic_spmv)"; then
    echo "SAFETY conflicting workload process found"
    exit 94
fi
dstate=$(ps -eo stat=,pid=,comm= | awk '$1 ~ /^D/ {print}' | tr "\n" ";")
echo "SAFETY pre_stop_dstate=${dstate:-none}"
if [[ -n "${dstate}" ]]; then
    exit 95
fi

global_state=$(global_portal_state 2>&1)
global_rc=$?
echo "SAFETY pre_stop_global_portal_state=${global_state}"
if ((global_rc != 0)); then
    echo "SAFETY global portal/controller state changed before service stop"
    exit 106
fi
systemd_jobs=$(systemctl list-jobs --no-legend --no-pager | wc -l | tr -d " ")
echo "SAFETY pre_stop_systemd_jobs=${systemd_jobs}"
if [[ "${systemd_jobs}" != "0" ]]; then
    exit 107
fi
interactive_users=$(who | wc -l | tr -d " ")
echo "SAFETY pre_stop_interactive_users=${interactive_users}"
if [[ "${interactive_users}" != "0" ]]; then
    exit 108
fi

trap restart_fold EXIT
sudo -n systemctl stop tt-fold
state=$(systemctl is-active tt-fold 2>/dev/null || true)
echo "SERVICE after_stop=${state}"
if [[ "${state}" == "active" ]]; then
    echo "SAFETY stop failed"
    exit 90
fi

global_state=$(global_portal_state 2>&1)
global_rc=$?
echo "SAFETY post_stop_global_portal_state=${global_state}"
if ((global_rc != 0)); then
    echo "SAFETY global portal/controller state became busy before holder release"
    exit 109
fi

holders=$(sudo -n fuser /dev/tenstorrent/* 2>/dev/null || true)
echo "SAFETY device_holders=${holders:-none}"
if [[ -n "${holders}" ]]; then
    exit 91
fi
devices=$(compgen -G "/dev/tenstorrent/*" | sort | tr "\n" " ")
echo "SAFETY devices=${devices}"
dstate=$(ps -eo stat=,pid=,comm= | awk '$1 ~ /^D/ {print}' | tr "\n" ";")
echo "SAFETY dstate=${dstate:-none}"
if [[ -n "${dstate}" ]]; then
    exit 96
fi

if ((POST_FOLD_STOP_SETTLE_SECONDS > 0)); then
    echo "SAFETY post_stop_settle_seconds=${POST_FOLD_STOP_SETTLE_SECONDS}"
    sleep "${POST_FOLD_STOP_SETTLE_SECONDS}"
fi
holders=$(sudo -n fuser /dev/tenstorrent/* 2>/dev/null || true)
echo "SAFETY post_settle_device_holders=${holders:-none}"
if [[ -n "${holders}" ]]; then
    exit 101
fi
dstate=$(ps -eo stat=,pid=,comm= | awk '$1 ~ /^D/ {print}' | tr "\n" ";")
echo "SAFETY post_settle_dstate=${dstate:-none}"
if [[ -n "${dstate}" ]]; then
    exit 102
fi

printf '%s\n' "${boot_id}" > "${BOOT_MARKER}"

export TT_METAL_HOME=/home/ttuser/src/tt-metal-073
export TT_METAL_RUNTIME_ROOT=/home/ttuser/src/tt-metal-073
export TT_METAL_LOGGER_LEVEL=ERROR
export HOME=/home/ttuser
ulimit -n 1048576 2>/dev/null || true
cd /home/ttuser/src/tt-metal-073

echo "WORKLOAD begin=$(date --iso-8601=seconds)"
echo "WORKLOAD timeout_seconds=${WORKLOAD_TIMEOUT_SECONDS}"
timeout "${WORKLOAD_TIMEOUT_SECONDS}s" \
    "${BINARY}" "${PACKED_ROOT}" "${DYNAMIC_ROOT}" 3 "${VECTOR_TAG}"
rc=$?
echo "WORKLOAD exit=${rc} end=$(date --iso-8601=seconds)"
exit "${rc}"
