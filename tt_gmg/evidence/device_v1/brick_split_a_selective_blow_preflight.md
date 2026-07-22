# Split-A selective-b-low preflight

Date: 2026-07-15 UTC.

This is preflight evidence for a default-off transport-only successor to run52. It is not hardware timing or
device-correctness evidence.

The candidate preserves run52's coefficient data, reverse term order, `63/81` x-low term policy, HiFi4 product
order, FP32 destination accumulation, resident halo, output layout, and three-sample gate rule. It changes only
coefficient transport and CB addressing:

- RISC-0/NOC-0 publishes A-hi and A-mid in c_0: `18` tiles per three-term batch, double-buffered to `36` tiles.
- RISC-1/NOC-1 publishes A-low in c_5: `9` tiles per three-term batch, double-buffered to `18` tiles.
- The former combined c_0 published `27` tiles per batch, double-buffered to `54` tiles. Total A tiles and total
  L1 allocation are unchanged.
- Per group, NOC-0 coefficient pages fall from `729` to `486`; NOC-1 takes the remaining `243`. B-hi, B-mid, and
  selective B-low materialization retain their run52 roles.

Checks:

- Host layout/policy self-test: `brick_split_a_layout_selftest.log`, SHA-256
  `e95fe4a08fa405e3197e00d6fec5f8807a4ed8cd75d9776000e4fd5f2bf02006`; result `pass=1`, combined `27 = 18 + 9`
  tiles and `63` retained B-low terms.
- QuietBox Wormhole offline role compile: `brick_offline_split_a_selective_blow_roles_compile.log`, SHA-256
  `c650a0b5031dba854d69ea4c172c061bf27ef4bba6bac2688ee930a903de03ab`; result `0/5` cache hits and successful
  compilation of BRISC, NCRISC, TRISC0, TRISC1, and TRISC2.
- Arithmetic authority remains the run52 four-vector host oracle
  `brick_host_selective_blow_manhattan2_opposite_m1_p1_m1_reverse_all_vectors.json`; the split-A candidate does not
  alter an operand value or MAC sequence.

The first hardware use still requires a fresh boot, the hardened runner, an empty portal, no conflicting workload,
zero D-state tasks, expected holders only before the runner stops `tt-fold`, and no second workload on that boot.
