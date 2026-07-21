# Run54 affine-corner-fold device deadlock diagnosis

Scope: APHYSICAL TT-GMG diagnostic evidence. This is not a timing or correctness result.

Run tag: `brick_spmv_run54_affine_corner_fold_vectorized_alow_manhattan2_blow_manhattan2_opposite_reverse_freshboot`.

## Observed state

- The hardened runner passed its fresh-boot, empty-portal, no-D-state, no-holder, and sole-workload checks.
- The vectorized real-artifact host benchmark completed beforehand: eight A-low shards read in `506.237 ms`, folded
  in `43.152 ms`, and completed in `601.866 ms` total.
- The TT inspector recorded all affine reader/compute kernel compiles finishing in about `10 ms`, then recorded mesh
  workload 0 changing from `InFlight` to `Committed`.
- The workload subsequently remained in its first device execution with one busy host polling thread, no child
  compiler, no coefficient file descriptor, and zero D-state tasks. It emitted no sample before the runner timeout.

These observations falsify the earlier Run53 hypothesis that scalar host preprocessing was the active timeout cause.
Run53 did contain an unnecessarily scalar transform, but the repeated no-sample failure is a device-kernel deadlock.

## Source-level cause

The first affine reader compacted each three-term A publish to a variable `18–27` tiles while c_0 retained the
proven `54`-tile two-publish capacity. In reverse term order the first publish sizes are:

```text
18, 27, 18, 27, 27, 27, ...
```

After the first two publishes, the producer pointer is at tile 45. The next 18-tile contiguous reservation crosses
the 54-tile ring boundary. Total free space can be sufficient while no legal contiguous 18-tile span exists, so the
producer and consumer can wait indefinitely. This matches the first-execution stall.

## Correction

The successor keeps the proven fixed `27`-tile publish/pop cadence and fixed `54`-tile CB capacity. For corner terms
it reserves the three A-low slots but performs no A-low DRAM read, and the compute kernel never consumes those slots.
Thus the intended A-low bandwidth and MAC reduction remains while the circular-buffer schedule stays fixed. The
correction requires offline BRISC/TRISC compilation and a new fresh-boot measurement; Run54 itself remains an
unmeasured negative diagnostic and cannot replace Run52.
