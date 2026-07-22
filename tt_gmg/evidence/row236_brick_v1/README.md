# Row236 brick-major operator v1 evidence

Captured on 2026-07-14 from the real row236 fine dump on tt-quietbox. This is representation/oracle evidence only; it does not close a device timing or solver gate.

- Remote operator: `/home/ttuser/ttgmg/staged/row236_brick_v1.bin`
- Operator bytes: `2,375,315,456`
- Operator SHA-256: `f1a956701882a164c3ddb62e522bbe1dc4111d2ce37990c746f7860b69fe4674`
- Manifest SHA-256: `16da772076e9b48fbdbcb2338a76fef7f1c55647fbd3cac66dd2918e4336ee87`
- Verification SHA-256: `9fe8a460121db7782f5e4a5fca9a32a4dcbb526e635bb134dd748c55291034df`
- Remote logs: `/home/ttuser/ttgmg/evidence/brick_v1/{serialize,verify}.log`
- `tt-fold.service` remained active throughout the CPU-only serialization and host-oracle run.

The verifier independently applies both the original fp64 BCSR and the serialized bf16x3 brick stream to random, smooth, boundary-only, and constant vectors. It emulates the six retained TT MAC products in device order. All metadata and vector checks passed.
