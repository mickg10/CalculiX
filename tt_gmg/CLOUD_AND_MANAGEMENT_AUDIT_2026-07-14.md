# TT-GMG management and “cloud” audit — 2026-07-14

Scope: read-only infrastructure orientation for the APHYSICAL solver-tooling workstream. This is not product,
geometry, structural, acoustic, or fitment evidence. No credential value was recorded or written.

## Term reconciliation

- The “BNC” reference is consistent with **BMC**: the quietbox Baseboard Management Controller. Local,
  passwordless `sudo -n ipmitool` is the working control path. It is used only after an idle/no-holder/no-D-state
  preflight for a true chassis power cycle. The observed controller is the ASRock Rack BMC on the SIENAD8-2L2T
  platform, firmware `2.05`, LAN `10.0.0.48`, IPMI `2.0`. `tt-smi -r` remains forbidden because it has re-wedged
  healthy cards. One direct Redfish Basic-auth probe was rejected with HTTP 401; no further password guessing was
  performed because in-band passwordless IPMI is already the known authority.
- The “GVM thingie” is **KVM**: `glmkvmigor`, a GL.iNet RM10 (`100.100.199.8` on Tailscale,
  `10.0.0.199` on LAN) running Buildroot
  `rmq1-1.8.1-beta1-5-g0549ea7d65`. It provides an authenticated SSH jump path and live HDMI capture for the
  quietbox. It is a management console, not a compute VM. The repository also contains **GMG**, geometric
  multigrid, which is the solver algorithm in TT-GMG. There is no substantive project subsystem named GVM.
- The repository’s “lattice cloud” is the sparse holder geometry represented by the row236 operator. It is not a
  public-cloud deployment. The active brick audit measures about 3.7% occupied lattice volume.

## The four different “cloud” meanings

1. **Public cloud:** a live AWS `default` profile exists and signed STS identity succeeds. The principal is an IAM
   user, but it is denied billing and resource-discovery calls, so account existence is proved while inventory,
   storage, instances, and spend remain unknown.
2. **Management cloud:** the GL.iNet relay services plus Tailscale that make the RM10 KVM reachable. This is a
   control path, not compute capacity.
3. **Physical product cloud:** the full-360-degree lattice-cloud microphone mount. The promoted physical frontier in
   `lattice_holder_goal.md` is Row681 v18b: one piece including the legal lug, visual green, actual voxel material
   fraction `0.14095`, static grid green, and Gate 13 still red at `+18.09 dB` with one high-frequency band green.
   `worklog.md` also records a newer v19b print-spec/prototype variant with bottom petals and the Carbon feature-size
   floor, but nonlinear petal insertion/retention remains pending and it has not superseded v18b as product authority.
4. **Solver lattice cloud:** row236 is the APHYSICAL sparse FE workload used by TT-GMG. Its brick box is about 3.7%
   occupied. Improving its SpMV changes tooling speed only; it does not change or validate physical holder geometry.

The Row682 “connected-domain cloud” branch is a separate geometry experiment. Its latest territory/SDF candidates
remain same-cycle visual-red preflight evidence and must not be voxelized, sent to CCX, material-bracketed, or promoted.
That negative branch does not invalidate the measured Row681 v18b one-piece frontier.

## Current compute reachability

| Resource | Observed state | Classification |
|---|---|---|
| `tt-quietbox` | Normally reachable through `glmkvmigor`; currently tailnet-offline, last seen `2026-07-15T15:59:50.1Z`; 8 Wormhole chips; four Tenstorrent character devices; `tt-fold` owns them outside guarded GMG windows | Physical private accelerator, current site reachability down |
| `ilnur` / `research3` | Reachable; VMware guest; 40 CPUs; 396,231,820 KiB RAM; no NVIDIA GPU reported | Private virtual compute |
| `research_740xd` / `queens` | SSH port timed out | Private compute, current state unknown |
| `desktop-ivlvav4` | SSH port timed out; repository history identifies an RTX 3090 workstation | Private GPU workstation, currently unreachable |
| `glmkvmigor` | Normally reachable with the dedicated SSH key; currently tailnet-offline, last seen `2026-07-15T15:59:38.1Z`; RM10/Buildroot KVM and jump appliance | Private out-of-band management appliance, current site reachability down |
| `glkvm4090` | Endpoint reachable but key authentication denied; no password attempt made in this audit | Private KVM endpoint, not compute authority |
| `nas642` | SSH alias exists, but its hostname does not currently resolve | Private storage alias, current state unknown |
| Historical TT galaxies | Documented in `GATHER_DESIGN.md`, but no directly resolvable local SSH alias exists now | Shared/private accelerator history, not current authority |
| AWS `default` profile | Signed STS identity succeeds; account fingerprint `f0d6fd850853`; billing/global inventory calls denied | Public-cloud account exists, inventory/spend unknown |

## Public-cloud conclusion

The current shell has no installed `aws`, `gcloud`, `az`, `doctl`, `flyctl`, or `terraform` client, but
`~/.aws/credentials` contains a live `default` profile. A temporary `boto3` install under `/tmp` made a signed STS
request without printing credential values. Identity succeeded and established an IAM-user principal; only the
non-reversible account fingerprint `f0d6fd850853` was recorded here. Cost Explorer, Budgets, EC2 region discovery,
S3 account listing, IAM metadata/policy/simulation, Route53, CloudFront, and Lightsail were all explicitly denied.
No repository configuration names an AWS resource. Therefore the correct state is: **AWS account/profile exists;
inventory and spend are unknown because this principal lacks discovery authority.** No public-cloud mutation was
attempted. GCP/Azure and other public-cloud inventory remain unobserved.

As a local security hardening only, `~/.aws` was changed from mode `0755` to `0700` and its credentials file from
`0644` to `0600`. Credential contents and cloud policy were not changed.

The RM10 does run GL.iNet management-relay processes (`gl-cloud`, `gl-pion`, and `rtty`) in addition to Tailscale.
That is a vendor management-cloud path, not public compute capacity, and it does not establish any AWS/GCP/Azure
inventory. [GL.iNet's current RM10 documentation](https://www.gl-inet.com/products/gl-rm10/) identifies
`glkvm.com` as the browser cloud-control portal, and
that public login endpoint is reachable. Device binding/online status could not be inspected because there is no
active pre-authenticated browser session; no credential entry, device bind, KVM input, or power action was tried.
The solver traffic observed here used the private Tailscale/LAN path.

## Operational conclusion

The active TT-GMG hardware path is the private quietbox plus BMC cold-cycle recovery. `glmkvmigor` can show the
console and route SSH, but its optional ATX controller is not attached (`/dev/ttyACM0` absent), so it is not the
power-control authority. The in-band BMC/IPMI cycle remains the known recovery path; one KVM frame confirmed a
normal ASRock Rack-to-Ubuntu boot after the cycle. The KVM/BMC plane and the GMG solver are separate concepts.
No public cloud appears in the solver's current execution path; the proved AWS identity does not establish any
solver resource. The repo’s “cloud” wording principally refers to the sparse lattice-cloud model.

## Observed recovery, Run66/Run67, and site outage — 2026-07-15

- The KVM's LAN view resolved the QuietBox at `10.0.0.91`. Before using the LAN path with the existing Tailnet host
  alias, the QuietBox ED25519 host-key fingerprint was verified to match the already-known key exactly. A temporary,
  read-only HDMI stream captured the normal Ubuntu login screen during boot; no KVM input or power-control action was
  needed.
- Run66 supersedes Run52 as the qualifying G3 frontier. Its complete canonical-base plus exact dense-fallback
  fixed-vector apply is correct on all eight chips, with samples `1.945499/1.822259/1.858679 ms`, median
  `1.858679 ms`, L2 relative error `9.314343407e-7`, and no nonfinite output. It closes G3. The same run measured
  `547.659 ms` for a conservative `943,718,400`-byte upload, so the durable gate report remains `2/8` green
  (G3/G5) with G2 red until Run67 evidence is recovered.
- Run67 exercised the exact changing-vector page reader. Plan/cardinality preflight passed; upload measured
  `282.974 ms` for `411,188,224` bytes and program build measured `7.103 ms`. Warmup correctness was red with
  `50,858` nonfinite outputs, so Run67 is not G3 evidence. The root cause is an on-device circular-buffer cycle
  violation, not the exact host mapping: variable stage-page advances crossed a 62-page CB boundary and compact
  mode-3 low-split advances crossed a three-page CB boundary. The local fix reserves each complete stage
  allocation once as fixed-address L1 scratch, never pushes that scratch CB, and uses complete three-page
  low-split cycles; `29/29` local tests pass. Remote recompile and fresh-boot validation remain pending.
- Run67 consumed boot `c94ec4dd-...` and may not be reused. At workload exit the hardened runner itself verified
  `tt-fold` active, API ready, three online workers, three expected device holders, and an empty jobs list.
- Roughly two minutes later both `tt-quietbox` and `glmkvmigor` went tailnet-offline within a 12-second window.
  Their last-seen timestamps are `15:59:50.1Z` and `15:59:38.1Z`, respectively. The Mac Tailscale backend remains
  healthy and other peers remain online. Because the QuietBox and its independent RM10 management appliance
  disappeared together, the best current diagnosis is a shared remote-site power/WAN/LAN/Tailscale-path outage,
  not a TT process, a single-host SSH problem, or QuietBox key expiry. There is no alternate reachable router for
  the `10.0.0.0/24` LAN.
- An independent connection probe from the reachable `ilnur` private host also found all four SSH endpoints
  unreachable: QuietBox and RM10 over both their Tailnet addresses and their `10.0.0.0/24` LAN addresses. Direct
  TCP probes to the BMC's LAN address on ports 22, 80, and 443 were also unreachable. This rules out a Mac-only
  routing/backend failure and did not expose an alternate private ingress or BMC path.
- No power action is safe or useful while both management paths are unreachable. After they return, first recover
  and hash the original Run67/BMC logs, then perform read-only service/global-jobs/D-state/holder checks. A new BMC
  cycle is permitted only after the full strict preflight and after the fixed host/offline role builds pass.
- The public-cloud audit proved a live AWS IAM-user profile but no resource or billing authority. All measured solver
  compute and recovery activity remained on the private QuietBox/KVM/BMC plane.

## Security and confidence limits

- No supplied password was required, stored, echoed into a command, or added to a project artifact. QuietBox SSH,
  `sudo -n`, and the RM10 key path work without it.
- Successful AWS STS identity proves an account/profile exists. It does **not** prove any particular resource or bill
  exists; inventory and spend remain unknown because the principal is denied the relevant read calls.
- `glkvm4090` answering the network but rejecting the available key proves only that the endpoint exists; it does not
  authorize a password attempt or establish the state of the attached compute host.
- A BMC cold cycle is a disruptive recovery action, not routine management. It remains gated on empty portal jobs,
  no foreign workload, no D-state process, no device holder, and restoration of `tt-fold` after the guarded run.
- The simultaneous QuietBox/KVM outage limits diagnosis to the shared site path. Do not infer chassis power state,
  attempt blind recovery, or claim the site restored until one management path returns and read-only checks pass.
