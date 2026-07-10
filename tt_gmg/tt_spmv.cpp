// tt_spmv.cpp — persistent Metalium fine-SpMV as the g_tt_fine_spmv PCG callback (G4/cold/warm/stretch unlock).
//
// Wires the DEVICE-PROVEN fast SpMV (spmv_mac's on-device gather+MAC: 3.46 ms/8-chip @ 1089 GB/s, gather
// correctness rel_err 1e-6 on real TT, Test 1 in GATHER_DESIGN) into the fp64 outer PCG, REPLACING gmg_tt.py's
// slow numpy-gather + ttnn matmul-diagonal (10.41 s/apply). This is GATHER_DESIGN option (A): integrate the
// fast MAC + on-device gather now (interleaved layout, device-proven), so G4(<=1s)/cold/warm/stretch become
// measurable; the de-interleaved kernel (gather_reader_deint.cpp, host-proven bit-exact) is the throughput
// follow-on that swaps in for <3ms without changing this integration.
//
// Lifecycle:  tt_spmv_init("/tmp/row236_real_op.bin")  -> opens MeshDevice, uploads `a` RESIDENT, builds the
//   gather+MAC program ONCE (x/nbr replicated, per-core windows).  tt_spmv(const double* x, double* y) per PCG
//   apply -> split x to bf16x3, EnqueueWrite the x planes, EnqueueMeshWorkload, EnqueueRead c -> y (unscaled).
//   Driver (tiny python or ccx hook) sets g_tt_fine_spmv = &tt_spmv, g_tt_fine_n = n, then calls
//   ccx_gmg_solve_from_dump (eig-defl+hybrid) -> rc=0 maxU=95.8129714, and captures per-apply(G3)/per-solve(G4).
//
// STATUS: PENDING device compile/validation (no tt-metal headers off-box). The tt-metal API calls mirror the
// PROVEN spmv_mac.cpp one-shot path 1:1 (MakeBuf/MakeReplBuf/MakeCB/CreateKernel/SetRuntimeArgs/Enqueue*), just
// factored into init()+apply(); the per-core window math is the same validated code. Build as a .so on-box:
//   place under tt-metal-073/tt_metal/programming_examples/spmv_mac/ next to spmv_mac.cpp, add a
//   metal_example_tt_spmv shared-lib target, build, then ctypes-load it from a gmg_tt-style driver.
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>
#include <thread>
#include <algorithm>

using namespace tt;
using namespace tt::tt_metal;

// --- buffer helpers (identical semantics to spmv_mac.cpp) ---
static std::shared_ptr<distributed::MeshBuffer> MakeBuf(
    const std::shared_ptr<distributed::MeshDevice>& dev, uint32_t global_tiles, uint32_t nshards, uint32_t ebytes = sizeof(bfloat16)) {
    constexpr uint32_t H = tt::constants::TILE_HEIGHT, W = tt::constants::TILE_WIDTH;
    uint32_t ts = ebytes * H * W, shard_tiles = global_tiles / nshards;
    distributed::DeviceLocalBufferConfig lc{.page_size = ts, .buffer_type = BufferType::DRAM};
    distributed::ShardedBufferConfig bc{.global_size = (uint64_t)ts * global_tiles,
        .global_buffer_shape = {H, global_tiles * W}, .shard_shape = {H, shard_tiles * W},
        .shard_orientation = ShardOrientation::ROW_MAJOR};
    return distributed::MeshBuffer::create(bc, lc, dev.get());
}
static std::shared_ptr<distributed::MeshBuffer> MakeReplBuf(
    const std::shared_ptr<distributed::MeshDevice>& dev, uint32_t global_tiles, uint32_t ebytes = sizeof(bfloat16)) {
    constexpr uint32_t H = tt::constants::TILE_HEIGHT, W = tt::constants::TILE_WIDTH;
    uint32_t ts = ebytes * H * W;
    distributed::DeviceLocalBufferConfig lc{.page_size = ts, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig bc{.size = (uint64_t)ts * global_tiles};
    return distributed::MeshBuffer::create(bc, lc, dev.get());
}
static void MakeCB(Program& p, const CoreRangeSet& cores, tt::CBIndex cb, uint32_t n_tiles, tt::DataFormat fmt = tt::DataFormat::Float16_b) {
    uint32_t eb = (fmt == tt::DataFormat::Float32) ? 4u : 2u, ts = eb * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    CreateCircularBuffer(p, cores, CircularBufferConfig(n_tiles * ts, {{cb, fmt}}).set_page_size(cb, ts));
}
static inline void split3(float v, bfloat16& h, bfloat16& m, bfloat16& l) {
    h = bfloat16(v); float r = v - (float)h; m = bfloat16(r); l = bfloat16(r - (float)m);
}

// --- persistent context: device + resident `a` + compiled gather+MAC program + reusable x/nbr/c buffers ---
struct TtSpmvCtx {
    std::shared_ptr<distributed::MeshDevice> dev;
    distributed::MeshWorkload wl;
    std::shared_ptr<distributed::MeshBuffer> ah, am, al, xh, xm, xl, nbrbuf, c;
    std::shared_ptr<distributed::MeshBuffer> a_mm, idmask;    // matmul-diagonal: a_mm[grp][kt] [elem,k] tiles + identity
    uint32_t n_grp = 0, KT = 0;                               // matmul-diagonal: 32-elem groups, k-tiles (ceil(81/32)=3)
    uint32_t n = 0, n_out = 0, n_out_pad = 0, K = 0, NCHIP = 8, tile_elems = 1024;
    float VSCALE = 1e6f;
    std::vector<bfloat16> xhd, xmd, xld;                 // scratch bf16x3 x planes (row-major, per apply)
    std::vector<int32_t> nmin_v, nmax_v; uint32_t NBn_v = 0;  // sharding inputs, kept for per-apply workload rebuild
};
// --- matmul-diagonal (fp32-accumulate) support ---------------------------------------------------------------
// build_a_mm: transpose the single-bf16 DIA operator to a_mm[grp][kt] = a 32x32 tile with a_mm[e][k]=cf[k][g*32+e].
// NOTE tt-metal tiles are stored as 4x(16x16) faces; the host must emit in that face order (tilize) OR the device
// tilizes on read. Draft below writes ROW-MAJOR [e*32+k]; device build must tilize (or use a tilize kernel) - one of
// the device-iteration items. cf = the DIA op adf[tile][k][elem_in_tile] (tile-major, 1024 elems/tile).
static std::vector<bfloat16> build_a_mm(const std::vector<float>& adf, uint32_t n_out, uint32_t K, uint32_t n_pad,
                                        uint32_t& n_grp, uint32_t& KT) {
    n_grp = n_pad / 32; KT = (K + 31) / 32;
    std::vector<bfloat16> a_mm((size_t)n_grp * KT * 1024, bfloat16(0.f));
    for (uint32_t g = 0; g < n_grp; g++) for (uint32_t kt = 0; kt < KT; kt++) {
        bfloat16* T = &a_mm[((size_t)g*KT + kt)*1024];
        for (uint32_t e = 0; e < 32; e++) { uint32_t ge = g*32 + e, tile = ge/1024, ein = ge%1024;
            for (uint32_t kk = 0; kk < 32; kk++) { uint32_t k = kt*32 + kk;
                float v = (k < K && tile < n_out) ? adf[((size_t)tile*K + k)*1024 + ein] : 0.f;
                T[e*32 + kk] = bfloat16(v); } } }   // ROW-MAJOR draft (needs tilize on device)
    return a_mm;
}
// identity mask tile (1.0 on the 32x32 diagonal) for the diag(A@B) extraction (mask + row-reduce).
static std::vector<bfloat16> build_identity() {
    std::vector<bfloat16> id(1024, bfloat16(0.f));
    for (uint32_t d = 0; d < 32; d++) id[d*32 + d] = bfloat16(1.f);   // ROW-MAJOR draft (needs tilize)
    return id;
}
// Build a FRESH program+workload from the resident buffers. Some tt-metal builds accumulate state when the SAME
// MeshWorkload is re-enqueued across calls (gather corrupts deterministically after ~4 applies). Rebuilding per
// apply (CreateKernel hits the JIT cache, so it's cheap) sidesteps that. Same body as the original init build.
static void tt_build_wl(TtSpmvCtx* K);
static TtSpmvCtx* g_ctx = nullptr;

extern "C" int tt_spmv_init(const char* real_op_path, const char* nbr_path, const char* x_path) {
    auto* K = new TtSpmvCtx();
    const uint32_t TE = K->tile_elems;
    FILE* fp = fopen(real_op_path, "rb"); if (!fp) { fprintf(stderr, "tt_spmv_init: open %s failed\n", real_op_path); return 1; }
    int64_t hdr[2]; if (fread(hdr, 8, 2, fp) != 2) { fclose(fp); return 2; }
    K->n_out = (uint32_t)hdr[0]; K->K = (uint32_t)hdr[1];
    const char* nce = getenv("SPMV_NCHIP"); K->NCHIP = nce ? (uint32_t)atoi(nce) : 8;
    K->n_out_pad = ((K->n_out + K->NCHIP - 1) / K->NCHIP) * K->NCHIP;
    const size_t nA = (size_t)K->n_out * K->K * TE, nA_pad = (size_t)K->n_out_pad * K->K * TE;
    std::vector<float> adf(nA_pad, 0.f); if (fread(adf.data(), 4, nA, fp) != nA) { fclose(fp); return 3; } fclose(fp);
    std::vector<bfloat16> ahd(nA_pad), amd(nA_pad), ald(nA_pad);
    for (size_t i = 0; i < nA_pad; i++) split3(adf[i], ahd[i], amd[i], ald[i]);
    K->n = 3 * ((uint32_t)0);                             // set below from nbr

    // nbr (int32[27,NBn]) + per-node source-element window bounds (element units, interleaved x[3*nbr+c])
    FILE* fn = fopen(nbr_path, "rb"); if (!fn) return 4; int64_t nh[2]; if (fread(nh, 8, 2, fn) != 2) { fclose(fn); return 5; }
    const uint32_t NBn = (uint32_t)nh[1]; std::vector<int32_t> nbrv((size_t)27 * NBn);
    if (fread(nbrv.data(), 4, (size_t)27 * NBn, fn) != (size_t)27 * NBn) { fclose(fn); return 6; } fclose(fn);
    K->n = 3 * NBn;                                       // padded element count matches x buffer
    const uint32_t n_pad_elems = K->n_out_pad * TE;
    std::vector<int32_t> nmin(NBn, INT32_MAX), nmax(NBn, -1);
    for (uint32_t o = 0; o < 27; o++) { const int32_t* row = &nbrv[(size_t)o * NBn];
        for (uint32_t nd = 0; nd < NBn; nd++) { int32_t nn = row[nd];
            if (nn >= 0) { if (3*nn < nmin[nd]) nmin[nd] = 3*nn; if (3*nn+2 > nmax[nd]) nmax[nd] = 3*nn+2; } } }

    K->dev = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(1, K->NCHIP)));
    auto& cq = K->dev->mesh_command_queue();
    Program program = CreateProgram();
    // a REPLICATED (full on each chip, ~1.25GB/chip fits Blackhole DRAM): each chip reads its output tiles' a
    // by GLOBAL page from the local full copy -> no shard-local-accessor dependency. c stays SHARDED (local write).
    // a SHARDED (proven spmv_mac architecture): chip i owns a-tiles [i*n_local*K..]; reader reads a by LOCAL page t*K
    // -> the shard accessor maps to the chip's slice. (My earlier REPLICATED-a global-page read never populated across
    // the reap mesh -> a=0 for 62% of tiles = the 85-at-123i bug. Sharded a distributes correctly by construction.)
    K->ah = MakeBuf(K->dev, K->n_out_pad * K->K, K->NCHIP); K->am = MakeBuf(K->dev, K->n_out_pad * K->K, K->NCHIP); K->al = MakeBuf(K->dev, K->n_out_pad * K->K, K->NCHIP);
    K->c  = MakeBuf(K->dev, K->n_out_pad, K->NCHIP, 4);   // c MUST be sharded (tt-metal: multi-mesh read requires SHARDED)
    K->xh = MakeReplBuf(K->dev, K->n_out_pad); K->xm = MakeReplBuf(K->dev, K->n_out_pad); K->xl = MakeReplBuf(K->dev, K->n_out_pad);
    uint32_t nbr_tiles_total = ((uint32_t)NBn * 27 + 1023) / 1024;
    K->nbrbuf = MakeReplBuf(K->dev, nbr_tiles_total, 4);
    // upload resident `a` PER-SHARD (EnqueueWriteMeshBuffer's auto-distribute to the sharded buffer populated only
    // shard-0-head + shard-7-tail = edges-only gather; WriteShard places each chip's a-slice on its coord explicitly).
    { const uint32_t nl0 = K->n_out_pad/K->NCHIP, cc0 = (K->NCHIP==32)?4u:K->NCHIP; const size_t se = (size_t)nl0*K->K*TE;
      for (uint32_t j = 0; j < K->NCHIP; j++) { distributed::MeshCoordinate co(j/cc0, j%cc0);
        std::vector<bfloat16> sh(ahd.begin()+(size_t)j*se, ahd.begin()+(size_t)(j+1)*se); distributed::WriteShard(cq, K->ah, sh, co, true);
        std::vector<bfloat16> sm(amd.begin()+(size_t)j*se, amd.begin()+(size_t)(j+1)*se); distributed::WriteShard(cq, K->am, sm, co, true);
        std::vector<bfloat16> sl(ald.begin()+(size_t)j*se, ald.begin()+(size_t)(j+1)*se); distributed::WriteShard(cq, K->al, sl, co, true); } }
    std::vector<int32_t> nbr2((size_t)nbr_tiles_total * 1024, -1);
    for (uint32_t nd = 0; nd < NBn; nd++) for (uint32_t o = 0; o < 27; o++) nbr2[(size_t)nd * 27 + o] = nbrv[(size_t)o * NBn + nd];
    distributed::EnqueueWriteMeshBuffer(cq, K->nbrbuf, nbr2, true);
    K->xhd.assign(n_pad_elems, bfloat16(0.f)); K->xmd.assign(n_pad_elems, bfloat16(0.f)); K->xld.assign(n_pad_elems, bfloat16(0.f));

    K->nmin_v = std::move(nmin); K->nmax_v = std::move(nmax); K->NBn_v = NBn;  // keep sharding inputs for rebuild
    (void)program;                                                              // build moved into tt_build_wl
    tt_build_wl(K);                                                             // build the initial workload
    g_ctx = K;
    fprintf(stderr, "tt_spmv_init: OK n=%u n_out=%u(pad %u) K=%u NCHIP=%u\n", K->n, K->n_out, K->n_out_pad, K->K, K->NCHIP);
    return 0;
}

// Rebuild the program+workload fresh from the resident buffers (CreateKernel hits the JIT cache -> cheap).
// FIX: a and c are SHARDED (chip i owns global output tiles [i*n_local..], addressed by LOCAL `start` via the
// shard), but nbr and x are REPLICATED (full on every chip). So the nbr/x node indices (pc_nlo/pc_xlo) must be
// GLOBAL = chip*n_local offset + local. The old code used one shared program with LOCAL pc_nlo for all chips, so
// only chip 0 (offset 0) was correct and chips 1..N-1 gathered chip-0's window against their own a-shard ->
// garbage. Build a per-chip program with tile_base = chip*n_local and add it to that chip's mesh coordinate.
static void tt_build_wl(TtSpmvCtx* K) {
    const uint32_t NBn = K->NBn_v; const std::vector<int32_t>& nmin = K->nmin_v; const std::vector<int32_t>& nmax = K->nmax_v;
    auto grid = K->dev->compute_with_storage_grid_size();
    fprintf(stderr, "GRID compute_with_storage_grid_size=%ux%u=%u cores (if << 119, physical grid is tiny -> only ~few cores/chip run)\n",
            (uint32_t)grid.x, (uint32_t)grid.y, (uint32_t)(grid.x * grid.y));
    uint32_t n_local = K->n_out_pad / K->NCHIP;
    uint32_t cols = (K->NCHIP == 32) ? 4u : K->NCHIP;     // mesh columns: reap galaxy is Mesh(8,4) -> cols=4 (per WORKLOG)
    K->wl = distributed::MeshWorkload();
    for (uint32_t chip = 0; chip < K->NCHIP; chip++) {
    Program program = CreateProgram();
    const uint32_t tile_base = chip * n_local;            // GLOBAL output-tile base for this chip's shard
    CoreRangeSet all_set(CoreRange({0,0},{grid.x-1,grid.y-1}));
    MakeCB(program, all_set, tt::CBIndex::c_0, 3); MakeCB(program, all_set, tt::CBIndex::c_1, 3);
    MakeCB(program, all_set, tt::CBIndex::c_16, 8, tt::DataFormat::Float32);
    // NON-REDUNDANT (proven spmv_mac shape, generalized to tile_base!=0): split this chip's n_local tiles across
    // cores; each core does npc tiles with ONE staged x-window (union over its tiles). GLOBAL g_start=tile_base+
    // start drives node0/nbr/x (replicated full); LOCAL start drives the SHARDED a-page + sharded c. Per-core args.
    auto [ncores, cores, grpA, grpB, n1, n2] = tt::tt_metal::split_work_to_cores(all_set, n_local, true);
    std::vector<uint32_t> pc_xlo, pc_xnt, pc_nlo, pc_nn; uint32_t max_xnt = 1, max_npg = 1, cs_ = 0;
    for (auto pr_ : {std::make_pair(grpA, n1), std::make_pair(grpB, n2)}) {
        for (const auto& cr : pr_.first.ranges()) for (const auto& cc : cr) { const uint32_t npc = pr_.second;
            uint32_t g0t = tile_base + cs_, g1t = g0t + npc;              // this core's GLOBAL tiles [g0t, g1t)
            uint32_t nlo = (g0t*1024)/3; if (nlo >= NBn) nlo = (NBn>0)?NBn-1:0;
            uint32_t nhi = (g1t*1024+1022)/3; if (nhi >= NBn) nhi = NBn-1; if (nhi < nlo) nhi = nlo;
            uint32_t nn = nhi - nlo + 1;
            int32_t emin = INT32_MAX, emax = -1;
            for (uint32_t nd = nlo; nd <= nhi; nd++) { if (nmin[nd] < emin) emin = nmin[nd]; if (nmax[nd] > emax) emax = nmax[nd]; }
            if (emax < 0) { emin = 0; emax = 0; }
            uint32_t xlo = (uint32_t)emin/1024, xnt = (uint32_t)emax/1024 - xlo + 1;
            uint32_t sil = nlo*27, plo = sil/1024, npg = (sil + nn*27 + 1023)/1024 - plo;
            pc_xlo.push_back(xlo); pc_xnt.push_back(xnt); pc_nlo.push_back(nlo); pc_nn.push_back(nn);
            if (xnt > max_xnt) max_xnt = xnt; if (npg > max_npg) max_npg = npg; cs_ += npc;
        }
    }
    MakeCB(program, all_set, tt::CBIndex::c_2, max_xnt+2); MakeCB(program, all_set, tt::CBIndex::c_3, max_xnt+2);
    MakeCB(program, all_set, tt::CBIndex::c_4, max_xnt+2); MakeCB(program, all_set, tt::CBIndex::c_5, max_npg+2, tt::DataFormat::Float32);
    std::vector<uint32_t> r_ct = {(uint32_t)tt::CBIndex::c_0,(uint32_t)tt::CBIndex::c_1,(uint32_t)tt::CBIndex::c_2,(uint32_t)tt::CBIndex::c_3,(uint32_t)tt::CBIndex::c_4,(uint32_t)tt::CBIndex::c_5};
    TensorAccessorArgs(*K->ah).append_to(r_ct); TensorAccessorArgs(*K->am).append_to(r_ct); TensorAccessorArgs(*K->al).append_to(r_ct);
    TensorAccessorArgs(*K->xh).append_to(r_ct); TensorAccessorArgs(*K->xm).append_to(r_ct); TensorAccessorArgs(*K->xl).append_to(r_ct);
    TensorAccessorArgs(*K->nbrbuf).append_to(r_ct);
    std::vector<uint32_t> w_ct = {(uint32_t)tt::CBIndex::c_16}; TensorAccessorArgs(*K->c).append_to(w_ct);
    auto reader = CreateKernel(program, "/tmp/kernels/gather_reader.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = r_ct});
    auto writer = CreateKernel(program, "/tmp/kernels/mac_writer.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = w_ct});
    auto compute = CreateKernel(program, "/tmp/kernels/mac_compute.cpp", cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false, .compile_args = {}});
    // per-core runtime args: reader gets GLOBAL g_start (node0/nbr/x) + per-core window + LOCAL start (a-page); the
    // writer gets LOCAL start (sharded c). start is the cumulative LOCAL tile offset within this chip's shard.
    uint32_t start = 0, ci = 0;
    for (auto pr_ : {std::make_pair(grpA, n1), std::make_pair(grpB, n2)}) {
        for (const auto& cr : pr_.first.ranges()) for (const auto& cc : cr) { const uint32_t npc = pr_.second;
            const uint32_t g_start = tile_base + start;
            SetRuntimeArgs(program, reader, cc, {(uint32_t)K->ah->address(),(uint32_t)K->am->address(),(uint32_t)K->al->address(),
                (uint32_t)K->xh->address(),(uint32_t)K->xm->address(),(uint32_t)K->xl->address(),(uint32_t)K->nbrbuf->address(),
                npc, K->K, g_start, pc_xlo[ci], pc_xnt[ci], pc_nlo[ci], pc_nn[ci], start});
            SetRuntimeArgs(program, compute, cc, {npc, K->K});
            SetRuntimeArgs(program, writer, cc, {(uint32_t)K->c->address(), npc, start});
            start += npc; ci++;
        }
    }
    if (chip == 0) fprintf(stderr, "tt_build_wl NONREDUNDANT: n_local=%u NCHIP=%u cols=%u ncores=%u max_xnt=%u max_npg=%u grid=%u (per-core g_start+local a-page)\n",
            n_local, K->NCHIP, cols, (uint32_t)ncores, max_xnt, max_npg, (uint32_t)(grid.x*grid.y));
    distributed::MeshCoordinate coord(chip / cols, chip % cols);           // this chip's mesh position
    K->wl.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }
}

// one PCG fine-SpMV: y = A x. x/y are the fp64 solver vectors (length n). On-device gather+MAC; a resident.
extern "C" int tt_spmv(const double* x, double* y) {
    TtSpmvCtx* K = g_ctx; if (!K) return 1;
    const uint32_t n = K->n, TE = K->tile_elems, n_pad_elems = K->n_out_pad * TE;
    // ADAPTIVE per-call scale: the fixed VSCALE=1e6 assumes a fixed |x| range, but the PCG's vectors (smoother
    // input, residual, A.p) vary in magnitude by orders of magnitude across calls. A fixed scale drives x into a
    // bad bf16x3/compensated-MAC precision regime for some magnitudes -> deterministic wrong SpMV after the first
    // few (small) calls. Rescale each call so max|x| maps to a fixed sweet-spot target; unscale y by the same.
    double xmax_ = 0.0; for (uint32_t i = 0; i < n; i++) { double a_ = std::fabs(x[i]); if (a_ > xmax_) xmax_ = a_; }
    const float vscale = (xmax_ > 0.0) ? (float)(1.0e3 / xmax_) : K->VSCALE;   // scaled max|x| ~ 1e3 (bf16x3 sweet spot)
    { const uint32_t NT = 16; std::vector<std::thread> ths_; const uint32_t ch_ = (n_pad_elems + NT - 1) / NT;
      for (uint32_t t_ = 0; t_ < NT; t_++) ths_.emplace_back([&, t_]() {
        const uint32_t lo_ = t_ * ch_, hi_ = std::min((t_ + 1) * ch_, n_pad_elems);
        for (uint32_t i = lo_; i < hi_; i++) { float v = (i < n) ? (float)x[i] * vscale : 0.f; split3(v, K->xhd[i], K->xmd[i], K->xld[i]); } });
      for (auto& th_ : ths_) th_.join(); }
    if (!getenv("SPMV_REUSE_WORKLOAD")) tt_build_wl(K);             // REBUILD per apply by default: reusing the same
                                                                     // MeshWorkload corrupts the gather after a few applies
                                                                     // (LAST-APPLY rel_err 1.0). Correctness > speed until the
                                                                     // resident-vector path (no host round-trip) lands for G3/G4.
    auto& cq = K->dev->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, K->xh, K->xhd, true);
    distributed::EnqueueWriteMeshBuffer(cq, K->xm, K->xmd, true);
    distributed::EnqueueWriteMeshBuffer(cq, K->xl, K->xld, true);
    distributed::Finish(cq);                                         // SYNC: guarantee x is fully propagated to ALL 32 chips
    distributed::EnqueueMeshWorkload(cq, K->wl, true);               // over the fabric BEFORE the gather workload reads it.
    distributed::Finish(cq);                                        // BLOCKING workload + Finish: workload fully done on all
    // PER-SHARD readback: EnqueueReadMeshBuffer's auto-assembly of the sharded c returned only shard-0-head +
    // shard-7-tail (edges-only 25/3781, middle chips missing). Read each chip's shard EXPLICITLY from its mesh
    // coord and place it at its GLOBAL tile offset -> correct multi-chip assembly. cols = mesh columns.
    const uint32_t nloc_ = K->n_out_pad / K->NCHIP, cols_ = (K->NCHIP == 32) ? 4u : K->NCHIP;
    std::vector<float> cd((size_t)K->n_out_pad * TE, 0.f);
    for (uint32_t j = 0; j < K->NCHIP; j++) {                       // chips before the c read. Fixes the mesh-write/gather
        std::vector<float> shard_;                                 // race (degraded link lagged fabric propagation);
        distributed::ReadShard(cq, shard_, K->c, distributed::MeshCoordinate(j / cols_, j % cols_), true);
        const size_t cnt_ = std::min(shard_.size(), (size_t)nloc_ * TE);
        std::copy(shard_.begin(), shard_.begin() + cnt_, cd.begin() + (size_t)j * nloc_ * TE);
    }
    { static int once_=0; if(!once_){ once_=1; uint32_t nloc=K->n_out_pad/K->NCHIP;
        fprintf(stderr,"CD_SIZE cd.size()=%zu tiles=%zu n=%u n_out_pad=%u n_local=%u cd_tiles/NCHIP=%zu (drift if !=n_local)\n",
                cd.size(), cd.size()/1024, K->n, K->n_out_pad, nloc, (cd.size()/1024)/K->NCHIP); } }
                                                                    // -> stale x for tiles>=1024 -> deterministic-wrong SpMV).
    const double inv = 1.0 / (double)vscale;
    { const uint32_t NT = 16; const uint32_t cds_ = (uint32_t)cd.size(); std::vector<std::thread> thr_; const uint32_t ch_ = (n + NT - 1) / NT;
      for (uint32_t t_ = 0; t_ < NT; t_++) thr_.emplace_back([&, t_]() {
        const uint32_t lo_ = t_ * ch_, hi_ = std::min((t_ + 1) * ch_, n);
        for (uint32_t i = lo_; i < hi_; i++) y[i] = (i < cds_) ? (double)cd[i] * inv : 0.0; });
      for (auto& th_ : thr_) th_.join(); }
    return 0;
}

extern "C" void tt_spmv_close() { if (g_ctx) { g_ctx->dev->close(); delete g_ctx; g_ctx = nullptr; } }
