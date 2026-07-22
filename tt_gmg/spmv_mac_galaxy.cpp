// spmv_mac.cpp — Metalium host driver for the MAC-reduction SpMV core (Stage B, first draft).
// Computes out[t] = sum_{k=0..K-1} a[t*K+k] * b[t*K+k] over n_out output tiles, on all Tensix cores.
// Purpose: measure Metalium's achievable bandwidth on the DIA MAC reduction (the op ttnn did at 12 GB/s /
// 18 ms) => the G3 (<=3 ms) feasibility number. Synthetic data first (validates against a host reference);
// real row236 coeff/xg tiles wired next. K=81 = 27 offsets * 3 in-components.
//
// Build/deploy: this file + kernels/ are the source of record in the fork. To build on tt-quietbox, copy into
// tt-metal-073/tt_metal/programming_examples/spmv_mac/ (with a CMakeLists mirroring vecadd_multi_core, target
// metal_example_spmv_mac), add the subdir to the programming_examples CMakeLists, then:
//   cmake -S . -B build_Release -DBUILD_PROGRAMMING_EXAMPLES=ON && cmake --build build_Release --target metal_example_spmv_mac
//   ./build_Release/programming_examples/metal_example_spmv_mac
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

static std::shared_ptr<distributed::MeshBuffer> MakeBuf(
    const std::shared_ptr<distributed::MeshDevice>& dev, uint32_t global_tiles, uint32_t nshards,
    uint32_t ebytes = sizeof(bfloat16)) {
    constexpr uint32_t H = tt::constants::TILE_HEIGHT, W = tt::constants::TILE_WIDTH;
    uint32_t ts = ebytes * H * W;
    uint32_t shard_tiles = global_tiles / nshards;      // 1xN mesh -> shard the width contiguously
    distributed::DeviceLocalBufferConfig lc{.page_size = ts, .buffer_type = BufferType::DRAM};
    distributed::ShardedBufferConfig bc{
        .global_size = (uint64_t)ts * global_tiles,
        .global_buffer_shape = {H, global_tiles * W},
        .shard_shape = {H, shard_tiles * W},
        .shard_orientation = ShardOrientation::ROW_MAJOR};
    return distributed::MeshBuffer::create(bc, lc, dev.get());
}
// Replicated: every chip gets the FULL buffer (for the resident x + nbr, whose neighbors cross shard boundaries).
static std::shared_ptr<distributed::MeshBuffer> MakeReplBuf(
    const std::shared_ptr<distributed::MeshDevice>& dev, uint32_t global_tiles, uint32_t ebytes = sizeof(bfloat16)) {
    constexpr uint32_t H = tt::constants::TILE_HEIGHT, W = tt::constants::TILE_WIDTH;
    uint32_t ts = ebytes * H * W;
    distributed::DeviceLocalBufferConfig lc{.page_size = ts, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig bc{.size = (uint64_t)ts * global_tiles};
    return distributed::MeshBuffer::create(bc, lc, dev.get());
}
static void MakeCB(Program& p, const CoreRangeSet& cores, tt::CBIndex cb, uint32_t n_tiles,
                   tt::DataFormat fmt = tt::DataFormat::Float16_b) {
    uint32_t eb = (fmt == tt::DataFormat::Float32) ? 4u : 2u;
    uint32_t ts = eb * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    CircularBufferConfig cfg = CircularBufferConfig(n_tiles * ts, {{cb, fmt}}).set_page_size(cb, ts);
    CreateCircularBuffer(p, cores, cfg);
}

int main() {
    const uint32_t tile_elems = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    // Load the REAL row236 fine operator (tile-laid: coeff a_k, gathered-x b_k, reference y=Ax) from file.
    FILE* fp = fopen("/tmp/row236_real_op.bin", "rb");
    int64_t hdr[2]; (void)!fread(hdr, 8, 2, fp);
    const uint32_t n_out = (uint32_t)hdr[0], K = (uint32_t)hdr[1];
    const char* ncenv = getenv("SPMV_NCHIP");            // fabric may be flaky -> allow single-chip fallback
    const uint32_t NCHIP = ncenv ? (uint32_t)atoi(ncenv) : 8;
    // Pad output tiles to a multiple of NCHIP so the 1xNCHIP DRAM sharding divides evenly (n_out=3782 -> 3784).
    // The extra tiles have zero coeff/gathered-x, so their MAC is 0 and they don't perturb correctness.
    const uint32_t n_out_pad = ((n_out + NCHIP - 1) / NCHIP) * NCHIP;
    const size_t nA = (size_t)n_out * K * tile_elems, nY = (size_t)n_out * tile_elems;
    const size_t nA_pad = (size_t)n_out_pad * K * tile_elems, nY_pad = (size_t)n_out_pad * tile_elems;
    std::vector<float> adf(nA_pad, 0.f), bdf(nA_pad, 0.f), ref(nY_pad, 0.f);
    (void)!fread(adf.data(), 4, nA, fp); (void)!fread(bdf.data(), 4, nA, fp); (void)!fread(ref.data(), 4, nY, fp);
    fclose(fp);
    const float VSCALE = 1e6f;   // b (displacement) ~1e-6 -> scale to O(1) for the bf16 tile engine; y unscales by same
    for (auto& v : bdf) v *= VSCALE;
    for (auto& v : ref) v *= VSCALE;
    std::vector<bfloat16> ahd(nA_pad), amd(nA_pad), ald(nA_pad), bhd(nA_pad), bmd(nA_pad), bld(nA_pad);
    auto split3 = [](float v, bfloat16& h, bfloat16& m, bfloat16& l){
        h = bfloat16(v); float r = v - (float)h; m = bfloat16(r); l = bfloat16(r - (float)m); };
    for (size_t i = 0; i < nA_pad; i++) { split3(adf[i], ahd[i], amd[i], ald[i]); split3(bdf[i], bhd[i], bmd[i], bld[i]); }
    printf("loaded REAL row236 operator + bf16x3 split: n_out=%u (pad %u) K=%u\n", n_out, n_out_pad, K);

    // ---- SPMV_GATHER: on-device gather of b from a resident x via nbr (a-only DRAM). Single-chip first. ----
    const bool GATHER = getenv("SPMV_GATHER") != nullptr;
    const uint32_t n_pad_elems = n_out_pad * tile_elems;       // x length (padded to output-tile boundary)
    std::vector<bfloat16> xhd, xmd, xld; std::vector<int32_t> nbrv; uint32_t NBn = 0;
    std::vector<int32_t> nmin, nmax;                            // per-node min/max source element (3*nbr+c) over 27 offsets
    if (GATHER) {
        FILE* fx = fopen("/tmp/row236_x.bin", "rb"); int64_t xh2[2]; (void)!fread(xh2, 8, 2, fx);
        const uint32_t xn = (uint32_t)xh2[0]; std::vector<float> xf(n_pad_elems, 0.f);
        (void)!fread(xf.data(), 4, xn < n_pad_elems ? xn : n_pad_elems, fx); fclose(fx);
        for (auto& v : xf) v *= VSCALE;
        xhd.resize(n_pad_elems); xmd.resize(n_pad_elems); xld.resize(n_pad_elems);
        for (uint32_t i = 0; i < n_pad_elems; i++) split3(xf[i], xhd[i], xmd[i], xld[i]);
        FILE* fn = fopen("/tmp/row236_nbrpad.bin", "rb"); int64_t nh[2]; (void)!fread(nh, 8, 2, fn);
        NBn = (uint32_t)nh[1]; nbrv.resize((size_t)27 * NBn); (void)!fread(nbrv.data(), 4, (size_t)27 * NBn, fn); fclose(fn);
        nmin.assign(NBn, INT32_MAX); nmax.assign(NBn, -1);
        for (uint32_t o = 0; o < 27; o++) { const int32_t* row = &nbrv[(size_t)o * NBn];
            for (uint32_t node = 0; node < NBn; node++) { int32_t nn = row[node];
                if (nn >= 0) { int32_t lo = 3 * nn, hi = 3 * nn + 2;
                    if (lo < nmin[node]) nmin[node] = lo; if (hi > nmax[node]) nmax[node] = hi; } } }
        printf("SPMV_GATHER: x n_pad=%u, nbr 27x%u, per-node window bounds ready\n", n_pad_elems, NBn);
    }
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D); /*REFAB*/
    auto dev = distributed::MeshDevice::create(distributed::MeshDeviceConfig(NCHIP==32 ? distributed::MeshShape(4,8) : distributed::MeshShape(1, NCHIP))); /*FS2*/  // NCHIP chips, 1D (SPMV_NCHIP env)
    auto& cq = dev->mesh_command_queue();

    const char* ntenv = getenv("SPMV_GATHER_NTILES");   // subset of output tiles (small L1 windows for correctness test)
    const uint32_t n_local = (GATHER && ntenv) ? std::min((uint32_t)atoi(ntenv), n_out_pad) : n_out_pad / NCHIP;   // output tiles per chip
    auto ah = MakeBuf(dev, n_out_pad * K, NCHIP), am = MakeBuf(dev, n_out_pad * K, NCHIP), al = MakeBuf(dev, n_out_pad * K, NCHIP);
    auto bh = MakeBuf(dev, n_out_pad * K, NCHIP), bm = MakeBuf(dev, n_out_pad * K, NCHIP), bl = MakeBuf(dev, n_out_pad * K, NCHIP);
    auto c = MakeBuf(dev, n_out_pad, NCHIP, 4);   // fp32 output (full bf16x3 accuracy, no bf16 truncation)
    // gather buffers: x row-major bf16 (n_out_pad tiles of 1024) + nbr node-major int32 [NBn*27]
    std::shared_ptr<distributed::MeshBuffer> xh, xm, xl, nbrbuf; uint32_t nbr_tiles_total = 0;
    if (GATHER) {
        // x + nbr REPLICATED on every chip so each core gathers locally (neighbors cross shard boundaries).
        xh = MakeReplBuf(dev, n_out_pad); xm = MakeReplBuf(dev, n_out_pad); xl = MakeReplBuf(dev, n_out_pad);
        nbr_tiles_total = ((uint32_t)NBn * 27 + 1023) / 1024;
        nbrbuf = MakeReplBuf(dev, nbr_tiles_total, 4);
    }

    auto tu0 = std::chrono::high_resolution_clock::now();
    distributed::EnqueueWriteMeshBuffer(cq, ah, ahd, true); distributed::EnqueueWriteMeshBuffer(cq, am, amd, true);
    distributed::EnqueueWriteMeshBuffer(cq, al, ald, true);
    if (!GATHER) {
        distributed::EnqueueWriteMeshBuffer(cq, bh, bhd, true);
        distributed::EnqueueWriteMeshBuffer(cq, bm, bmd, true); distributed::EnqueueWriteMeshBuffer(cq, bl, bld, true);
    } else {
        std::vector<int32_t> nbr2((size_t)nbr_tiles_total * 1024, -1);      // node-major, padded
        for (uint32_t node = 0; node < NBn; node++) for (uint32_t o = 0; o < 27; o++) nbr2[(size_t)node * 27 + o] = nbrv[(size_t)o * NBn + node];
        distributed::EnqueueWriteMeshBuffer(cq, xh, xhd, true); distributed::EnqueueWriteMeshBuffer(cq, xm, xmd, true);
        distributed::EnqueueWriteMeshBuffer(cq, xl, xld, true); distributed::EnqueueWriteMeshBuffer(cq, nbrbuf, nbr2, true);
    }
    double g2_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tu0).count();
    printf("DBG G2 matrix upload (%s) = %.1f ms\n", GATHER ? "a + x + nbr" : "6 bf16 streams", g2_ms);
    std::vector<float> zc(nY_pad, 0.f);
    distributed::EnqueueWriteMeshBuffer(cq, c, zc, true);

    distributed::MeshWorkload wl;
    for (const auto& _coord : distributed::MeshCoordinateRange(dev->shape())) {
    Program program = CreateProgram();
    auto grid = dev->compute_with_storage_grid_size();
    (void)grid; //WORKERFIX
    CoreRangeSet all_set = dev->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0});
    MakeCB(program, all_set, tt::CBIndex::c_0, 3);   // cb_a: [ah,am,al] per k (unique tiles, read once)
    MakeCB(program, all_set, tt::CBIndex::c_1, 3);   // cb_b: [bh,bm,bl] per k
    MakeCB(program, all_set, tt::CBIndex::c_16, 8, tt::DataFormat::Float32);

    auto [ncores, cores, g1, g2, n1, n2] = tt::tt_metal::split_work_to_cores(all_set, n_local, true);   // per-chip work

    // per-core gather windows (scan nmin/nmax over each core's node range), + size the x/nbr CBs to the max
    std::vector<uint32_t> pc_xlo, pc_xnt, pc_nlo, pc_nn;
    if (GATHER) {
        uint32_t s = 0, max_xnt = 1, max_npg = 1;
        for (auto [grp, npc] : {std::make_pair(g1, n1), std::make_pair(g2, n2)})
            for (const auto& cr : grp.ranges()) for (const auto& cc : cr) {
                (void)cc;
                uint32_t nlo = (s * 1024) / 3, nhi = ((s + npc) * 1024 - 1) / 3; if (nhi >= NBn) nhi = NBn - 1;
                int32_t emin = INT32_MAX, emax = -1;
                for (uint32_t nd = nlo; nd <= nhi; nd++) { if (nmin[nd] < emin) emin = nmin[nd]; if (nmax[nd] > emax) emax = nmax[nd]; }
                if (emax < 0) { emin = 0; emax = 0; }
                uint32_t tlo = (uint32_t)emin / 1024, tnt = (uint32_t)emax / 1024 - tlo + 1;
                pc_xlo.push_back(tlo); pc_xnt.push_back(tnt); pc_nlo.push_back(nlo); pc_nn.push_back(nhi - nlo + 1);
                if (tnt > max_xnt) max_xnt = tnt;
                uint32_t sil = nlo * 27, plo = sil / 1024, npg = (sil + (nhi - nlo + 1) * 27 + 1023) / 1024 - plo;
                if (npg > max_npg) max_npg = npg;
                s += npc;
            }
        MakeCB(program, all_set, tt::CBIndex::c_2, max_xnt); MakeCB(program, all_set, tt::CBIndex::c_3, max_xnt);
        MakeCB(program, all_set, tt::CBIndex::c_4, max_xnt); MakeCB(program, all_set, tt::CBIndex::c_5, max_npg, tt::DataFormat::Float32);
        printf("SPMV_GATHER: max x-window tiles=%u, max nbr pages=%u (per core)\n", max_xnt, max_npg);
    }

    std::vector<uint32_t> r_ct = {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_1};
    if (GATHER) { r_ct.push_back((uint32_t)tt::CBIndex::c_2); r_ct.push_back((uint32_t)tt::CBIndex::c_3);
                  r_ct.push_back((uint32_t)tt::CBIndex::c_4); r_ct.push_back((uint32_t)tt::CBIndex::c_5); }
    TensorAccessorArgs(*ah).append_to(r_ct); TensorAccessorArgs(*am).append_to(r_ct); TensorAccessorArgs(*al).append_to(r_ct);
    if (GATHER) { TensorAccessorArgs(*xh).append_to(r_ct); TensorAccessorArgs(*xm).append_to(r_ct);
                  TensorAccessorArgs(*xl).append_to(r_ct); TensorAccessorArgs(*nbrbuf).append_to(r_ct); }
    else { TensorAccessorArgs(*bh).append_to(r_ct); TensorAccessorArgs(*bm).append_to(r_ct); TensorAccessorArgs(*bl).append_to(r_ct); }
    std::vector<uint32_t> w_ct = {(uint32_t)tt::CBIndex::c_16};
    TensorAccessorArgs(*c).append_to(w_ct);

    auto reader = CreateKernel(program,
        GATHER ? "tt_metal/programming_examples/spmv_mac/kernels/gather_reader.cpp"
               : "tt_metal/programming_examples/spmv_mac/kernels/mac_reader.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = r_ct});
    auto writer = CreateKernel(program, "tt_metal/programming_examples/spmv_mac/kernels/mac_writer.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = w_ct});
    auto compute = CreateKernel(program, "tt_metal/programming_examples/spmv_mac/kernels/mac_compute.cpp", cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false, .compile_args = {}});

    uint32_t start = 0, ci = 0;
    for (auto [grp, npc] : {std::make_pair(g1, n1), std::make_pair(g2, n2)}) {
        for (const auto& cr : grp.ranges())
            for (const auto& cc : cr) {
                if (GATHER)
                    SetRuntimeArgs(program, reader, cc, {(uint32_t)ah->address(), (uint32_t)am->address(), (uint32_t)al->address(),
                        (uint32_t)xh->address(), (uint32_t)xm->address(), (uint32_t)xl->address(), (uint32_t)nbrbuf->address(),
                        npc, K, start, pc_xlo[ci], pc_xnt[ci], pc_nlo[ci], pc_nn[ci]});
                else
                    SetRuntimeArgs(program, reader, cc, {(uint32_t)ah->address(), (uint32_t)am->address(), (uint32_t)al->address(),
                        (uint32_t)bh->address(), (uint32_t)bm->address(), (uint32_t)bl->address(), npc, K, start});
                SetRuntimeArgs(program, compute, cc, {npc, K});
                SetRuntimeArgs(program, writer, cc, {(uint32_t)c->address(), npc, start});
                start += npc; ci++;
            }
    }
    wl.add_program(distributed::MeshCoordinateRange(_coord, _coord), std::move(program));
    }

    // warm + time
    distributed::EnqueueMeshWorkload(cq, wl, true);
    auto t0 = std::chrono::high_resolution_clock::now();
    const int reps = 20;
    for (int i = 0; i < reps; ++i) distributed::EnqueueMeshWorkload(cq, wl, false);
    Finish(cq);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count() / reps;

    std::vector<float> cd;
    auto to0 = std::chrono::high_resolution_clock::now();
    distributed::EnqueueReadMeshBuffer(cq, cd, c, true);     // G5: output read/un-permute (solution vector back to host)
    double g5_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - to0).count();
    printf("DBG G5 output read (%zu MB) = %.1f ms\n", (size_t)cd.size() * 2 / 1000000, g5_ms);
    printf("DBG cd.size=%zu ref.size=%zu  cd[0..2]=%.3f,%.3f,%.3f  ref[0..2]=%.3f,%.3f,%.3f\n",
           cd.size(), ref.size(), (float)cd[0], (float)cd[1], (float)cd[2], ref[0], ref[1], ref[2]);
    { std::vector<float> cdf(cd.size()); for (size_t i=0;i<cd.size();++i) cdf[i]=(float)cd[i];
      FILE* of=fopen("/tmp/tt_cd.bin","wb"); fwrite(cdf.data(),4,cdf.size(),of); fclose(of);
      // worst element + a few large-|ref| elements
      size_t am=0; double best=-1; for (size_t i=0;i<cd.size();++i){double e=std::abs((double)cdf[i]-ref[i]); if(std::isfinite(e)&&e>best){best=e;am=i;}}
      printf("DBG worst i=%zu cd=%.4e ref=%.4e | sample large-ref:", am, cdf[am], ref[am]);
      double s2=0; size_t bi=0; for(size_t i=0;i<cd.size();++i) if(std::abs(ref[i])>s2){s2=std::abs(ref[i]);bi=i;}
      printf(" i=%zu cd=%.4e ref=%.4e\n", bi, cdf[bi], ref[bi]); }
    size_t ninf = 0, first_bad = SIZE_MAX, last_bad = 0, bad_lo_half = 0, bad_hi_half = 0;
    std::vector<int> badpos(tile_elems, 0);          // per-position-within-tile bad count
    for (size_t i = 0; i < cd.size(); ++i) { float v=(float)cd[i];
        if (!std::isfinite(v)) { ninf++; if (i<first_bad) first_bad=i; last_bad=i;
            (i < cd.size()/2 ? bad_lo_half : bad_hi_half)++; badpos[i % tile_elems]++; } }
    int poswith = 0, posmax = 0; for (int p=0;p<(int)tile_elems;p++){ if(badpos[p]) poswith++; if(badpos[p]>posmax) posmax=badpos[p]; }
    printf("DBG non-finite: %zu/%zu  firstTile=%zu lastTile=%zu  loHalf=%zu hiHalf=%zu  positions-with-bad=%d/%zu (max %d)\n",
           ninf, cd.size(), first_bad/tile_elems, last_bad/tile_elems, bad_lo_half, bad_hi_half, poswith, (size_t)tile_elems, posmax);
    double maxrel = 0, scale = 0;
    size_t n = std::min(cd.size(), ref.size());
    if (GATHER) n = std::min(n, (size_t)n_local * tile_elems);   // only the processed subset of output tiles
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, (double)std::abs(ref[i]));
    for (size_t i = 0; i < n; ++i) { float v=(float)cd[i]; if (std::isfinite(v)) maxrel = std::max(maxrel, std::abs((double)v - ref[i])); }
    double gb = (double)n_out_pad * K * 6 /*6 bf16x3 streams*/ * tile_elems * sizeof(bfloat16) / 1e9;
    printf("SpMV-MAC(metalium): %.3f ms/apply  %.0f GB/s  rel_err=%.3e  (n_out=%u K=%u)\n",
           ms, gb / (ms / 1000.0), maxrel / (scale + 1e-30), n_out, K);
    return 0;
}
