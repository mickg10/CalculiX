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
#include <chrono>
#include <cstdint>
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
    const size_t nA = (size_t)n_out * K * tile_elems, nY = (size_t)n_out * tile_elems;
    std::vector<float> adf(nA), bdf(nA), ref(nY);
    (void)!fread(adf.data(), 4, nA, fp); (void)!fread(bdf.data(), 4, nA, fp); (void)!fread(ref.data(), 4, nY, fp);
    fclose(fp);
    const float VSCALE = 1e6f;   // b (displacement) ~1e-6 -> scale to O(1) for the bf16 tile engine; y unscales by same
    for (auto& v : bdf) v *= VSCALE;
    for (auto& v : ref) v *= VSCALE;
    std::vector<bfloat16> ahd(nA), amd(nA), ald(nA), bhd(nA), bmd(nA), bld(nA);
    auto split3 = [](float v, bfloat16& h, bfloat16& m, bfloat16& l){
        h = bfloat16(v); float r = v - (float)h; m = bfloat16(r); l = bfloat16(r - (float)m); };
    for (size_t i = 0; i < nA; i++) { split3(adf[i], ahd[i], amd[i], ald[i]); split3(bdf[i], bhd[i], bmd[i], bld[i]); }
    printf("loaded REAL row236 operator + bf16x3 split: n_out=%u K=%u\n", n_out, K);
    auto dev = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(1, 8)));  // 8 chips, 1D
    const uint32_t NCHIP = 8;
    Program program = CreateProgram();
    auto& cq = dev->mesh_command_queue();

    const uint32_t n_local = n_out / NCHIP;       // output tiles per chip
    auto ah = MakeBuf(dev, n_out * K, NCHIP), am = MakeBuf(dev, n_out * K, NCHIP), al = MakeBuf(dev, n_out * K, NCHIP);
    auto bh = MakeBuf(dev, n_out * K, NCHIP), bm = MakeBuf(dev, n_out * K, NCHIP), bl = MakeBuf(dev, n_out * K, NCHIP);
    auto c = MakeBuf(dev, n_out, NCHIP, 4);   // fp32 output (full bf16x3 accuracy, no bf16 truncation)

    auto tu0 = std::chrono::high_resolution_clock::now();
    distributed::EnqueueWriteMeshBuffer(cq, ah, ahd, true); distributed::EnqueueWriteMeshBuffer(cq, am, amd, true);
    distributed::EnqueueWriteMeshBuffer(cq, al, ald, true); distributed::EnqueueWriteMeshBuffer(cq, bh, bhd, true);
    distributed::EnqueueWriteMeshBuffer(cq, bm, bmd, true); distributed::EnqueueWriteMeshBuffer(cq, bl, bld, true);
    double g2_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tu0).count();
    printf("DBG G2 matrix upload (6 x %zu MB -> 8 chips) = %.1f ms\n", (size_t)ahd.size() * 2 / 1000000, g2_ms);
    std::vector<float> zc(nY, 0.f);
    distributed::EnqueueWriteMeshBuffer(cq, c, zc, true);

    auto grid = dev->compute_with_storage_grid_size();
    auto all = CoreRange({0, 0}, {grid.x - 1, grid.y - 1});
    CoreRangeSet all_set(all);
    MakeCB(program, all_set, tt::CBIndex::c_0, 6);   // cb_a: 6 interleaved cross-term a-tiles per k (depth==6 -> clean wrap)
    MakeCB(program, all_set, tt::CBIndex::c_1, 6);   // cb_b: 6 interleaved cross-term b-tiles per k
    MakeCB(program, all_set, tt::CBIndex::c_16, 8, tt::DataFormat::Float32);

    auto [ncores, cores, g1, g2, n1, n2] = tt::tt_metal::split_work_to_cores(grid, n_local, true);   // per-chip work

    std::vector<uint32_t> r_ct = {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_1};
    TensorAccessorArgs(*ah).append_to(r_ct); TensorAccessorArgs(*am).append_to(r_ct); TensorAccessorArgs(*al).append_to(r_ct);
    TensorAccessorArgs(*bh).append_to(r_ct); TensorAccessorArgs(*bm).append_to(r_ct); TensorAccessorArgs(*bl).append_to(r_ct);
    std::vector<uint32_t> w_ct = {(uint32_t)tt::CBIndex::c_16};
    TensorAccessorArgs(*c).append_to(w_ct);

    auto reader = CreateKernel(program, "tt_metal/programming_examples/spmv_mac/kernels/mac_reader.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = r_ct});
    auto writer = CreateKernel(program, "tt_metal/programming_examples/spmv_mac/kernels/mac_writer.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = w_ct});
    auto compute = CreateKernel(program, "tt_metal/programming_examples/spmv_mac/kernels/mac_compute.cpp", cores,
        ComputeConfig{.fp32_dest_acc_en = true, .math_approx_mode = false, .compile_args = {}});

    uint32_t start = 0;
    for (auto [grp, npc] : {std::make_pair(g1, n1), std::make_pair(g2, n2)}) {
        for (const auto& cr : grp.ranges())
            for (const auto& cc : cr) {
                SetRuntimeArgs(program, reader, cc, {(uint32_t)ah->address(), (uint32_t)am->address(), (uint32_t)al->address(),
                    (uint32_t)bh->address(), (uint32_t)bm->address(), (uint32_t)bl->address(), npc, K, start});
                SetRuntimeArgs(program, compute, cc, {npc, K});
                SetRuntimeArgs(program, writer, cc, {(uint32_t)c->address(), npc, start});
                start += npc;
            }
    }

    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(program));
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
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, (double)std::abs(ref[i]));
    for (size_t i = 0; i < n; ++i) { float v=(float)cd[i]; if (std::isfinite(v)) maxrel = std::max(maxrel, std::abs((double)v - ref[i])); }
    double gb = (double)n_out * K * 6 /*6 bf16x3 streams*/ * tile_elems * sizeof(bfloat16) / 1e9;
    printf("SpMV-MAC(metalium): %.3f ms/apply  %.0f GB/s  rel_err=%.3e  (n_out=%u K=%u)\n",
           ms, gb / (ms / 1000.0), maxrel / (scale + 1e-30), n_out, K);
    return 0;
}
