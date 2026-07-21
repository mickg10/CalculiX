// reduce_fp32.cpp — verify reduce_tile<SUM,REDUCE_ROW,enforce_fp32_accumulation=true> preserves EXTREME
// cancellation (Sum of O(1e3) terms -> ~0.02). This is the make-or-break primitive for the TT-GMG fp32 fine-SpMV:
// matmul (11-bit products), ttnn.sum (bf16 accumulate), and packer_l1_acc all FAILED cancellation; only fp32
// products + fp32 accumulate works (CPU emu bf16x3 = 2.6e-4). If this hits ~fp32, the transposed-reduce fix is real.
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
using namespace tt;
using namespace tt::tt_metal;

static std::shared_ptr<distributed::MeshBuffer> MakeBuf(
    const std::shared_ptr<distributed::MeshDevice>& dev, uint32_t global_tiles, uint32_t ebytes) {
    constexpr uint32_t H = tt::constants::TILE_HEIGHT, W = tt::constants::TILE_WIDTH;
    uint32_t ts = ebytes * H * W;
    distributed::DeviceLocalBufferConfig lc{.page_size = ts, .buffer_type = BufferType::DRAM};
    distributed::ShardedBufferConfig bc{
        .global_size = (uint64_t)ts * global_tiles,
        .global_buffer_shape = {H, global_tiles * W},
        .shard_shape = {H, global_tiles * W},
        .shard_orientation = ShardOrientation::ROW_MAJOR};
    return distributed::MeshBuffer::create(bc, lc, dev.get());
}
static void MakeCB(Program& p, const CoreRangeSet& cores, tt::CBIndex cb, uint32_t n_tiles, tt::DataFormat fmt) {
    uint32_t eb = (fmt == tt::DataFormat::Float32) ? 4u : 2u;
    uint32_t ts = eb * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    CircularBufferConfig cfg = CircularBufferConfig(n_tiles * ts, {{cb, fmt}}).set_page_size(cb, ts);
    CreateCircularBuffer(p, cores, cfg);
}
int main() {
    const uint32_t NT = 16, H = 32, W = 32, C = NT * W;   // 32 rows x 512 cols = 16 fp32 tiles
    std::vector<float> prod((size_t)H * C);
    std::vector<double> ref(H, 0.0);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1000.f);
    for (uint32_t r = 0; r < H; ++r) {
        double m = 0;
        for (uint32_t c = 0; c < C; ++c) { float v = nd(rng); prod[r * C + c] = v; m += v; }
        m /= C;
        for (uint32_t c = 0; c < C; ++c) prod[r * C + c] -= (float)m;   // per-row Sum ~ 0
        prod[r * C + 0] += 0.02f;                                        // small known target
        for (uint32_t c = 0; c < C; ++c) ref[r] += (double)prod[r * C + c];
    }
    // tilize into NT [32,32] fp32 tiles (tile ti = columns [ti*32, ti*32+32))
    std::vector<float> tiled((size_t)H * C);
    for (uint32_t ti = 0; ti < NT; ++ti)
        for (uint32_t rr = 0; rr < H; ++rr)
            for (uint32_t cc = 0; cc < W; ++cc)
                tiled[(size_t)ti * H * W + rr * W + cc] = prod[rr * C + ti * W + cc];

    auto dev = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(1, 1)));
    Program program = CreateProgram();
    auto& cq = dev->mesh_command_queue();
    auto in = MakeBuf(dev, NT, 4), out = MakeBuf(dev, 1, 4);                    // in = fp32 products, out = fp32
    distributed::EnqueueWriteMeshBuffer(cq, in, tiled, true);
    std::vector<float> zc(H * W, 0.f); distributed::EnqueueWriteMeshBuffer(cq, out, zc, true);

    CoreRange one({0, 0}, {0, 0}); CoreRangeSet cores(one);
    MakeCB(program, cores, tt::CBIndex::c_0, 4, tt::DataFormat::Float32);   // input products (fp32)
    MakeCB(program, cores, tt::CBIndex::c_2, 1, tt::DataFormat::Float16_b); // scaler
    MakeCB(program, cores, tt::CBIndex::c_16, 1, tt::DataFormat::Float32);  // output

    std::vector<uint32_t> r_ct = {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_2};
    TensorAccessorArgs(*in).append_to(r_ct);
    std::vector<uint32_t> w_ct = {(uint32_t)tt::CBIndex::c_16};
    TensorAccessorArgs(*out).append_to(w_ct);
    auto reader = CreateKernel(program, "tt_metal/programming_examples/reduce_fp32/kernels/reduce_reader.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = r_ct});
    auto writer = CreateKernel(program, "tt_metal/programming_examples/reduce_fp32/kernels/reduce_writer.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = w_ct});
    auto compute = CreateKernel(program, "tt_metal/programming_examples/reduce_fp32/kernels/reduce_compute.cpp", cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false, .compile_args = {}});
    SetRuntimeArgs(program, reader, one, {(uint32_t)in->address(), NT, 0x3f803f80u});  // bf16 1.0 packed x2
    SetRuntimeArgs(program, compute, one, {NT});
    SetRuntimeArgs(program, writer, one, {(uint32_t)out->address()});

    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, wl, true);
    std::vector<float> od; distributed::EnqueueReadMeshBuffer(cq, od, out, true);
    // reduce output: row sums in column 0 of the output tile
    double maxerr = 0, maxref = 0;
    for (uint32_t r = 0; r < H; ++r) {
        double got = (double)od[r * W + 0], e = std::abs(got - ref[r]);
        if (e > maxerr) maxerr = e; if (std::abs(ref[r]) > maxref) maxref = std::abs(ref[r]);
        if (r < 4) printf("  row%u got=%.5f ref=%.5f\n", r, got, ref[r]);
    }
    printf("REDUCE fp32 cancellation: max_abs_err=%.3e  |ref|max=%.3e  term_scale~1e3  (HOST fp32=2.6e-4; bf16 fails ~9)\n",
           maxerr, maxref);
    return 0;
}
