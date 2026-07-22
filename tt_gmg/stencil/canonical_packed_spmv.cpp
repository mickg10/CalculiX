// Standalone correctness/timing harness for the Row236 canonical packed
// base-stencil plus exact dense-fallback SpMV candidate.
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kChips = 8;
constexpr uint32_t kTerms = 81;
constexpr uint32_t kOutputComponents = 3;
constexpr uint32_t kTileElements = 1024;
constexpr uint32_t kCoresPerChip = 56;
constexpr uint32_t kPackedGroupsPerChip = 158;
constexpr uint32_t kBaseGroupsPerChip = 133;
constexpr uint32_t kFallbackGroupsPerChip = 25;
constexpr uint32_t kBaseCoresPerChip = 42;
constexpr uint32_t kFallbackCoresPerChip = 14;
constexpr uint32_t kPaletteSize = 11;
constexpr uint32_t kPaletteTiles = 33;
constexpr uint32_t kPackedBTilesPerChip = kPackedGroupsPerChip * kTerms;
constexpr uint32_t kFallbackATilesPerChip =
    kFallbackGroupsPerChip * kTerms * kOutputComponents;
constexpr uint32_t kOutputTilesPerChip =
    kPackedGroupsPerChip * kOutputComponents;
constexpr uint32_t kTermOrderMode = 2;
constexpr uint32_t kBaseProductsPerGroup = 864;
constexpr uint32_t kFallbackProductsPerGroup = 1404;

static_assert(kBaseCoresPerChip + kFallbackCoresPerChip == kCoresPerChip);
static_assert(kPaletteTiles == 3 * kPaletteSize);

template <typename T>
std::vector<T> read_segment(
    const std::filesystem::path& path,
    uint64_t element_offset,
    uint64_t element_count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    input.seekg(static_cast<std::streamoff>(element_offset * sizeof(T)));
    std::vector<T> result(element_count);
    input.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(element_count * sizeof(T)));
    if (input.gcount() !=
        static_cast<std::streamsize>(element_count * sizeof(T))) {
        throw std::runtime_error("short read from " + path.string());
    }
    return result;
}

void require_file_bytes(
    const std::filesystem::path& path,
    uint64_t expected_bytes) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("missing input " + path.string());
    }
    const uint64_t actual = std::filesystem::file_size(path);
    if (actual != expected_bytes) {
        throw std::runtime_error(
            "input size mismatch for " + path.string() + ": expected " +
            std::to_string(expected_bytes) + ", got " +
            std::to_string(actual));
    }
}

std::shared_ptr<distributed::MeshBuffer> make_sharded_dram(
    const std::shared_ptr<distributed::MeshDevice>& device,
    uint32_t global_tiles,
    uint32_t element_bytes) {
    if (global_tiles % kChips != 0) {
        throw std::runtime_error("mesh buffer does not divide across chips");
    }
    const uint32_t tile_bytes = element_bytes * kTileElements;
    distributed::DeviceLocalBufferConfig local{
        .page_size = tile_bytes,
        .buffer_type = BufferType::DRAM};
    distributed::ShardedBufferConfig global{
        .global_size = static_cast<uint64_t>(tile_bytes) * global_tiles,
        .global_buffer_shape = {32, global_tiles * 32},
        .shard_shape = {32, (global_tiles / kChips) * 32},
        .shard_orientation = ShardOrientation::ROW_MAJOR};
    return distributed::MeshBuffer::create(global, local, device.get());
}

void make_cb(
    Program& program,
    const CoreRangeSet& cores,
    tt::CBIndex cb,
    uint32_t tiles,
    tt::DataFormat format = tt::DataFormat::Float16_b) {
    const uint32_t element_bytes =
        format == tt::DataFormat::Float32 ? 4 : 2;
    const uint32_t tile_bytes = element_bytes * kTileElements;
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(tiles * tile_bytes, {{cb, format}})
            .set_page_size(cb, tile_bytes));
}

void upload_partitioned_u16(
    distributed::MeshCommandQueue& queue,
    const std::shared_ptr<distributed::MeshBuffer>& buffer,
    const std::filesystem::path& path,
    uint64_t elements_per_chip) {
    for (uint32_t chip = 0; chip < kChips; ++chip) {
        auto shard = read_segment<uint16_t>(
            path,
            static_cast<uint64_t>(chip) * elements_per_chip,
            elements_per_chip);
        distributed::WriteShard(
            queue,
            buffer,
            shard,
            distributed::MeshCoordinate(0, chip),
            true);
    }
}

void upload_replicated_u16(
    distributed::MeshCommandQueue& queue,
    const std::shared_ptr<distributed::MeshBuffer>& buffer,
    const std::filesystem::path& path,
    uint64_t elements_per_chip) {
    auto shard = read_segment<uint16_t>(path, 0, elements_per_chip);
    for (uint32_t chip = 0; chip < kChips; ++chip) {
        distributed::WriteShard(
            queue,
            buffer,
            shard,
            distributed::MeshCoordinate(0, chip),
            true);
    }
}

std::vector<float> read_output(
    distributed::MeshCommandQueue& queue,
    const std::shared_ptr<distributed::MeshBuffer>& output) {
    const size_t elements_per_chip =
        static_cast<size_t>(kOutputTilesPerChip) * kTileElements;
    std::vector<float> assembled(kChips * elements_per_chip);
    std::vector<float> shard;
    for (uint32_t chip = 0; chip < kChips; ++chip) {
        distributed::ReadShard(
            queue,
            shard,
            output,
            distributed::MeshCoordinate(0, chip),
            true);
        if (shard.size() != elements_per_chip) {
            throw std::runtime_error("unexpected output shard size");
        }
        std::copy(
            shard.begin(),
            shard.end(),
            assembled.begin() + chip * elements_per_chip);
    }
    return assembled;
}

struct ErrorMetrics {
    double l2_relative = 0.0;
    double max_relative = 0.0;
    double max_absolute = 0.0;
    uint64_t nonfinite = 0;
    bool pass = false;
};

ErrorMetrics compare(
    const std::vector<float>& candidate,
    const std::vector<float>& reference) {
    if (candidate.size() != reference.size()) {
        throw std::runtime_error("candidate/reference size mismatch");
    }
    long double diff2 = 0.0;
    long double ref2 = 0.0;
    double diff_max = 0.0;
    double ref_max = 0.0;
    uint64_t nonfinite = 0;
    for (size_t index = 0; index < candidate.size(); ++index) {
        const double value = candidate[index];
        const double expected = reference[index];
        if (!std::isfinite(value)) {
            ++nonfinite;
            continue;
        }
        const double difference = value - expected;
        diff2 += static_cast<long double>(difference) * difference;
        ref2 += static_cast<long double>(expected) * expected;
        diff_max = std::max(diff_max, std::abs(difference));
        ref_max = std::max(ref_max, std::abs(expected));
    }
    ErrorMetrics result;
    result.l2_relative = std::sqrt(static_cast<double>(
        diff2 / std::max(ref2, static_cast<long double>(1e-300))));
    result.max_relative = diff_max / std::max(ref_max, 1e-300);
    result.max_absolute = diff_max;
    result.nonfinite = nonfinite;
    result.pass =
        nonfinite == 0 && result.l2_relative <= 2.0e-5 &&
        result.max_relative <= 5.0e-5;
    return result;
}

void print_error_diagnostics(
    const std::vector<float>& candidate,
    const std::vector<float>& reference) {
    struct RankedError {
        double absolute;
        size_t index;
    };
    std::vector<RankedError> ranked;
    ranked.reserve(candidate.size());
    for (size_t index = 0; index < candidate.size(); ++index) {
        ranked.push_back({
            std::abs(
                static_cast<double>(candidate[index]) -
                static_cast<double>(reference[index])),
            index});
    }
    const size_t shown = std::min<size_t>(ranked.size(), 16);
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + shown,
        ranked.end(),
        [](const RankedError& left, const RankedError& right) {
            return left.absolute > right.absolute;
        });
    for (size_t rank = 0; rank < shown; ++rank) {
        size_t decoded = ranked[rank].index;
        const uint32_t lane = decoded % kTileElements;
        decoded /= kTileElements;
        const uint32_t component = decoded % kOutputComponents;
        decoded /= kOutputComponents;
        const uint32_t group = decoded % kPackedGroupsPerChip;
        const uint32_t chip = decoded / kPackedGroupsPerChip;
        const size_t index = ranked[rank].index;
        std::printf(
            "DIAG rank=%zu chip=%u group=%u component=%u lane=%u "
            "candidate=%.9e reference=%.9e diff=%.9e\n",
            rank,
            chip,
            group,
            component,
            lane,
            candidate[index],
            reference[index],
            static_cast<double>(candidate[index]) - reference[index]);
    }
}

double elapsed_ms(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        const std::filesystem::path root = argc > 1
            ? argv[1]
            : "/home/ttuser/ttgmg/canonical_packed_v1";
        const uint32_t repetitions = argc > 2
            ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10))
            : 3;
        const std::string vector_tag =
            argc > 3 ? argv[3] : "random_seed351";
        if (repetitions == 0) {
            throw std::runtime_error("repetitions must be positive");
        }

        const auto palette0_path = root / "canonical_base_palette_a0.bf16";
        const auto palette1_path = root / "canonical_base_palette_a1.bf16";
        const auto palette2_path = root / "canonical_base_palette_a2.bf16";
        const auto fallback0_path = root / "fallback_operator_a0.bf16";
        const auto fallback1_path = root / "fallback_operator_a1.bf16";
        const auto fallback2_path = root / "fallback_operator_a2.bf16";
        const auto b0_path =
            root / ("packed_shifted_b0_" + vector_tag + ".bf16");
        const auto b1_path =
            root / ("packed_shifted_b1_" + vector_tag + ".bf16");
        const auto b2_path =
            root / ("packed_shifted_b2_" + vector_tag + ".bf16");
        const auto reference_path =
            root / ("packed_reference_" + vector_tag + ".fp32");

        const uint64_t palette_bytes =
            static_cast<uint64_t>(kPaletteSize) * kTileElements * 2;
        const uint64_t fallback_bytes =
            static_cast<uint64_t>(kChips) * kFallbackATilesPerChip *
            kTileElements * 2;
        const uint64_t packed_b_bytes =
            static_cast<uint64_t>(kChips) * kPackedBTilesPerChip *
            kTileElements * 2;
        const uint64_t output_bytes =
            static_cast<uint64_t>(kChips) * kOutputTilesPerChip *
            kTileElements * 4;
        for (const auto& path : {
                 palette0_path, palette1_path, palette2_path}) {
            require_file_bytes(path, palette_bytes);
        }
        for (const auto& path : {
                 fallback0_path, fallback1_path, fallback2_path}) {
            require_file_bytes(path, fallback_bytes);
        }
        for (const auto& path : {b0_path, b1_path, b2_path}) {
            require_file_bytes(path, packed_b_bytes);
        }
        require_file_bytes(reference_path, output_bytes);

        std::printf(
            "CANONICAL_PACKED_INPUT schema=tt_gmg_canonical_packed_layout_v1 "
            "vector=%s groups_per_chip=%u base_groups=%u fallback_groups_max=%u "
            "term_order=reverse products_base=%u products_fallback=%u\n",
            vector_tag.c_str(),
            kPackedGroupsPerChip,
            kBaseGroupsPerChip,
            kFallbackGroupsPerChip,
            kBaseProductsPerGroup,
            kFallbackProductsPerGroup);
        if (std::getenv("CANONICAL_PACKED_HOST_PREFLIGHT_ONLY") != nullptr) {
            const uint64_t dense_products = 2'223'936;
            const uint64_t packed_products = 1'195'884;
            std::printf(
                "CANONICAL_PACKED_HOST_PREFLIGHT pass=1 opens_device=0 "
                "base_cores=%u fallback_cores=%u dense_products=%llu "
                "packed_products=%llu product_reduction_fraction=%.9f\n",
                kBaseCoresPerChip,
                kFallbackCoresPerChip,
                static_cast<unsigned long long>(dense_products),
                static_cast<unsigned long long>(packed_products),
                1.0 - static_cast<double>(packed_products) / dense_products);
            return 0;
        }

        auto device = distributed::MeshDevice::create(
            distributed::MeshDeviceConfig(distributed::MeshShape(1, kChips)));
        auto& queue = device->mesh_command_queue();
        const auto grid = device->compute_with_storage_grid_size();
        if (grid.x != 8 || grid.y != 7 || grid.x * grid.y != kCoresPerChip) {
            throw std::runtime_error(
                "canonical 42/14 role split requires the QuietBox 8x7 grid");
        }

        auto palette0 = make_sharded_dram(device, kChips * kPaletteSize, 2);
        auto palette1 = make_sharded_dram(device, kChips * kPaletteSize, 2);
        auto palette2 = make_sharded_dram(device, kChips * kPaletteSize, 2);
        auto fallback0 = make_sharded_dram(
            device, kChips * kFallbackATilesPerChip, 2);
        auto fallback1 = make_sharded_dram(
            device, kChips * kFallbackATilesPerChip, 2);
        auto fallback2 = make_sharded_dram(
            device, kChips * kFallbackATilesPerChip, 2);
        auto b0 = make_sharded_dram(device, kChips * kPackedBTilesPerChip, 2);
        auto b1 = make_sharded_dram(device, kChips * kPackedBTilesPerChip, 2);
        auto b2 = make_sharded_dram(device, kChips * kPackedBTilesPerChip, 2);
        auto output = make_sharded_dram(device, kChips * kOutputTilesPerChip, 4);

        const auto upload_start = std::chrono::high_resolution_clock::now();
        const uint64_t palette_elements =
            static_cast<uint64_t>(kPaletteSize) * kTileElements;
        upload_replicated_u16(queue, palette0, palette0_path, palette_elements);
        upload_replicated_u16(queue, palette1, palette1_path, palette_elements);
        upload_replicated_u16(queue, palette2, palette2_path, palette_elements);
        const uint64_t fallback_elements =
            static_cast<uint64_t>(kFallbackATilesPerChip) * kTileElements;
        upload_partitioned_u16(
            queue, fallback0, fallback0_path, fallback_elements);
        upload_partitioned_u16(
            queue, fallback1, fallback1_path, fallback_elements);
        upload_partitioned_u16(
            queue, fallback2, fallback2_path, fallback_elements);
        const uint64_t packed_b_elements =
            static_cast<uint64_t>(kPackedBTilesPerChip) * kTileElements;
        upload_partitioned_u16(queue, b0, b0_path, packed_b_elements);
        upload_partitioned_u16(queue, b1, b1_path, packed_b_elements);
        upload_partitioned_u16(queue, b2, b2_path, packed_b_elements);
        std::vector<float> zero_output(
            static_cast<size_t>(kOutputTilesPerChip) * kTileElements,
            0.0f);
        for (uint32_t chip = 0; chip < kChips; ++chip) {
            distributed::WriteShard(
                queue,
                output,
                zero_output,
                distributed::MeshCoordinate(0, chip),
                true);
        }
        const auto upload_end = std::chrono::high_resolution_clock::now();
        std::printf(
            "UPLOAD canonical_packed_ms=%.3f bytes=%llu\n",
            elapsed_ms(upload_start, upload_end),
            static_cast<unsigned long long>(
                3 * kChips * palette_bytes + 3 * fallback_bytes +
                3 * packed_b_bytes + output_bytes));

        distributed::MeshWorkload workload;
        const auto build_start = std::chrono::high_resolution_clock::now();
        for (uint32_t chip = 0; chip < kChips; ++chip) {
            Program program = CreateProgram();
            const CoreRangeSet base_cores(
                CoreRange({0, 0}, {5, grid.y - 1}));
            const CoreRangeSet fallback_cores(
                CoreRange({6, 0}, {7, grid.y - 1}));

            make_cb(program, base_cores, tt::CBIndex::c_0, kPaletteTiles);
            make_cb(program, base_cores, tt::CBIndex::c_1, 12);
            make_cb(program, base_cores, tt::CBIndex::c_3, 12);
            make_cb(program, base_cores, tt::CBIndex::c_4, 12);
            make_cb(
                program,
                base_cores,
                tt::CBIndex::c_16,
                36,
                tt::DataFormat::Float32);

            make_cb(program, fallback_cores, tt::CBIndex::c_0, 36);
            make_cb(program, fallback_cores, tt::CBIndex::c_1, 12);
            make_cb(program, fallback_cores, tt::CBIndex::c_3, 12);
            make_cb(program, fallback_cores, tt::CBIndex::c_4, 12);
            make_cb(program, fallback_cores, tt::CBIndex::c_5, 18);
            make_cb(
                program,
                fallback_cores,
                tt::CBIndex::c_16,
                36,
                tt::DataFormat::Float32);

            std::vector<uint32_t> base_reader_compile = {
                static_cast<uint32_t>(tt::CBIndex::c_0),
                static_cast<uint32_t>(tt::CBIndex::c_1)};
            TensorAccessorArgs(*palette0).append_to(base_reader_compile);
            TensorAccessorArgs(*palette1).append_to(base_reader_compile);
            TensorAccessorArgs(*palette2).append_to(base_reader_compile);
            TensorAccessorArgs(*b0).append_to(base_reader_compile);
            std::vector<uint32_t> base_writer_compile = {
                static_cast<uint32_t>(tt::CBIndex::c_3),
                static_cast<uint32_t>(tt::CBIndex::c_4),
                static_cast<uint32_t>(tt::CBIndex::c_16)};
            TensorAccessorArgs(*b1).append_to(base_writer_compile);
            TensorAccessorArgs(*b2).append_to(base_writer_compile);
            TensorAccessorArgs(*output).append_to(base_writer_compile);

            const auto base_reader = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_canonical_packed_base_bhi.cpp",
                base_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = base_reader_compile});
            const auto base_writer = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_canonical_packed_base_bmid_blow_writer.cpp",
                base_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = base_writer_compile});
            const auto base_compute = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_canonical_packed_base.cpp",
                base_cores,
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true,
                    .math_approx_mode = false,
                    .compile_args = {kTerms, 1u}});

            std::vector<uint32_t> fallback_reader_compile = {
                static_cast<uint32_t>(tt::CBIndex::c_0),
                static_cast<uint32_t>(tt::CBIndex::c_1)};
            TensorAccessorArgs(*fallback0).append_to(fallback_reader_compile);
            TensorAccessorArgs(*fallback1).append_to(fallback_reader_compile);
            TensorAccessorArgs(*b0).append_to(fallback_reader_compile);
            std::vector<uint32_t> fallback_writer_compile = {
                static_cast<uint32_t>(tt::CBIndex::c_5),
                static_cast<uint32_t>(tt::CBIndex::c_3),
                static_cast<uint32_t>(tt::CBIndex::c_4),
                static_cast<uint32_t>(tt::CBIndex::c_16)};
            TensorAccessorArgs(*fallback2).append_to(fallback_writer_compile);
            TensorAccessorArgs(*b1).append_to(fallback_writer_compile);
            TensorAccessorArgs(*b2).append_to(fallback_writer_compile);
            TensorAccessorArgs(*output).append_to(fallback_writer_compile);

            const auto fallback_reader = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_canonical_packed_fallback_bhi.cpp",
                fallback_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = fallback_reader_compile});
            const auto fallback_writer = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_canonical_packed_fallback_bmid_blow_writer.cpp",
                fallback_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = fallback_writer_compile});
            const auto fallback_compute = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_preexpanded_b.cpp",
                fallback_cores,
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true,
                    .math_approx_mode = false,
                    .compile_args = {kTerms, 1u}});

            auto [base_ncores,
                  base_used_cores,
                  base_group_a,
                  base_group_b,
                  base_count_a,
                  base_count_b] = tt::tt_metal::split_work_to_cores(
                base_cores,
                kBaseGroupsPerChip,
                true);
            uint32_t base_start = 0;
            uint32_t base_core_count = 0;
            for (auto assignment : {
                     std::make_pair(base_group_a, base_count_a),
                     std::make_pair(base_group_b, base_count_b)}) {
                for (const auto& range : assignment.first.ranges()) {
                    for (const auto& core : range) {
                        const uint32_t count = assignment.second;
                        SetRuntimeArgs(
                            program,
                            base_reader,
                            core,
                            {
                                static_cast<uint32_t>(palette0->address()),
                                static_cast<uint32_t>(palette1->address()),
                                static_cast<uint32_t>(palette2->address()),
                                static_cast<uint32_t>(b0->address()),
                                count,
                                base_start,
                                kTermOrderMode,
                            });
                        SetRuntimeArgs(
                            program,
                            base_compute,
                            core,
                            {count, kTermOrderMode});
                        SetRuntimeArgs(
                            program,
                            base_writer,
                            core,
                            {
                                static_cast<uint32_t>(b1->address()),
                                static_cast<uint32_t>(b2->address()),
                                static_cast<uint32_t>(output->address()),
                                count,
                                base_start,
                                kTermOrderMode,
                            });
                        base_start += count;
                        ++base_core_count;
                    }
                }
            }
            if (base_ncores != kBaseCoresPerChip ||
                base_core_count != kBaseCoresPerChip ||
                base_start != kBaseGroupsPerChip ||
                base_used_cores.num_cores() != kBaseCoresPerChip) {
                throw std::runtime_error("unexpected canonical base work split");
            }

            const uint32_t fallback_groups = chip < 5 ? 25u : 24u;
            auto [fallback_ncores,
                  fallback_used_cores,
                  fallback_group_a,
                  fallback_group_b,
                  fallback_count_a,
                  fallback_count_b] = tt::tt_metal::split_work_to_cores(
                fallback_cores,
                fallback_groups,
                true);
            uint32_t fallback_start = 0;
            uint32_t fallback_core_count = 0;
            for (auto assignment : {
                     std::make_pair(fallback_group_a, fallback_count_a),
                     std::make_pair(fallback_group_b, fallback_count_b)}) {
                for (const auto& range : assignment.first.ranges()) {
                    for (const auto& core : range) {
                        const uint32_t count = assignment.second;
                        const uint32_t packed_start =
                            kBaseGroupsPerChip + fallback_start;
                        SetRuntimeArgs(
                            program,
                            fallback_reader,
                            core,
                            {
                                static_cast<uint32_t>(fallback0->address()),
                                static_cast<uint32_t>(fallback1->address()),
                                static_cast<uint32_t>(b0->address()),
                                count,
                                fallback_start,
                                packed_start,
                                kTermOrderMode,
                            });
                        SetRuntimeArgs(
                            program,
                            fallback_compute,
                            core,
                            {count, kTermOrderMode});
                        SetRuntimeArgs(
                            program,
                            fallback_writer,
                            core,
                            {
                                static_cast<uint32_t>(fallback2->address()),
                                static_cast<uint32_t>(b1->address()),
                                static_cast<uint32_t>(b2->address()),
                                static_cast<uint32_t>(output->address()),
                                count,
                                fallback_start,
                                packed_start,
                                kTermOrderMode,
                            });
                        fallback_start += count;
                        ++fallback_core_count;
                    }
                }
            }
            if (fallback_ncores != kFallbackCoresPerChip ||
                fallback_core_count != kFallbackCoresPerChip ||
                fallback_start != fallback_groups ||
                fallback_used_cores.num_cores() != kFallbackCoresPerChip) {
                throw std::runtime_error("unexpected exact fallback work split");
            }

            const distributed::MeshCoordinate coordinate(0, chip);
            workload.add_program(
                distributed::MeshCoordinateRange(coordinate, coordinate),
                std::move(program));
        }
        const auto build_end = std::chrono::high_resolution_clock::now();
        std::printf(
            "PROGRAM_BUILD host_ms=%.3f stages=1 base_cores=%u fallback_cores=%u "
            "base_max_groups_per_core=4 fallback_max_groups_per_core=2\n",
            elapsed_ms(build_start, build_end),
            kBaseCoresPerChip,
            kFallbackCoresPerChip);

        const uint64_t output_elements =
            static_cast<uint64_t>(kChips) * kOutputTilesPerChip *
            kTileElements;
        const auto reference =
            read_segment<float>(reference_path, 0, output_elements);
        const auto enqueue_apply = [&]() {
            distributed::EnqueueMeshWorkload(queue, workload, false);
            distributed::Finish(queue);
        };

        enqueue_apply();
        const auto warm_candidate = read_output(queue, output);
        const auto warm_metrics = compare(warm_candidate, reference);
        std::printf(
            "WARMUP correct=%d l2_rel=%.9e max_rel=%.9e max_abs=%.9e "
            "nonfinite=%llu\n",
            warm_metrics.pass ? 1 : 0,
            warm_metrics.l2_relative,
            warm_metrics.max_relative,
            warm_metrics.max_absolute,
            static_cast<unsigned long long>(warm_metrics.nonfinite));
        if (!warm_metrics.pass) {
            print_error_diagnostics(warm_candidate, reference);
            device->close();
            return 2;
        }

        std::vector<double> samples;
        bool all_correct = true;
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
            const auto start = std::chrono::high_resolution_clock::now();
            enqueue_apply();
            const auto end = std::chrono::high_resolution_clock::now();
            const double milliseconds = elapsed_ms(start, end);
            const auto candidate = read_output(queue, output);
            const auto metrics = compare(candidate, reference);
            all_correct = all_correct && metrics.pass;
            samples.push_back(milliseconds / 1000.0);
            std::printf(
                "SAMPLE index=%u workload_ms=%.6f correct=%d l2_rel=%.9e "
                "max_rel=%.9e max_abs=%.9e nonfinite=%llu\n",
                repetition,
                milliseconds,
                metrics.pass ? 1 : 0,
                metrics.l2_relative,
                metrics.max_relative,
                metrics.max_absolute,
                static_cast<unsigned long long>(metrics.nonfinite));
        }
        auto ordered = samples;
        std::sort(ordered.begin(), ordered.end());
        const double median_seconds = ordered[ordered.size() / 2];
        const bool g3 = all_correct && median_seconds <= 0.003;
        std::printf(
            R"(RESULT_JSON {"schema":"tt_gmg_canonical_packed_spmv_v1","vector":"%s","samples_seconds":[)",
            vector_tag.c_str());
        for (size_t index = 0; index < samples.size(); ++index) {
            std::printf("%s%.9f", index == 0 ? "" : ",", samples[index]);
        }
        std::printf(
            R"(],"median_seconds":%.9f,"all_correct":%s,"l2_limit":2e-5,"max_relative_limit":5e-5,"g3_budget_seconds":0.003,"g3_pass":%s})"
            "\n",
            median_seconds,
            all_correct ? "true" : "false",
            g3 ? "true" : "false");
        device->close();
        return all_correct ? 0 : 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "canonical_packed_spmv fatal: %s\n", error.what());
        return 1;
    }
}
