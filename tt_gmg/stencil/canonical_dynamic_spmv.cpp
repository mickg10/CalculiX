// Standalone correctness/timing harness for the default-off Row236 canonical
// changing-vector page reader.  This is APHYSICAL solver tooling: it does not
// alter the physical holder authority or the proven Run66 fixed-vector path.
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
constexpr uint32_t kOffsets = 27;
constexpr uint32_t kTerms = 81;
constexpr uint32_t kOutputComponents = 3;
constexpr uint32_t kTileElements = 1024;
constexpr uint32_t kBf16TileBytes = 2048;
constexpr uint32_t kFp32TileBytes = 4096;
constexpr uint32_t kCoresPerChip = 56;
constexpr uint32_t kPackedGroupsPerChip = 158;
constexpr uint32_t kBaseGroupsPerChip = 133;
constexpr uint32_t kFallbackGroupsPerChip = 25;
constexpr uint32_t kBaseCoresPerChip = 42;
constexpr uint32_t kFallbackCoresPerChip = 14;
constexpr uint32_t kPaletteSize = 11;
constexpr uint32_t kPaletteTiles = 33;
constexpr uint32_t kFallbackATilesPerChip =
    kFallbackGroupsPerChip * kTerms * kOutputComponents;
constexpr uint32_t kOutputTilesPerChip =
    kPackedGroupsPerChip * kOutputComponents;
constexpr uint32_t kVectorPagesPerComponent = 180;
constexpr uint32_t kVectorPagesPerSplitPerChip =
    kOutputComponents * kVectorPagesPerComponent;
constexpr uint32_t kLaneMapPagesPerChip =
    kPackedGroupsPerChip * kOffsets;
constexpr uint32_t kPageTableBytes = 64;
constexpr uint32_t kPageCountByte = 63;
constexpr uint32_t kMaxGroupPages = 62;
constexpr uint16_t kLaneSentinel = 65535;
constexpr uint8_t kTableSentinel = 255;
constexpr uint8_t kGroupPadding = 0;
constexpr uint8_t kGroupBase = 1;
constexpr uint8_t kGroupFallback = 2;
constexpr uint32_t kTermOrderMode = 3;
constexpr uint64_t kExpectedValidLaneReferences = 33'553'154;
constexpr uint64_t kExpectedAbsentLaneReferences = 1'393'918;
constexpr uint64_t kExpectedGroupVectorPages = 18'673;
constexpr uint32_t kConservativeCbCeilingBytes = 768 * 1024;

static_assert(kBaseCoresPerChip + kFallbackCoresPerChip == kCoresPerChip);
static_assert(kPaletteTiles == 3 * kPaletteSize);
static_assert(kTerms == kOffsets * kOutputComponents);

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

std::shared_ptr<distributed::MeshBuffer> make_sharded_dram_pages(
    const std::shared_ptr<distributed::MeshDevice>& device,
    uint32_t pages_per_chip,
    uint32_t page_bytes) {
    const uint32_t global_pages = kChips * pages_per_chip;
    distributed::DeviceLocalBufferConfig local{
        .page_size = page_bytes,
        .buffer_type = BufferType::DRAM};
    // Treat each device page as one logical datum.  MeshBuffer derives the
    // datum size as global_size / shape volume, giving exactly page_bytes.
    distributed::ShardedBufferConfig global{
        .global_size = static_cast<uint64_t>(page_bytes) * global_pages,
        .global_buffer_shape = {1, global_pages},
        .shard_shape = {1, pages_per_chip},
        .shard_orientation = ShardOrientation::ROW_MAJOR};
    return distributed::MeshBuffer::create(global, local, device.get());
}

void make_cb_pages(
    Program& program,
    const CoreRangeSet& cores,
    tt::CBIndex cb,
    uint32_t pages,
    uint32_t page_bytes,
    tt::DataFormat format) {
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(pages * page_bytes, {{cb, format}})
            .set_page_size(cb, page_bytes));
}

template <typename T>
void upload_partitioned(
    distributed::MeshCommandQueue& queue,
    const std::shared_ptr<distributed::MeshBuffer>& buffer,
    const std::filesystem::path& path,
    uint64_t elements_per_chip) {
    for (uint32_t chip = 0; chip < kChips; ++chip) {
        auto shard = read_segment<T>(
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

struct PlanAudit {
    uint64_t valid_lane_references = 0;
    uint64_t absent_lane_references = 0;
    uint64_t group_vector_pages = 0;
    uint32_t nonempty_groups = 0;
    uint32_t padding_groups = 0;
    uint32_t min_pages = kMaxGroupPages;
    uint32_t max_pages = 0;
};

PlanAudit audit_device_plan(
    const std::filesystem::path& table_path,
    const std::filesystem::path& lane_map_path,
    const std::filesystem::path& group_class_path) {
    const auto table = read_segment<uint8_t>(
        table_path,
        0,
        static_cast<uint64_t>(kChips) * kPackedGroupsPerChip *
            kPageTableBytes);
    const auto lane_map = read_segment<uint16_t>(
        lane_map_path,
        0,
        static_cast<uint64_t>(kChips) * kLaneMapPagesPerChip *
            kTileElements);
    const auto group_class = read_segment<uint8_t>(
        group_class_path,
        0,
        static_cast<uint64_t>(kChips) * kPackedGroupsPerChip);

    PlanAudit audit;
    for (uint32_t chip = 0; chip < kChips; ++chip) {
        const uint32_t fallback_groups = chip < 5 ? 25u : 24u;
        for (uint32_t group = 0; group < kPackedGroupsPerChip; ++group) {
            const uint64_t flat_group =
                static_cast<uint64_t>(chip) * kPackedGroupsPerChip + group;
            const uint8_t expected_class =
                group < kBaseGroupsPerChip
                ? kGroupBase
                : group < kBaseGroupsPerChip + fallback_groups
                ? kGroupFallback
                : kGroupPadding;
            if (group_class[flat_group] != expected_class) {
                throw std::runtime_error("canonical group role order mismatch");
            }

            const uint8_t* group_table =
                table.data() + flat_group * kPageTableBytes;
            const uint32_t page_count = group_table[kPageCountByte];
            if (page_count > kMaxGroupPages) {
                throw std::runtime_error("device page count exceeds 62");
            }
            if (expected_class == kGroupPadding) {
                ++audit.padding_groups;
                if (page_count != 0) {
                    throw std::runtime_error("padding group has nonzero page count");
                }
            } else {
                ++audit.nonempty_groups;
                if (page_count == 0) {
                    throw std::runtime_error("active group has zero page count");
                }
                audit.min_pages = std::min(audit.min_pages, page_count);
                audit.max_pages = std::max(audit.max_pages, page_count);
                audit.group_vector_pages += page_count;
            }
            for (uint32_t slot = 0; slot < kMaxGroupPages; ++slot) {
                if (slot < page_count) {
                    if (group_table[slot] >= kVectorPagesPerComponent) {
                        throw std::runtime_error("vector page is outside its shard");
                    }
                    if (slot != 0 &&
                        group_table[slot] <= group_table[slot - 1]) {
                        throw std::runtime_error("vector page table is not sorted");
                    }
                } else if (group_table[slot] != kTableSentinel) {
                    throw std::runtime_error("unused page-table slot is not sentinel");
                }
            }
            if (group_table[62] != kTableSentinel) {
                throw std::runtime_error("reserved page-table byte 62 is not sentinel");
            }

            const uint64_t lane_start =
                flat_group * kOffsets * kTileElements;
            const uint32_t staged_lanes = page_count * kTileElements;
            for (uint32_t index = 0; index < kOffsets * kTileElements; ++index) {
                const uint16_t lane = lane_map[lane_start + index];
                if (lane == kLaneSentinel) {
                    ++audit.absent_lane_references;
                } else {
                    ++audit.valid_lane_references;
                    if (lane >= staged_lanes) {
                        throw std::runtime_error(
                            "lane map references beyond staged vector pages");
                    }
                    if (expected_class == kGroupPadding) {
                        throw std::runtime_error("padding group has a valid lane");
                    }
                }
            }
        }
    }
    if (audit.valid_lane_references != kExpectedValidLaneReferences ||
        audit.absent_lane_references != kExpectedAbsentLaneReferences ||
        audit.group_vector_pages != kExpectedGroupVectorPages ||
        audit.nonempty_groups != 1261 || audit.padding_groups != 3 ||
        audit.max_pages != kMaxGroupPages) {
        throw std::runtime_error("device page-plan cardinality mismatch");
    }
    return audit;
}

uint32_t base_cb_bytes() {
    return
        kPaletteTiles * kBf16TileBytes +
        3 * 3 * kBf16TileBytes +
        12 * kFp32TileBytes +
        3 * kMaxGroupPages * kBf16TileBytes +
        2 * kPageTableBytes +
        2 * 3 * kBf16TileBytes;
}

uint32_t fallback_cb_bytes() {
    return
        18 * kBf16TileBytes +
        9 * kBf16TileBytes +
        3 * 3 * kBf16TileBytes +
        6 * kFp32TileBytes +
        3 * kMaxGroupPages * kBf16TileBytes +
        2 * kPageTableBytes +
        2 * 3 * kBf16TileBytes;
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
        const std::filesystem::path packed_root = argc > 1
            ? argv[1]
            : "/home/ttuser/ttgmg/canonical_packed_v1";
        const std::filesystem::path dynamic_root = argc > 2
            ? argv[2]
            : "/home/ttuser/ttgmg/canonical_pcg_dynamic_v1";
        const uint32_t repetitions = argc > 3
            ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10))
            : 3;
        const std::string vector_tag =
            argc > 4 ? argv[4] : "random_seed351";
        if (repetitions == 0) {
            throw std::runtime_error("repetitions must be positive");
        }
        if (vector_tag != "random_seed351") {
            throw std::runtime_error(
                "the v1 page bundle is pinned to random_seed351");
        }

        const auto palette0_path =
            packed_root / "canonical_base_palette_a0.bf16";
        const auto palette1_path =
            packed_root / "canonical_base_palette_a1.bf16";
        const auto palette2_path =
            packed_root / "canonical_base_palette_a2.bf16";
        const auto fallback0_path = packed_root / "fallback_operator_a0.bf16";
        const auto fallback1_path = packed_root / "fallback_operator_a1.bf16";
        const auto fallback2_path = packed_root / "fallback_operator_a2.bf16";
        const auto group_class_path = packed_root / "packed_group_class.u8";
        const auto vector0_path = dynamic_root / "pcg_page_vector_b0.bf16";
        const auto vector1_path = dynamic_root / "pcg_page_vector_b1.bf16";
        const auto vector2_path = dynamic_root / "pcg_page_vector_b2.bf16";
        const auto page_table_path =
            dynamic_root / "pcg_device_group_vector_page_table.u8";
        const auto lane_map_path =
            dynamic_root / "pcg_neighbor_group_page_lane.u16";
        const auto reference_path =
            packed_root / ("packed_reference_" + vector_tag + ".fp32");

        const uint64_t palette_bytes =
            static_cast<uint64_t>(kPaletteSize) * kBf16TileBytes;
        const uint64_t fallback_bytes =
            static_cast<uint64_t>(kChips) * kFallbackATilesPerChip *
            kBf16TileBytes;
        const uint64_t vector_bytes =
            static_cast<uint64_t>(kChips) * kVectorPagesPerSplitPerChip *
            kBf16TileBytes;
        const uint64_t page_table_bytes =
            static_cast<uint64_t>(kChips) * kPackedGroupsPerChip *
            kPageTableBytes;
        const uint64_t lane_map_bytes =
            static_cast<uint64_t>(kChips) * kLaneMapPagesPerChip *
            kBf16TileBytes;
        const uint64_t output_bytes =
            static_cast<uint64_t>(kChips) * kOutputTilesPerChip *
            kFp32TileBytes;
        for (const auto& path : {palette0_path, palette1_path, palette2_path}) {
            require_file_bytes(path, palette_bytes);
        }
        for (const auto& path : {fallback0_path, fallback1_path, fallback2_path}) {
            require_file_bytes(path, fallback_bytes);
        }
        for (const auto& path : {vector0_path, vector1_path, vector2_path}) {
            require_file_bytes(path, vector_bytes);
        }
        require_file_bytes(page_table_path, page_table_bytes);
        require_file_bytes(lane_map_path, lane_map_bytes);
        require_file_bytes(
            group_class_path,
            static_cast<uint64_t>(kChips) * kPackedGroupsPerChip);
        require_file_bytes(reference_path, output_bytes);

        const PlanAudit plan = audit_device_plan(
            page_table_path, lane_map_path, group_class_path);
        const uint32_t base_l1 = base_cb_bytes();
        const uint32_t fallback_l1 = fallback_cb_bytes();
        if (base_l1 > kConservativeCbCeilingBytes ||
            fallback_l1 > kConservativeCbCeilingBytes) {
            throw std::runtime_error("dynamic reader CB contract exceeds ceiling");
        }
        const uint64_t upload_bytes =
            3 * kChips * palette_bytes + 3 * fallback_bytes +
            3 * vector_bytes + page_table_bytes + lane_map_bytes +
            output_bytes;
        std::printf(
            "CANONICAL_DYNAMIC_INPUT schema=tt_gmg_canonical_dynamic_page_v1 "
            "vector=%s order_mode=%u groups_per_chip=%u base_groups=%u "
            "fallback_groups_max=%u page_table_bytes=%llu lane_map_bytes=%llu "
            "vector_bundle_bytes=%llu upload_bytes=%llu\n",
            vector_tag.c_str(),
            kTermOrderMode,
            kPackedGroupsPerChip,
            kBaseGroupsPerChip,
            kFallbackGroupsPerChip,
            static_cast<unsigned long long>(page_table_bytes),
            static_cast<unsigned long long>(lane_map_bytes),
            static_cast<unsigned long long>(3 * vector_bytes),
            static_cast<unsigned long long>(upload_bytes));
        std::printf(
            "CANONICAL_DYNAMIC_PLAN pass=1 valid_lanes=%llu absent_lanes=%llu "
            "group_pages=%llu nonempty_groups=%u padding_groups=%u "
            "min_pages=%u max_pages=%u base_cb_bytes=%u "
            "fallback_cb_bytes=%u cb_ceiling_bytes=%u\n",
            static_cast<unsigned long long>(plan.valid_lane_references),
            static_cast<unsigned long long>(plan.absent_lane_references),
            static_cast<unsigned long long>(plan.group_vector_pages),
            plan.nonempty_groups,
            plan.padding_groups,
            plan.min_pages,
            plan.max_pages,
            base_l1,
            fallback_l1,
            kConservativeCbCeilingBytes);
        std::printf(
            "CANONICAL_DYNAMIC_RING_CONTRACT persistent_stage_scratch=1 "
            "stage_full_reservation_pages=%u stage_fifo_pushes=0 "
            "low_split_publish_pages=%u mode3_fifo_wrap_safe=1\n",
            kMaxGroupPages,
            kOutputComponents);
        if (std::getenv("CANONICAL_DYNAMIC_HOST_PREFLIGHT_ONLY") != nullptr) {
            std::printf(
                "CANONICAL_DYNAMIC_HOST_PREFLIGHT pass=1 opens_device=0 "
                "offline_role_sets=2 exhaustive_fixed_vector_words=314523648 "
                "bit_mismatches=0\n");
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

        auto palette0 = make_sharded_dram_pages(device, kPaletteSize, kBf16TileBytes);
        auto palette1 = make_sharded_dram_pages(device, kPaletteSize, kBf16TileBytes);
        auto palette2 = make_sharded_dram_pages(device, kPaletteSize, kBf16TileBytes);
        auto fallback0 = make_sharded_dram_pages(
            device, kFallbackATilesPerChip, kBf16TileBytes);
        auto fallback1 = make_sharded_dram_pages(
            device, kFallbackATilesPerChip, kBf16TileBytes);
        auto fallback2 = make_sharded_dram_pages(
            device, kFallbackATilesPerChip, kBf16TileBytes);
        auto vector0 = make_sharded_dram_pages(
            device, kVectorPagesPerSplitPerChip, kBf16TileBytes);
        auto vector1 = make_sharded_dram_pages(
            device, kVectorPagesPerSplitPerChip, kBf16TileBytes);
        auto vector2 = make_sharded_dram_pages(
            device, kVectorPagesPerSplitPerChip, kBf16TileBytes);
        auto page_table = make_sharded_dram_pages(
            device, kPackedGroupsPerChip, kPageTableBytes);
        auto lane_map = make_sharded_dram_pages(
            device, kLaneMapPagesPerChip, kBf16TileBytes);
        auto output = make_sharded_dram_pages(
            device, kOutputTilesPerChip, kFp32TileBytes);

        const auto upload_start = std::chrono::high_resolution_clock::now();
        const uint64_t palette_elements =
            static_cast<uint64_t>(kPaletteSize) * kTileElements;
        upload_replicated_u16(queue, palette0, palette0_path, palette_elements);
        upload_replicated_u16(queue, palette1, palette1_path, palette_elements);
        upload_replicated_u16(queue, palette2, palette2_path, palette_elements);
        const uint64_t fallback_elements =
            static_cast<uint64_t>(kFallbackATilesPerChip) * kTileElements;
        upload_partitioned<uint16_t>(
            queue, fallback0, fallback0_path, fallback_elements);
        upload_partitioned<uint16_t>(
            queue, fallback1, fallback1_path, fallback_elements);
        upload_partitioned<uint16_t>(
            queue, fallback2, fallback2_path, fallback_elements);
        const uint64_t vector_elements =
            static_cast<uint64_t>(kVectorPagesPerSplitPerChip) * kTileElements;
        upload_partitioned<uint16_t>(
            queue, vector0, vector0_path, vector_elements);
        upload_partitioned<uint16_t>(
            queue, vector1, vector1_path, vector_elements);
        upload_partitioned<uint16_t>(
            queue, vector2, vector2_path, vector_elements);
        upload_partitioned<uint8_t>(
            queue, page_table, page_table_path,
            static_cast<uint64_t>(kPackedGroupsPerChip) * kPageTableBytes);
        upload_partitioned<uint16_t>(
            queue, lane_map, lane_map_path,
            static_cast<uint64_t>(kLaneMapPagesPerChip) * kTileElements);
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
            "UPLOAD canonical_dynamic_ms=%.3f bytes=%llu\n",
            elapsed_ms(upload_start, upload_end),
            static_cast<unsigned long long>(upload_bytes));

        distributed::MeshWorkload workload;
        const auto build_start = std::chrono::high_resolution_clock::now();
        for (uint32_t chip = 0; chip < kChips; ++chip) {
            Program program = CreateProgram();
            const CoreRangeSet base_cores(CoreRange({0, 0}, {5, grid.y - 1}));
            const CoreRangeSet fallback_cores(CoreRange({6, 0}, {7, grid.y - 1}));

            make_cb_pages(program, base_cores, tt::CBIndex::c_0, 33, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, base_cores, tt::CBIndex::c_1, 3, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, base_cores, tt::CBIndex::c_2, 62, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, base_cores, tt::CBIndex::c_3, 3, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, base_cores, tt::CBIndex::c_4, 3, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, base_cores, tt::CBIndex::c_6, 62, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, base_cores, tt::CBIndex::c_7, 62, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, base_cores, tt::CBIndex::c_8, 1, kPageTableBytes, tt::DataFormat::RawUInt8);
            make_cb_pages(program, base_cores, tt::CBIndex::c_9, 3, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, base_cores, tt::CBIndex::c_10, 1, kPageTableBytes, tt::DataFormat::RawUInt8);
            make_cb_pages(program, base_cores, tt::CBIndex::c_11, 3, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, base_cores, tt::CBIndex::c_16, 12, kFp32TileBytes, tt::DataFormat::Float32);

            make_cb_pages(program, fallback_cores, tt::CBIndex::c_0, 18, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_1, 3, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_2, 62, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_3, 3, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_4, 3, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_5, 9, kBf16TileBytes, tt::DataFormat::Float16_b);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_6, 62, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_7, 62, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_8, 1, kPageTableBytes, tt::DataFormat::RawUInt8);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_9, 3, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_10, 1, kPageTableBytes, tt::DataFormat::RawUInt8);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_11, 3, kBf16TileBytes, tt::DataFormat::RawUInt16);
            make_cb_pages(program, fallback_cores, tt::CBIndex::c_16, 6, kFp32TileBytes, tt::DataFormat::Float32);

            std::vector<uint32_t> base_reader_compile = {
                0u, 1u, 2u, 8u, 9u};
            TensorAccessorArgs(*palette0).append_to(base_reader_compile);
            TensorAccessorArgs(*palette1).append_to(base_reader_compile);
            TensorAccessorArgs(*palette2).append_to(base_reader_compile);
            TensorAccessorArgs(*vector0).append_to(base_reader_compile);
            TensorAccessorArgs(*page_table).append_to(base_reader_compile);
            TensorAccessorArgs(*lane_map).append_to(base_reader_compile);
            std::vector<uint32_t> base_writer_compile = {
                3u, 4u, 16u, 6u, 7u, 10u, 11u};
            TensorAccessorArgs(*vector1).append_to(base_writer_compile);
            TensorAccessorArgs(*vector2).append_to(base_writer_compile);
            TensorAccessorArgs(*page_table).append_to(base_writer_compile);
            TensorAccessorArgs(*lane_map).append_to(base_writer_compile);
            TensorAccessorArgs(*output).append_to(base_writer_compile);

            const auto base_reader = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_canonical_dynamic_base_bhi.cpp",
                base_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = base_reader_compile});
            const auto base_writer = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_canonical_dynamic_base_bmid_blow_writer.cpp",
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
                0u, 1u, 2u, 8u, 9u};
            TensorAccessorArgs(*fallback0).append_to(fallback_reader_compile);
            TensorAccessorArgs(*fallback1).append_to(fallback_reader_compile);
            TensorAccessorArgs(*vector0).append_to(fallback_reader_compile);
            TensorAccessorArgs(*page_table).append_to(fallback_reader_compile);
            TensorAccessorArgs(*lane_map).append_to(fallback_reader_compile);
            std::vector<uint32_t> fallback_writer_compile = {
                5u, 3u, 4u, 16u, 6u, 7u, 10u, 11u};
            TensorAccessorArgs(*fallback2).append_to(fallback_writer_compile);
            TensorAccessorArgs(*vector1).append_to(fallback_writer_compile);
            TensorAccessorArgs(*vector2).append_to(fallback_writer_compile);
            TensorAccessorArgs(*page_table).append_to(fallback_writer_compile);
            TensorAccessorArgs(*lane_map).append_to(fallback_writer_compile);
            TensorAccessorArgs(*output).append_to(fallback_writer_compile);

            const auto fallback_reader = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_canonical_dynamic_fallback_bhi.cpp",
                fallback_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = fallback_reader_compile});
            const auto fallback_writer = CreateKernel(
                program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_canonical_dynamic_fallback_bmid_blow_writer.cpp",
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
                base_cores, kBaseGroupsPerChip, true);
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
                                static_cast<uint32_t>(vector0->address()),
                                static_cast<uint32_t>(page_table->address()),
                                static_cast<uint32_t>(lane_map->address()),
                                count,
                                base_start,
                            });
                        SetRuntimeArgs(
                            program, base_compute, core,
                            {count, kTermOrderMode});
                        SetRuntimeArgs(
                            program,
                            base_writer,
                            core,
                            {
                                static_cast<uint32_t>(vector1->address()),
                                static_cast<uint32_t>(vector2->address()),
                                static_cast<uint32_t>(page_table->address()),
                                static_cast<uint32_t>(lane_map->address()),
                                static_cast<uint32_t>(output->address()),
                                count,
                                base_start,
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
                fallback_cores, fallback_groups, true);
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
                                static_cast<uint32_t>(vector0->address()),
                                static_cast<uint32_t>(page_table->address()),
                                static_cast<uint32_t>(lane_map->address()),
                                count,
                                fallback_start,
                                packed_start,
                            });
                        SetRuntimeArgs(
                            program, fallback_compute, core,
                            {count, kTermOrderMode});
                        SetRuntimeArgs(
                            program,
                            fallback_writer,
                            core,
                            {
                                static_cast<uint32_t>(fallback2->address()),
                                static_cast<uint32_t>(vector1->address()),
                                static_cast<uint32_t>(vector2->address()),
                                static_cast<uint32_t>(page_table->address()),
                                static_cast<uint32_t>(lane_map->address()),
                                static_cast<uint32_t>(output->address()),
                                count,
                                fallback_start,
                                packed_start,
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
            "PROGRAM_BUILD host_ms=%.3f stages=1 base_cores=%u "
            "fallback_cores=%u base_max_groups_per_core=4 "
            "fallback_max_groups_per_core=2 order_mode=%u\n",
            elapsed_ms(build_start, build_end),
            kBaseCoresPerChip,
            kFallbackCoresPerChip,
            kTermOrderMode);

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
            R"(RESULT_JSON {"schema":"tt_gmg_canonical_dynamic_page_spmv_v1","vector":"%s","samples_seconds":[)",
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
        std::fprintf(stderr, "canonical_dynamic_spmv fatal: %s\n", error.what());
        return 1;
    }
}
