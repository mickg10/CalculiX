// Standalone correctness and timing harness for the real Row236 brick-stencil operator.
#include <tt-metalium/bfloat16.hpp>
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
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kChips = 8;
constexpr uint32_t kGroupsPerChip = 198;
constexpr uint32_t kTerms = 81;
constexpr uint32_t kOutputComponents = 3;
constexpr uint32_t kChunks = 3;
constexpr uint32_t kTileElements = 1024;
constexpr uint32_t kCoresPerChip = 56;
constexpr uint32_t kMaxGroupsPerCore = 4;
constexpr uint32_t kHaloPlanes = 9;
constexpr uint32_t kHaloPlaneTiles = 8;
constexpr uint32_t kHaloTilesPerGroup = kHaloPlanes * kHaloPlaneTiles;
constexpr uint32_t kHaloTilesPerCore = kMaxGroupsPerCore * kHaloTilesPerGroup;
constexpr uint32_t kATilesPerChip = kGroupsPerChip * kTerms * kOutputComponents;
constexpr uint32_t kBTilesPerChip = kGroupsPerChip * kTerms;
constexpr uint32_t kATilesPerReaderPage = 1;
constexpr uint32_t kHaloTilesPerChip = kCoresPerChip * kHaloTilesPerCore;
constexpr uint32_t kOutputTilesPerChip = kGroupsPerChip * kOutputComponents;
constexpr uint32_t kPartialTilesPerGroup = kChunks * kOutputComponents;
constexpr uint32_t kPartialTilesPerChip = kGroupsPerChip * kPartialTilesPerGroup;
constexpr uint32_t kPacklowTermsPerPublish = 3;
constexpr uint32_t kPacklowATilesPerPublish =
    kPacklowTermsPerPublish * 3 * kOutputComponents;
constexpr uint32_t kPacklowACbTiles = 2 * kPacklowATilesPerPublish;
constexpr uint32_t kPacklowBCbTiles = 12;
constexpr uint32_t kSplitAHiMidTilesPerPublish =
    kPacklowTermsPerPublish * 2 * kOutputComponents;
constexpr uint32_t kSplitALowTilesPerPublish =
    kPacklowTermsPerPublish * kOutputComponents;
constexpr uint32_t kSplitAHiMidCbTiles = 2 * kSplitAHiMidTilesPerPublish;
constexpr uint32_t kSplitALowCbTiles = 2 * kSplitALowTilesPerPublish;
constexpr uint32_t kCornerOffsets = 8;
constexpr uint32_t kCornerCoefficientTypes = 3;
constexpr uint32_t kCornerCompressedTilesPerBatch =
    kChunks * kCornerCoefficientTypes;
constexpr uint32_t kCornerCompressedTilesPerChip =
    kGroupsPerChip * kCornerOffsets * kCornerCompressedTilesPerBatch;
constexpr uint32_t kCornerCompressedCbTiles =
    2 * kCornerCompressedTilesPerBatch;
static_assert(kPacklowACbTiles % kPacklowATilesPerPublish == 0);
static_assert(kSplitAHiMidCbTiles % kSplitAHiMidTilesPerPublish == 0);
static_assert(kSplitALowCbTiles % kSplitALowTilesPerPublish == 0);
static_assert(
    kCornerCompressedCbTiles % kCornerCompressedTilesPerBatch == 0);
static_assert(kATilesPerChip % kATilesPerReaderPage == 0);

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
    if (input.gcount() != static_cast<std::streamsize>(element_count * sizeof(T))) {
        throw std::runtime_error("short read from " + path.string());
    }
    return result;
}

float bf16_to_float(uint16_t bits) {
    const uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

uint16_t float_to_bf16(float value) {
    uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    const uint32_t rounded =
        word + 0x7fffu + ((word >> 16) & 1u);
    return static_cast<uint16_t>(rounded >> 16);
}

void fold_corner_a_low_affine(std::vector<uint16_t>& shard) {
    const uint64_t expected =
        static_cast<uint64_t>(kATilesPerChip) * kTileElements;
    if (shard.size() != expected) {
        throw std::runtime_error("unexpected A-low shard size for affine corner fold");
    }
    constexpr uint32_t corner_offsets[] = {0, 2, 6, 8, 18, 20, 24, 26};
    constexpr uint32_t destination_offsets[] = {13, 4, 22, 10, 16, 12, 14};
    const auto tile_pointer = [&shard](
                                  uint32_t group,
                                  uint32_t offset,
                                  uint32_t input_component,
                                  uint32_t output_component) {
        const uint32_t term = offset * 3 + input_component;
        const uint64_t tile =
            (static_cast<uint64_t>(group) * kTerms + term) *
                kOutputComponents +
            output_component;
        return shard.data() + tile * kTileElements;
    };
    for (uint32_t group = 0; group < kGroupsPerChip; ++group) {
        for (uint32_t output_component = 0;
             output_component < kOutputComponents;
             ++output_component) {
            for (uint32_t input_component = 0;
                 input_component < 3;
                 ++input_component) {
                const uint16_t* corners[8];
                uint16_t* destinations[7];
                for (uint32_t index = 0; index < 8; ++index) {
                    corners[index] = tile_pointer(
                        group,
                        corner_offsets[index],
                        input_component,
                        output_component);
                }
                for (uint32_t index = 0; index < 7; ++index) {
                    destinations[index] = tile_pointer(
                        group,
                        destination_offsets[index],
                        input_component,
                        output_component);
                }
                for (uint32_t lane = 0; lane < kTileElements; ++lane) {
                    const float c0 = bf16_to_float(corners[0][lane]);
                    const float c1 = bf16_to_float(corners[1][lane]);
                    const float c2 = bf16_to_float(corners[2][lane]);
                    const float c3 = bf16_to_float(corners[3][lane]);
                    const float c4 = bf16_to_float(corners[4][lane]);
                    const float c5 = bf16_to_float(corners[5][lane]);
                    const float c6 = bf16_to_float(corners[6][lane]);
                    const float c7 = bf16_to_float(corners[7][lane]);
                    float correction[7] = {};
                    correction[0] -= 2.0f * c0;
                    correction[0] -= 2.0f * c1;
                    correction[0] -= 2.0f * c2;
                    correction[0] -= 2.0f * c3;
                    correction[0] -= 2.0f * c4;
                    correction[0] -= 2.0f * c5;
                    correction[0] -= 2.0f * c6;
                    correction[0] -= 2.0f * c7;
                    correction[1] = ((c0 + c1) + c2) + c3;
                    correction[2] = ((c4 + c5) + c6) + c7;
                    correction[3] = ((c0 + c1) + c4) + c5;
                    correction[4] = ((c2 + c3) + c6) + c7;
                    correction[5] = ((c0 + c2) + c4) + c6;
                    correction[6] = ((c1 + c3) + c5) + c7;
                    for (uint32_t destination = 0; destination < 7; ++destination) {
                        const float adjusted =
                            bf16_to_float(destinations[destination][lane]) +
                            correction[destination];
                        destinations[destination][lane] = float_to_bf16(adjusted);
                    }
                }
            }
        }
    }
}

void self_test_corner_a_low_affine() {
    std::vector<uint16_t> shard(
        static_cast<size_t>(kATilesPerChip) * kTileElements,
        float_to_bf16(0.0f));
    constexpr uint32_t corner_offsets[] = {0, 2, 6, 8, 18, 20, 24, 26};
    constexpr uint32_t face_offsets[] = {4, 22, 10, 16, 12, 14};
    constexpr uint32_t destination_offsets[] = {13, 4, 22, 10, 16, 12, 14};
    const auto element_index = [](
                                   uint32_t group,
                                   uint32_t offset,
                                   uint32_t input_component,
                                   uint32_t output_component,
                                   uint32_t lane) {
        const uint32_t term = offset * 3 + input_component;
        const uint64_t tile =
            (static_cast<uint64_t>(group) * kTerms + term) *
                kOutputComponents +
            output_component;
        return tile * kTileElements + lane;
    };
    constexpr uint32_t group = 17;
    constexpr uint32_t input_component = 2;
    constexpr uint32_t output_component = 1;
    constexpr uint32_t lane = 351;
    for (const uint32_t offset : corner_offsets) {
        shard[element_index(
            group, offset, input_component, output_component, lane)] =
            float_to_bf16(1.0f);
    }
    fold_corner_a_low_affine(shard);
    const uint16_t expected_center = float_to_bf16(-16.0f);
    const uint16_t expected_face = float_to_bf16(4.0f);
    const uint16_t actual_center = shard[element_index(
        group, 13, input_component, output_component, lane)];
    if (actual_center != expected_center) {
        throw std::runtime_error("affine corner-fold self-test center mismatch");
    }
    for (const uint32_t offset : face_offsets) {
        if (shard[element_index(
                group, offset, input_component, output_component, lane)] !=
            expected_face) {
            throw std::runtime_error("affine corner-fold self-test face mismatch");
        }
    }
    if (shard[element_index(
            group, 1, input_component, output_component, lane)] !=
        float_to_bf16(0.0f)) {
        throw std::runtime_error("affine corner-fold self-test changed an edge");
    }
    std::printf(
        "AFFINE_FOLD_SELFTEST pass=1 center=%.1f face=%.1f shard_elements=%llu\n",
        bf16_to_float(actual_center),
        bf16_to_float(expected_face),
        static_cast<unsigned long long>(shard.size()));

    shard.clear();
    shard.shrink_to_fit();
    std::vector<uint16_t> optimized(
        static_cast<size_t>(kATilesPerChip) * kTileElements,
        float_to_bf16(0.0f));
    constexpr uint32_t cases = 32;
    struct Case {
        uint32_t group;
        uint32_t input_component;
        uint32_t output_component;
        uint32_t lane;
    };
    Case test_cases[cases];
    for (uint32_t sample = 0; sample < cases; ++sample) {
        const Case test_case{
            (17u + 37u * sample) % kGroupsPerChip,
            sample % 3u,
            (sample / 3u) % kOutputComponents,
            (351u + 73u * sample) % kTileElements};
        test_cases[sample] = test_case;
        for (uint32_t index = 0; index < 8; ++index) {
            const float value = static_cast<float>(
                static_cast<int32_t>((sample + 3u) * (index + 5u) % 29u) - 14) /
                16.0f;
            optimized[element_index(
                test_case.group,
                corner_offsets[index],
                test_case.input_component,
                test_case.output_component,
                test_case.lane)] = float_to_bf16(value);
        }
        for (uint32_t index = 0; index < 7; ++index) {
            const float value = static_cast<float>(
                static_cast<int32_t>((sample + 11u) * (index + 2u) % 31u) - 15) /
                32.0f;
            optimized[element_index(
                test_case.group,
                destination_offsets[index],
                test_case.input_component,
                test_case.output_component,
                test_case.lane)] = float_to_bf16(value);
        }
    }
    auto reference = optimized;
    for (const Case& test_case : test_cases) {
        float correction[7] = {};
        for (const uint32_t offset : corner_offsets) {
            const uint32_t di = offset / 9;
            const uint32_t dj = (offset / 3) % 3;
            const uint32_t dk = offset % 3;
            const float corner = bf16_to_float(reference[element_index(
                test_case.group,
                offset,
                test_case.input_component,
                test_case.output_component,
                test_case.lane)]);
            correction[0] -= 2.0f * corner;
            correction[di == 0 ? 1 : 2] += corner;
            correction[dj == 0 ? 3 : 4] += corner;
            correction[dk == 0 ? 5 : 6] += corner;
        }
        for (uint32_t destination = 0; destination < 7; ++destination) {
            const uint64_t index = element_index(
                test_case.group,
                destination_offsets[destination],
                test_case.input_component,
                test_case.output_component,
                test_case.lane);
            reference[index] = float_to_bf16(
                bf16_to_float(reference[index]) + correction[destination]);
        }
    }
    fold_corner_a_low_affine(optimized);
    if (optimized != reference) {
        throw std::runtime_error("optimized affine corner fold differs from scalar reference");
    }
    std::printf(
        "AFFINE_FOLD_EQUIV_SELFTEST pass=1 cases=%u full_shard_compare=1\n",
        cases);
}

void self_test_split_a_layout() {
    constexpr uint32_t terms_per_publish = 3;
    constexpr uint32_t output_components = 3;
    std::vector<uint32_t> combined(terms_per_publish * 9);
    std::iota(combined.begin(), combined.end(), 1u);
    std::vector<uint32_t> hi_mid(kSplitAHiMidTilesPerPublish);
    std::vector<uint32_t> low(kSplitALowTilesPerPublish);
    for (uint32_t batch_lane = 0;
         batch_lane < terms_per_publish;
         ++batch_lane) {
        for (uint32_t output_component = 0;
             output_component < output_components;
             ++output_component) {
            hi_mid[batch_lane * 6 + output_component] =
                combined[batch_lane * 9 + output_component];
            hi_mid[batch_lane * 6 + 3 + output_component] =
                combined[batch_lane * 9 + 3 + output_component];
            low[batch_lane * 3 + output_component] =
                combined[batch_lane * 9 + 6 + output_component];
        }
    }
    for (uint32_t batch_lane = 0;
         batch_lane < terms_per_publish;
         ++batch_lane) {
        for (uint32_t output_component = 0;
             output_component < output_components;
             ++output_component) {
            if (hi_mid[batch_lane * 6 + output_component] !=
                    combined[batch_lane * 9 + output_component] ||
                hi_mid[batch_lane * 6 + 3 + output_component] !=
                    combined[batch_lane * 9 + 3 + output_component] ||
                low[batch_lane * 3 + output_component] !=
                    combined[batch_lane * 9 + 6 + output_component]) {
                throw std::runtime_error("split-A layout self-test mismatch");
            }
        }
    }
    uint32_t retained_b_low_terms = 0;
    for (uint32_t term = 0; term < kTerms; ++term) {
        const uint32_t offset_id = term / 3;
        const uint32_t di = offset_id / 9;
        const uint32_t dj = (offset_id / 3) % 3;
        const uint32_t dk = offset_id % 3;
        const uint32_t manhattan =
            (di > 1 ? di - 1 : 1 - di) +
            (dj > 1 ? dj - 1 : 1 - dj) +
            (dk > 1 ? dk - 1 : 1 - dk);
        retained_b_low_terms +=
            manhattan <= 2 || offset_id == 6 || offset_id == 20 ? 1u : 0u;
    }
    if (retained_b_low_terms != 63) {
        throw std::runtime_error("split-A run52 b-low policy mismatch");
    }
    std::printf(
        "SPLIT_A_LAYOUT_SELFTEST pass=1 combined_tiles=%zu hi_mid_tiles=%zu "
        "low_tiles=%zu retained_b_low_terms=%u\n",
        combined.size(),
        hi_mid.size(),
        low.size(),
        retained_b_low_terms);
}

void self_test_corner_constant_layout() {
    constexpr uint8_t expected_type_map[kCornerOffsets][3][3] = {
        {{1, 2, 2}, {2, 1, 2}, {2, 2, 1}},
        {{1, 2, 0}, {2, 1, 0}, {0, 0, 1}},
        {{1, 0, 2}, {0, 1, 0}, {2, 0, 1}},
        {{1, 0, 0}, {0, 1, 2}, {0, 2, 1}},
        {{1, 0, 0}, {0, 1, 2}, {0, 2, 1}},
        {{1, 0, 2}, {0, 1, 0}, {2, 0, 1}},
        {{1, 2, 0}, {2, 1, 0}, {0, 0, 1}},
        {{1, 2, 2}, {2, 1, 2}, {2, 2, 1}},
    };
    constexpr uint32_t corner_offsets[kCornerOffsets] = {
        0, 2, 6, 8, 18, 20, 24, 26};
    for (uint32_t corner = 0; corner < kCornerOffsets; ++corner) {
        const uint32_t offset = corner_offsets[corner];
        const bool signs[3] = {
            offset / 9 == 2,
            (offset / 3) % 3 == 2,
            offset % 3 == 2,
        };
        for (uint32_t input = 0; input < 3; ++input) {
            for (uint32_t output = 0; output < 3; ++output) {
                const uint32_t type = input == output
                    ? 1u
                    : signs[input] == signs[output] ? 2u : 0u;
                if (type != expected_type_map[corner][input][output]) {
                    throw std::runtime_error(
                        "corner constant coefficient-type map mismatch");
                }
            }
        }
        for (uint32_t type = 0; type < 3; ++type) {
            uint32_t representative_input = 0;
            uint32_t representative_output = 0;
            if (type != 1) {
                const bool seek_equal = type == 2;
                if ((signs[0] == signs[1]) == seek_equal) {
                    representative_output = 1;
                } else if ((signs[0] == signs[2]) == seek_equal) {
                    representative_output = 2;
                } else if (seek_equal) {
                    representative_input = 1;
                    representative_output = 2;
                }
            }
            bool type_is_used = false;
            for (uint32_t input = 0; input < 3; ++input) {
                for (uint32_t output = 0; output < 3; ++output) {
                    type_is_used = type_is_used ||
                        expected_type_map[corner][input][output] == type;
                }
            }
            if (type_is_used &&
                expected_type_map[corner][representative_input]
                                 [representative_output] != type) {
                throw std::runtime_error(
                    "corner representative does not carry requested type");
            }
        }
    }
    const auto ordered_term = [](uint32_t stream_term, uint32_t mode) {
        if (mode == 1) {
            if (stream_term < 3) {
                return 39u + stream_term;
            }
            const uint32_t remainder = stream_term - 3;
            return remainder < 39 ? remainder : remainder + 3;
        }
        return mode == 2 ? 80u - stream_term : stream_term;
    };
    for (uint32_t mode = 0; mode < 3; ++mode) {
        bool seen[81] = {};
        for (uint32_t batch = 0; batch < 27; ++batch) {
            const uint32_t offset = ordered_term(batch * 3, mode) / 3;
            for (uint32_t lane = 0; lane < 3; ++lane) {
                const uint32_t term = ordered_term(batch * 3 + lane, mode);
                if (term / 3 != offset || seen[term]) {
                    throw std::runtime_error(
                        "term order breaks corner compression batches");
                }
                seen[term] = true;
            }
        }
        if (std::count(std::begin(seen), std::end(seen), true) != 81) {
            throw std::runtime_error("term order does not cover all terms");
        }
    }
    if (kCornerCompressedTilesPerChip != 14256 ||
        kGroupsPerChip * kCornerOffsets * 27 != 42768) {
        throw std::runtime_error("corner compression tile-count mismatch");
    }
    std::printf(
        "CORNER_CONSTANT_LAYOUT_SELFTEST pass=1 coefficient_types=%u "
        "compressed_tiles_per_chip=%u original_corner_tiles_per_chip=%u "
        "term_orders_checked=3\n",
        kCornerCoefficientTypes,
        kCornerCompressedTilesPerChip,
        kGroupsPerChip * kCornerOffsets * 27);
}

std::shared_ptr<distributed::MeshBuffer> make_sharded_dram(
    const std::shared_ptr<distributed::MeshDevice>& device,
    uint32_t global_tiles,
    uint32_t element_bytes,
    uint32_t tiles_per_page = 1) {
    const uint32_t tile_bytes = element_bytes * kTileElements;
    if (global_tiles % kChips != 0) {
        throw std::runtime_error("mesh buffer does not divide evenly across chips");
    }
    if (tiles_per_page == 0 || global_tiles % tiles_per_page != 0 ||
        (global_tiles / kChips) % tiles_per_page != 0) {
        throw std::runtime_error("mesh buffer page does not divide tile shards");
    }
    distributed::DeviceLocalBufferConfig local{
        .page_size = tile_bytes * tiles_per_page,
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
    const uint32_t element_bytes = format == tt::DataFormat::Float32 ? 4 : 2;
    const uint32_t tile_bytes = element_bytes * kTileElements;
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(tiles * tile_bytes, {{cb, format}})
            .set_page_size(cb, tile_bytes));
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
    result.l2_relative =
        std::sqrt(static_cast<double>(diff2 / std::max(ref2, static_cast<long double>(1e-300))));
    result.max_relative = diff_max / std::max(ref_max, 1e-300);
    result.max_absolute = diff_max;
    result.nonfinite = nonfinite;
    result.pass =
        nonfinite == 0 && result.l2_relative <= 2.0e-5 && result.max_relative <= 5.0e-5;
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
    long double component_diff2[kOutputComponents] = {};
    long double component_ref2[kOutputComponents] = {};
    double component_max[kOutputComponents] = {};
    uint64_t exact = 0;
    for (size_t index = 0; index < candidate.size(); ++index) {
        const size_t within_chip = index %
            (static_cast<size_t>(kOutputTilesPerChip) * kTileElements);
        const uint32_t component =
            static_cast<uint32_t>((within_chip / kTileElements) % kOutputComponents);
        const double difference =
            static_cast<double>(candidate[index]) - static_cast<double>(reference[index]);
        component_diff2[component] +=
            static_cast<long double>(difference) * difference;
        component_ref2[component] +=
            static_cast<long double>(reference[index]) * reference[index];
        component_max[component] =
            std::max(component_max[component], std::abs(difference));
        exact += candidate[index] == reference[index];
        ranked.push_back({std::abs(difference), index});
    }
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + std::min<size_t>(ranked.size(), 24),
        ranked.end(),
        [](const RankedError& left, const RankedError& right) {
            return left.absolute > right.absolute;
        });
    std::printf(
        "DIAG exact=%llu/%llu\n",
        static_cast<unsigned long long>(exact),
        static_cast<unsigned long long>(candidate.size()));
    for (uint32_t component = 0; component < kOutputComponents; ++component) {
        const double relative = std::sqrt(static_cast<double>(
            component_diff2[component] /
            std::max(component_ref2[component], static_cast<long double>(1e-300))));
        std::printf(
            "DIAG component=%u l2_rel=%.9e max_abs=%.9e\n",
            component,
            relative,
            component_max[component]);
    }
    const size_t shown = std::min<size_t>(ranked.size(), 24);
    for (size_t rank = 0; rank < shown; ++rank) {
        const size_t index = ranked[rank].index;
        size_t decoded = index;
        const uint32_t lane = static_cast<uint32_t>(decoded % kTileElements);
        decoded /= kTileElements;
        const uint32_t component = static_cast<uint32_t>(decoded % kOutputComponents);
        decoded /= kOutputComponents;
        const uint32_t local_group = static_cast<uint32_t>(decoded % kGroupsPerChip);
        const uint32_t chip = static_cast<uint32_t>(decoded / kGroupsPerChip);
        std::printf(
            "DIAG rank=%zu chip=%u group=%u component=%u lane=%u "
            "candidate=%.9e reference=%.9e diff=%.9e\n",
            rank,
            chip,
            local_group,
            component,
            lane,
            candidate[index],
            reference[index],
            static_cast<double>(candidate[index]) - reference[index]);
    }
}

std::vector<float> read_output(
    distributed::MeshCommandQueue& queue,
    const std::shared_ptr<distributed::MeshBuffer>& output) {
    std::vector<float> assembled(
        static_cast<size_t>(kChips) * kOutputTilesPerChip * kTileElements);
    std::vector<float> shard;
    for (uint32_t chip = 0; chip < kChips; ++chip) {
        distributed::ReadShard(
            queue,
            shard,
            output,
            distributed::MeshCoordinate(0, chip),
            true);
        const size_t expected =
            static_cast<size_t>(kOutputTilesPerChip) * kTileElements;
        if (shard.size() != expected) {
            throw std::runtime_error("unexpected output shard size");
        }
        std::copy(
            shard.begin(),
            shard.end(),
            assembled.begin() + static_cast<size_t>(chip) * expected);
    }
    return assembled;
}

void print_partial_diagnostics(
    distributed::MeshCommandQueue& queue,
    const std::shared_ptr<distributed::MeshBuffer>& partial) {
    struct Target {
        uint32_t chip;
        uint32_t group;
        uint32_t component;
        uint32_t lane;
    };
    constexpr Target targets[] = {
        {5, 110, 0, 54},
        {5, 79, 0, 539},
    };
    std::vector<float> shard;
    uint32_t loaded_chip = kChips;
    for (const auto& target : targets) {
        if (loaded_chip != target.chip) {
            distributed::ReadShard(
                queue,
                shard,
                partial,
                distributed::MeshCoordinate(0, target.chip),
                true);
            const size_t expected =
                static_cast<size_t>(kPartialTilesPerChip) * kTileElements;
            if (shard.size() != expected) {
                throw std::runtime_error("unexpected partial shard size");
            }
            loaded_chip = target.chip;
        }
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
            const size_t tile =
                static_cast<size_t>(target.group) * kPartialTilesPerGroup +
                chunk * kOutputComponents + target.component;
            const float value = shard[tile * kTileElements + target.lane];
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            std::printf(
                "PARTIAL_DIAG chip=%u group=%u component=%u lane=%u chunk=%u "
                "value=%.9e bits=0x%08x\n",
                target.chip,
                target.group,
                target.component,
                target.lane,
                chunk,
                value,
                bits);
        }
    }
}

void print_target_outputs(const std::vector<float>& output) {
    struct Target {
        uint32_t chip;
        uint32_t group;
        uint32_t component;
        uint32_t lane;
    };
    constexpr Target targets[] = {
        {5, 110, 0, 54},
        {5, 79, 0, 539},
    };
    for (const auto& target : targets) {
        const size_t index =
            (((static_cast<size_t>(target.chip) * kGroupsPerChip + target.group) *
              kOutputComponents + target.component) *
             kTileElements) +
            target.lane;
        uint32_t bits = 0;
        std::memcpy(&bits, &output[index], sizeof(bits));
        std::printf(
            "OUTPUT_DIAG chip=%u group=%u component=%u lane=%u value=%.9e bits=0x%08x\n",
            target.chip,
            target.group,
            target.component,
            target.lane,
            output[index],
            bits);
    }
}

uint64_t verify_u16_upload(
    distributed::MeshCommandQueue& queue,
    const std::shared_ptr<distributed::MeshBuffer>& buffer,
    const std::filesystem::path& source_path,
    uint64_t elements_per_chip,
    const char* label,
    bool affine_corner_fold = false) {
    uint64_t mismatches = 0;
    uint64_t first_global = 0;
    uint16_t first_expected = 0;
    uint16_t first_actual = 0;
    bool have_first = false;
    std::vector<uint16_t> actual;
    for (uint32_t chip = 0; chip < kChips; ++chip) {
        distributed::ReadShard(
            queue,
            actual,
            buffer,
            distributed::MeshCoordinate(0, chip),
            true);
        auto expected = read_segment<uint16_t>(
            source_path,
            static_cast<uint64_t>(chip) * elements_per_chip,
            elements_per_chip);
        if (affine_corner_fold) {
            fold_corner_a_low_affine(expected);
        }
        if (actual.size() != expected.size()) {
            throw std::runtime_error(std::string("unexpected readback size for ") + label);
        }
        for (size_t index = 0; index < expected.size(); ++index) {
            if (actual[index] == expected[index]) {
                continue;
            }
            if (!have_first) {
                first_global = static_cast<uint64_t>(chip) * elements_per_chip + index;
                first_expected = expected[index];
                first_actual = actual[index];
                have_first = true;
            }
            ++mismatches;
        }
    }
    std::printf(
        "UPLOAD_VERIFY label=%s mismatches=%llu first_global=%llu expected=0x%04x "
        "actual=0x%04x\n",
        label,
        static_cast<unsigned long long>(mismatches),
        static_cast<unsigned long long>(first_global),
        first_expected,
        first_actual);
    return mismatches;
}

uint64_t bitwise_output_mismatches(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("output reread size mismatch");
    }
    uint64_t mismatches = 0;
    for (size_t index = 0; index < left.size(); ++index) {
        uint32_t left_bits = 0;
        uint32_t right_bits = 0;
        std::memcpy(&left_bits, &left[index], sizeof(left_bits));
        std::memcpy(&right_bits, &right[index], sizeof(right_bits));
        mismatches += left_bits != right_bits;
    }
    return mismatches;
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
        const std::filesystem::path root =
            argc > 1 ? argv[1] : "/home/ttuser/ttgmg/device_v1";
        const uint32_t repetitions =
            argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 3;
        const std::string vector_tag = argc > 3 ? argv[3] : "random_seed351";
        const char* first_chunk_terms_env = std::getenv("BRICK_FIRST_CHUNK_TERMS");
        const uint32_t first_chunk_terms = first_chunk_terms_env == nullptr
            ? 27u
            : static_cast<uint32_t>(std::strtoul(first_chunk_terms_env, nullptr, 10));
        const bool swap_chip_0_5 = std::getenv("BRICK_SWAP_CHIP_0_5") != nullptr;
        const bool flush_tiny_a = std::getenv("BRICK_FLUSH_TINY_A") != nullptr;
        const bool term_sfpu_accum =
            std::getenv("BRICK_TERM_SFPU_ACCUM") != nullptr;
        const bool component_pass_sfpu =
            std::getenv("BRICK_COMPONENT_PASS_SFPU") != nullptr;
        const bool one_stage_fpu =
            std::getenv("BRICK_ONE_STAGE_FPU") != nullptr;
        const bool bf16x2_dual =
            std::getenv("BRICK_BF16X2_DUAL") != nullptr;
        const bool bf16x3_dual =
            std::getenv("BRICK_BF16X3_DUAL") != nullptr;
        const bool bf16x3_packlow =
            std::getenv("BRICK_BF16X3_PACKLOW") != nullptr;
        const bool a_reuse_profile =
            std::getenv("BRICK_A_REUSE_PROFILE") != nullptr;
        const bool preexpanded_b =
            std::getenv("BRICK_PREEXPANDED_B") != nullptr;
        const bool bf16x3_selective_blow =
            std::getenv("BRICK_BF16X3_SELECTIVE_BLOW") != nullptr ||
            a_reuse_profile || preexpanded_b;
        const bool bf16x3_corner_constant_selective_blow =
            std::getenv("BRICK_BF16X3_CORNER_CONSTANT_SELECTIVE_BLOW") != nullptr;
        const bool bf16x3_corner_reuse_selective_blow =
            std::getenv("BRICK_BF16X3_CORNER_REUSE_SELECTIVE_BLOW") != nullptr;
        const bool bf16x3_split_a_selective_blow =
            std::getenv("BRICK_BF16X3_SPLIT_A_SELECTIVE_BLOW") != nullptr;
        const bool bf16x3_affine_corner_fold =
            std::getenv("BRICK_BF16X3_AFFINE_CORNER_FOLD") != nullptr;
        const bool affine_corner_fold_host_bench =
            std::getenv("BRICK_AFFINE_FOLD_HOST_BENCH") != nullptr;
        const bool bf16x3_hifi2_exact_leading =
            std::getenv("BRICK_BF16X3_HIFI2_EXACT_LEADING") != nullptr;
        const bool allow_incorrect_profile =
            std::getenv("BRICK_ALLOW_INCORRECT_PROFILE") != nullptr;
        if (a_reuse_profile && !allow_incorrect_profile) {
            throw std::runtime_error(
                "BRICK_A_REUSE_PROFILE requires BRICK_ALLOW_INCORRECT_PROFILE");
        }
        if (a_reuse_profile && preexpanded_b) {
            throw std::runtime_error(
                "BRICK_A_REUSE_PROFILE and BRICK_PREEXPANDED_B are mutually exclusive");
        }
        const bool dual_reader =
            bf16x2_dual || bf16x3_dual || bf16x3_packlow ||
            bf16x3_selective_blow ||
            bf16x3_corner_constant_selective_blow ||
            bf16x3_corner_reuse_selective_blow ||
            bf16x3_split_a_selective_blow ||
            bf16x3_affine_corner_fold ||
            bf16x3_hifi2_exact_leading;
        const bool single_stage =
            term_sfpu_accum || component_pass_sfpu || one_stage_fpu ||
            dual_reader;
        const char* term_order_env = std::getenv("BRICK_TERM_ORDER");
        const std::string term_order =
            term_order_env == nullptr ? "natural" : term_order_env;
        const uint32_t term_order_mode =
            term_order == "natural" ? 0u
            : term_order == "center_first" ? 1u
            : term_order == "reverse" ? 2u
            : 3u;
        const char* term_count_env = std::getenv("BRICK_TERM_COUNT");
        const uint32_t term_count = term_count_env == nullptr
            ? 81u
            : static_cast<uint32_t>(std::strtoul(term_count_env, nullptr, 10));
        const char* sfpu_batch_tiles_env =
            std::getenv("BRICK_SFPU_BATCH_TILES");
        const uint32_t sfpu_batch_tiles = sfpu_batch_tiles_env == nullptr
            ? 1u
            : static_cast<uint32_t>(
                  std::strtoul(sfpu_batch_tiles_env, nullptr, 10));
        const char* reduce_mode_env = std::getenv("BRICK_REDUCE_MODE");
        const uint32_t reduce_mode = reduce_mode_env == nullptr
            ? 0u
            : static_cast<uint32_t>(std::strtoul(reduce_mode_env, nullptr, 10));
        const auto source_chip_for = [swap_chip_0_5](uint32_t physical_chip) {
            if (!swap_chip_0_5) {
                return physical_chip;
            }
            if (physical_chip == 0) {
                return 5u;
            }
            if (physical_chip == 5) {
                return 0u;
            }
            return physical_chip;
        };
        std::printf("ARGS argc=%d repetitions=%u", argc, repetitions);
        for (int index = 0; index < argc; ++index) {
            std::printf(" argv%d=%s", index, argv[index]);
        }
        std::printf("\n");
        std::printf(
            "CHIP_MAPPING swap_0_5=%d physical0_source=%u physical5_source=%u\n",
            swap_chip_0_5 ? 1 : 0,
            source_chip_for(0),
            source_chip_for(5));
        std::printf(
            "TINY_A_CANONICALIZATION enabled=%d raw_bf16_exponent_lt=88\n",
            flush_tiny_a ? 1 : 0);
        std::printf("FIRST_CHUNK_TERMS count=%u\n", first_chunk_terms);
        if (first_chunk_terms < 1 || first_chunk_terms > 27) {
            throw std::runtime_error("BRICK_FIRST_CHUNK_TERMS must be in [1,27]");
        }
        std::printf("REDUCE_MODE mode=%u\n", reduce_mode);
        if (reduce_mode > 1) {
            throw std::runtime_error("BRICK_REDUCE_MODE must be 0 (sum) or 1 (copy chunk 0)");
        }
        std::printf(
            "ACCUMULATION mode=%s term_count=%u sfpu_batch_tiles=%u\n",
            one_stage_fpu
                ? "fpu_one_stage_ordered"
                : preexpanded_b
                ? "fpu_bf16x3_preexpanded_b_selective_blow_manhattan2_opposite"
                : bf16x2_dual
                ? "fpu_bf16x2_dual_reader"
                : bf16x3_dual
                ? "fpu_bf16x3_dual_reader"
                : bf16x3_packlow
                ? "fpu_bf16x3_packlow_reader"
                : bf16x3_selective_blow
                ? "fpu_bf16x3_selective_blow_manhattan2_opposite"
                : bf16x3_corner_constant_selective_blow
                ? "fpu_bf16x3_exact_corner_constant_selective_blow_manhattan2_opposite"
                : bf16x3_corner_reuse_selective_blow
                ? "fpu_bf16x3_exact_corner_reuse_fixed_cb_selective_blow_manhattan2_opposite"
                : bf16x3_split_a_selective_blow
                ? "fpu_bf16x3_split_a_selective_blow_manhattan2_opposite"
                : bf16x3_affine_corner_fold
                ? "fpu_bf16x3_affine_corner_fold_alow_manhattan2_blow_manhattan2_opposite"
                : bf16x3_hifi2_exact_leading
                ? "fpu_bf16x3_fixed_hifi2_exact_leading"
                : component_pass_sfpu
                ? "fpu_component_pass_sfpu_batch3"
                : (term_sfpu_accum
                       ? "fpu_term_sfpu_sum"
                       : "fpu_chunk_sfpu_reduce"),
            term_count,
            sfpu_batch_tiles);
        std::printf(
            "DIAGNOSTIC allow_incorrect_profile=%d gate_eligible=%d\n",
            allow_incorrect_profile ? 1 : 0,
            allow_incorrect_profile ? 0 : 1);
        std::printf(
            "A_REUSE_PROFILE enabled=%d seeded_a_cb_windows=2 gate_eligible=0\n",
            a_reuse_profile ? 1 : 0);
        std::printf(
            "PREEXPANDED_B enabled=%d representation=shifted_resident_vector_v1\n",
            preexpanded_b ? 1 : 0);
        std::printf("TERM_ORDER name=%s mode=%u\n", term_order.c_str(), term_order_mode);
        if (term_order_mode > 2) {
            throw std::runtime_error(
                "BRICK_TERM_ORDER must be natural, center_first, or reverse");
        }
        if (term_count < 1 || term_count > 81) {
            throw std::runtime_error("BRICK_TERM_COUNT must be in [1,81]");
        }
        if (sfpu_batch_tiles < 1 || sfpu_batch_tiles > 5) {
            throw std::runtime_error("BRICK_SFPU_BATCH_TILES must be in [1,5]");
        }
        if ((term_sfpu_accum ? 1 : 0) +
                (component_pass_sfpu ? 1 : 0) +
                (one_stage_fpu ? 1 : 0) +
                (bf16x2_dual ? 1 : 0) +
                (bf16x3_dual ? 1 : 0) +
                (bf16x3_packlow ? 1 : 0) +
                (bf16x3_selective_blow ? 1 : 0) +
                (bf16x3_corner_constant_selective_blow ? 1 : 0) +
                (bf16x3_corner_reuse_selective_blow ? 1 : 0) +
                (bf16x3_split_a_selective_blow ? 1 : 0) +
                (bf16x3_affine_corner_fold ? 1 : 0) +
                (bf16x3_hifi2_exact_leading ? 1 : 0) >
            1) {
            throw std::runtime_error(
                "accumulation-mode environment flags are mutually exclusive");
        }
        if (bf16x3_corner_constant_selective_blow && flush_tiny_a) {
            throw std::runtime_error(
                "corner-constant artifact is bitwise source data; do not combine it with BRICK_FLUSH_TINY_A");
        }
        if (std::getenv("BRICK_AFFINE_FOLD_SELFTEST") != nullptr) {
            self_test_corner_a_low_affine();
            return 0;
        }
        if (std::getenv("BRICK_SPLIT_A_LAYOUT_SELFTEST") != nullptr) {
            self_test_split_a_layout();
            return 0;
        }
        if (std::getenv("BRICK_CORNER_CONSTANT_LAYOUT_SELFTEST") != nullptr) {
            self_test_corner_constant_layout();
            return 0;
        }
        if (term_sfpu_accum && sfpu_batch_tiles != 1) {
            throw std::runtime_error(
                "fp32 destination mode has four tiles; term-SFPU batching must stay at 1");
        }
        if (repetitions < 3) {
            throw std::runtime_error("at least three timed samples are required");
        }
        const auto halo_path = root / ("halo_" + vector_tag + ".bf16");
        const auto reference_path = root / ("reference_" + vector_tag + ".fp32");
        const uint64_t a_elements_per_chip =
            static_cast<uint64_t>(kATilesPerChip) * kTileElements;
        const uint64_t b_elements_per_chip =
            static_cast<uint64_t>(kBTilesPerChip) * kTileElements;
        const uint64_t corner_elements_per_chip =
            static_cast<uint64_t>(kCornerCompressedTilesPerChip) *
            kTileElements;
        const uint64_t halo_elements_per_chip =
            static_cast<uint64_t>(kHaloTilesPerChip) * kTileElements;
        const uint64_t output_elements_per_chip =
            static_cast<uint64_t>(kOutputTilesPerChip) * kTileElements;

        if (affine_corner_fold_host_bench) {
            const auto path = root / "operator_a2.bf16";
            const auto total_start = std::chrono::high_resolution_clock::now();
            double total_read_ms = 0.0;
            double total_fold_ms = 0.0;
            uint64_t checksum = 1469598103934665603ull;
            for (uint32_t chip = 0; chip < kChips; ++chip) {
                const uint32_t source_chip = source_chip_for(chip);
                const auto read_start = std::chrono::high_resolution_clock::now();
                auto shard = read_segment<uint16_t>(
                    path,
                    static_cast<uint64_t>(source_chip) * a_elements_per_chip,
                    a_elements_per_chip);
                const auto read_end = std::chrono::high_resolution_clock::now();
                fold_corner_a_low_affine(shard);
                const auto fold_end = std::chrono::high_resolution_clock::now();
                const size_t stride = std::max<size_t>(1, shard.size() / 4096);
                for (size_t index = 0; index < shard.size(); index += stride) {
                    checksum ^= static_cast<uint64_t>(shard[index]);
                    checksum *= 1099511628211ull;
                }
                const double read_ms = elapsed_ms(read_start, read_end);
                const double fold_ms = elapsed_ms(read_end, fold_end);
                total_read_ms += read_ms;
                total_fold_ms += fold_ms;
                std::printf(
                    "AFFINE_FOLD_HOST_BENCH chip=%u source_chip=%u read_ms=%.3f "
                    "fold_ms=%.3f sampled_checksum=%016llx\n",
                    chip,
                    source_chip,
                    read_ms,
                    fold_ms,
                    static_cast<unsigned long long>(checksum));
                std::fflush(stdout);
            }
            const auto total_end = std::chrono::high_resolution_clock::now();
            std::printf(
                "AFFINE_FOLD_HOST_BENCH_RESULT chips=%u read_ms=%.3f fold_ms=%.3f "
                "total_ms=%.3f sampled_checksum=%016llx\n",
                kChips,
                total_read_ms,
                total_fold_ms,
                elapsed_ms(total_start, total_end),
                static_cast<unsigned long long>(checksum));
            return 0;
        }

        auto device = distributed::MeshDevice::create(
            distributed::MeshDeviceConfig(distributed::MeshShape(1, kChips)));
        auto& queue = device->mesh_command_queue();
        const auto grid = device->compute_with_storage_grid_size();
        std::printf(
            "BRICK_SPMV target chips=%u grid=%ux%u groups/chip=%u halo/core=%u KiB\n",
            kChips,
            static_cast<uint32_t>(grid.x),
            static_cast<uint32_t>(grid.y),
            kGroupsPerChip,
            kHaloTilesPerCore * 2);
        if (grid.x * grid.y != kCoresPerChip) {
            throw std::runtime_error("unexpected worker grid");
        }

        // Keep each coefficient tile as one DRAM page.  The measured
        // nine-tile-page experiment regressed because large page reads
        // serialized the coefficient streams.  Batching belongs at the
        // circular-buffer boundary, where it removes synchronization
        // without changing the DRAM page geometry.
        auto a0 = make_sharded_dram(
            device, kChips * kATilesPerChip, 2, kATilesPerReaderPage);
        auto a1 = make_sharded_dram(
            device, kChips * kATilesPerChip, 2, kATilesPerReaderPage);
        auto a2 = make_sharded_dram(
            device, kChips * kATilesPerChip, 2, kATilesPerReaderPage);
        std::shared_ptr<distributed::MeshBuffer> b0;
        std::shared_ptr<distributed::MeshBuffer> b1;
        std::shared_ptr<distributed::MeshBuffer> b2;
        if (preexpanded_b) {
            b0 = make_sharded_dram(
                device, kChips * kBTilesPerChip, 2, kATilesPerReaderPage);
            b1 = make_sharded_dram(
                device, kChips * kBTilesPerChip, 2, kATilesPerReaderPage);
            b2 = make_sharded_dram(
                device, kChips * kBTilesPerChip, 2, kATilesPerReaderPage);
        }
        std::shared_ptr<distributed::MeshBuffer> corner_a;
        if (bf16x3_corner_constant_selective_blow) {
            corner_a = make_sharded_dram(
                device,
                kChips * kCornerCompressedTilesPerChip,
                2,
                kATilesPerReaderPage);
        }
        auto halo = make_sharded_dram(device, kChips * kHaloTilesPerChip, 2);
        auto partial =
            make_sharded_dram(device, kChips * kPartialTilesPerChip, 4);
        auto output =
            make_sharded_dram(device, kChips * kOutputTilesPerChip, 4);
        const std::shared_ptr<distributed::MeshBuffer> a_buffers[] = {a0, a1, a2};
        const std::shared_ptr<distributed::MeshBuffer> b_buffers[] = {b0, b1, b2};

        const auto upload_a_start = std::chrono::high_resolution_clock::now();
        uint64_t tiny_a_canonicalized = 0;
        for (uint32_t split = 0; split < 3; ++split) {
            const auto path = root / ("operator_a" + std::to_string(split) + ".bf16");
            for (uint32_t chip = 0; chip < kChips; ++chip) {
                const uint32_t source_chip = source_chip_for(chip);
                auto shard = read_segment<uint16_t>(
                    path,
                    static_cast<uint64_t>(source_chip) * a_elements_per_chip,
                    a_elements_per_chip);
                if (flush_tiny_a) {
                    for (auto& bits : shard) {
                        const uint16_t magnitude = bits & 0x7fffu;
                        const uint16_t exponent = (bits >> 7) & 0xffu;
                        if (magnitude != 0 && exponent < 88) {
                            bits = 0;
                            ++tiny_a_canonicalized;
                        }
                    }
                }
                if (bf16x3_affine_corner_fold && split == 2) {
                    fold_corner_a_low_affine(shard);
                }
                distributed::WriteShard(
                    queue,
                    a_buffers[split],
                    shard,
                    distributed::MeshCoordinate(0, chip),
                    true);
            }
        }
        const auto upload_a_end = std::chrono::high_resolution_clock::now();
        const auto upload_b_start = upload_a_end;
        if (preexpanded_b) {
            for (uint32_t split = 0; split < 3; ++split) {
                const auto path = root /
                    ("shifted_b" + std::to_string(split) + "_" +
                     vector_tag + ".bf16");
                for (uint32_t chip = 0; chip < kChips; ++chip) {
                    const uint32_t source_chip = source_chip_for(chip);
                    auto shard = read_segment<uint16_t>(
                        path,
                        static_cast<uint64_t>(source_chip) *
                            b_elements_per_chip,
                        b_elements_per_chip);
                    distributed::WriteShard(
                        queue,
                        b_buffers[split],
                        shard,
                        distributed::MeshCoordinate(0, chip),
                        true);
                }
            }
        }
        const auto upload_b_end = std::chrono::high_resolution_clock::now();
        const auto upload_corner_start = upload_b_end;
        if (bf16x3_corner_constant_selective_blow) {
            const auto path = root / "corner_a_exact.bf16";
            for (uint32_t chip = 0; chip < kChips; ++chip) {
                const uint32_t source_chip = source_chip_for(chip);
                auto shard = read_segment<uint16_t>(
                    path,
                    static_cast<uint64_t>(source_chip) *
                        corner_elements_per_chip,
                    corner_elements_per_chip);
                distributed::WriteShard(
                    queue,
                    corner_a,
                    shard,
                    distributed::MeshCoordinate(0, chip),
                    true);
            }
        }
        const auto upload_corner_end = std::chrono::high_resolution_clock::now();
        const auto upload_halo_start = upload_corner_end;
        for (uint32_t chip = 0; chip < kChips; ++chip) {
            const uint32_t source_chip = source_chip_for(chip);
            auto shard = read_segment<uint16_t>(
                halo_path,
                static_cast<uint64_t>(source_chip) * halo_elements_per_chip,
                halo_elements_per_chip);
            distributed::WriteShard(
                queue,
                halo,
                shard,
                distributed::MeshCoordinate(0, chip),
                true);
        }
        const auto upload_halo_end = std::chrono::high_resolution_clock::now();
        std::vector<float> zero_output(output_elements_per_chip, 0.0f);
        for (uint32_t chip = 0; chip < kChips; ++chip) {
            distributed::WriteShard(
                queue,
                output,
                zero_output,
                distributed::MeshCoordinate(0, chip),
                true);
        }
        std::printf(
            "UPLOAD operator_bf16x3_ms=%.3f corner_exact_ms=%.3f "
            "preexpanded_b_bf16x3_ms=%.3f halo_bf16x3_ms=%.3f "
            "tiny_a_canonicalized=%llu "
            "affine_corner_fold=%d corner_constant=%d\n",
            elapsed_ms(upload_a_start, upload_a_end),
            elapsed_ms(upload_corner_start, upload_corner_end),
            elapsed_ms(upload_b_start, upload_b_end),
            elapsed_ms(upload_halo_start, upload_halo_end),
            static_cast<unsigned long long>(tiny_a_canonicalized),
            bf16x3_affine_corner_fold ? 1 : 0,
            bf16x3_corner_constant_selective_blow ? 1 : 0);
        if (std::getenv("BRICK_VERIFY_UPLOAD") != nullptr) {
            uint64_t mismatches = 0;
            for (uint32_t split = 0; split < 3; ++split) {
                const auto path = root / ("operator_a" + std::to_string(split) + ".bf16");
                const std::string label = "operator_a" + std::to_string(split);
                mismatches += verify_u16_upload(
                    queue,
                    a_buffers[split],
                    path,
                    a_elements_per_chip,
                    label.c_str(),
                    bf16x3_affine_corner_fold && split == 2);
            }
            if (preexpanded_b) {
                for (uint32_t split = 0; split < 3; ++split) {
                    const auto path = root /
                        ("shifted_b" + std::to_string(split) + "_" +
                         vector_tag + ".bf16");
                    const std::string label =
                        "shifted_b" + std::to_string(split);
                    mismatches += verify_u16_upload(
                        queue,
                        b_buffers[split],
                        path,
                        b_elements_per_chip,
                        label.c_str());
                }
            }
            if (bf16x3_corner_constant_selective_blow) {
                mismatches += verify_u16_upload(
                    queue,
                    corner_a,
                    root / "corner_a_exact.bf16",
                    corner_elements_per_chip,
                    "corner_a_exact");
            }
            mismatches += verify_u16_upload(
                queue,
                halo,
                halo_path,
                halo_elements_per_chip,
                "halo");
            std::printf(
                "UPLOAD_VERIFY total_mismatches=%llu\n",
                static_cast<unsigned long long>(mismatches));
            if (mismatches != 0) {
                device->close();
                return 3;
            }
        }

        distributed::MeshWorkload partial_workload;
        distributed::MeshWorkload reduce_workload;
        const auto build_start = std::chrono::high_resolution_clock::now();
        for (uint32_t chip = 0; chip < kChips; ++chip) {
            Program partial_program = CreateProgram();
            Program reduce_program = CreateProgram();
            CoreRangeSet all_cores(
                CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
            // Publish three terms at a time. Two A batches make one exact CB
            // pointer cycle; the measured nine-term experiment was neutral.
            make_cb(
                partial_program,
                all_cores,
                tt::CBIndex::c_0,
                (bf16x3_split_a_selective_blow || preexpanded_b)
                    ? kSplitAHiMidCbTiles
                    : (bf16x3_packlow || bf16x3_selective_blow ||
                 bf16x3_corner_constant_selective_blow ||
                 bf16x3_corner_reuse_selective_blow ||
                 bf16x3_affine_corner_fold ||
                 bf16x3_hifi2_exact_leading)
                    ? kPacklowACbTiles
                    : 36);
            make_cb(
                partial_program,
                all_cores,
                tt::CBIndex::c_1,
                (bf16x3_packlow || bf16x3_selective_blow ||
                 bf16x3_corner_constant_selective_blow ||
                 bf16x3_corner_reuse_selective_blow ||
                 bf16x3_affine_corner_fold ||
                 bf16x3_hifi2_exact_leading)
                    ? kPacklowBCbTiles
                    : 12);
            if (dual_reader) {
                make_cb(
                    partial_program,
                    all_cores,
                    tt::CBIndex::c_3,
                    (bf16x3_packlow || bf16x3_selective_blow ||
                     bf16x3_corner_constant_selective_blow ||
                     bf16x3_corner_reuse_selective_blow ||
                     bf16x3_affine_corner_fold ||
                     bf16x3_hifi2_exact_leading)
                        ? kPacklowBCbTiles
                        : 12);
            }
            if (bf16x3_packlow || bf16x3_selective_blow ||
                bf16x3_corner_constant_selective_blow ||
                bf16x3_corner_reuse_selective_blow ||
                bf16x3_split_a_selective_blow ||
                bf16x3_affine_corner_fold ||
                bf16x3_hifi2_exact_leading) {
                make_cb(
                    partial_program,
                    all_cores,
                    tt::CBIndex::c_4,
                    kPacklowBCbTiles);
            }
            if (bf16x3_split_a_selective_blow || preexpanded_b ||
                bf16x3_corner_constant_selective_blow) {
                make_cb(
                    partial_program,
                    all_cores,
                    tt::CBIndex::c_5,
                    bf16x3_corner_constant_selective_blow
                        ? kCornerCompressedCbTiles
                        : kSplitALowCbTiles);
            }
            if (!preexpanded_b) {
                make_cb(
                    partial_program,
                    all_cores,
                    tt::CBIndex::c_2,
                    kHaloTilesPerCore);
            }
            make_cb(
                partial_program,
                all_cores,
                tt::CBIndex::c_16,
                36,
                tt::DataFormat::Float32);

            make_cb(
                reduce_program,
                all_cores,
                tt::CBIndex::c_0,
                36,
                tt::DataFormat::Float32);
            make_cb(
                reduce_program,
                all_cores,
                tt::CBIndex::c_16,
                12,
                tt::DataFormat::Float32);

            auto [ncores, cores, group_a, group_b, count_a, count_b] =
                tt::tt_metal::split_work_to_cores(
                    all_cores,
                    kGroupsPerChip,
                    true);
            std::vector<uint32_t> partial_reader_compile;
            if (preexpanded_b) {
                partial_reader_compile = {
                    static_cast<uint32_t>(tt::CBIndex::c_0),
                    static_cast<uint32_t>(tt::CBIndex::c_1)};
                TensorAccessorArgs(*a0).append_to(partial_reader_compile);
                TensorAccessorArgs(*a1).append_to(partial_reader_compile);
                TensorAccessorArgs(*b0).append_to(partial_reader_compile);
            } else if (bf16x3_corner_constant_selective_blow) {
                partial_reader_compile = {
                    static_cast<uint32_t>(tt::CBIndex::c_0),
                    static_cast<uint32_t>(tt::CBIndex::c_1),
                    static_cast<uint32_t>(tt::CBIndex::c_5),
                    static_cast<uint32_t>(tt::CBIndex::c_2)};
                TensorAccessorArgs(*a0).append_to(partial_reader_compile);
                TensorAccessorArgs(*a1).append_to(partial_reader_compile);
                TensorAccessorArgs(*a2).append_to(partial_reader_compile);
                TensorAccessorArgs(*corner_a).append_to(partial_reader_compile);
            } else {
                partial_reader_compile = {
                    static_cast<uint32_t>(tt::CBIndex::c_0),
                    static_cast<uint32_t>(tt::CBIndex::c_1),
                    static_cast<uint32_t>(tt::CBIndex::c_2)};
                TensorAccessorArgs(*a0).append_to(partial_reader_compile);
                TensorAccessorArgs(*a1).append_to(partial_reader_compile);
                if (!bf16x3_split_a_selective_blow) {
                    TensorAccessorArgs(*a2).append_to(partial_reader_compile);
                }
                if (!dual_reader) {
                    TensorAccessorArgs(*halo).append_to(partial_reader_compile);
                }
            }
            std::vector<uint32_t> partial_writer_compile;
            if (preexpanded_b) {
                partial_writer_compile = {
                    static_cast<uint32_t>(tt::CBIndex::c_5),
                    static_cast<uint32_t>(tt::CBIndex::c_3),
                    static_cast<uint32_t>(tt::CBIndex::c_4),
                    static_cast<uint32_t>(tt::CBIndex::c_16)};
                TensorAccessorArgs(*a2).append_to(partial_writer_compile);
                TensorAccessorArgs(*b1).append_to(partial_writer_compile);
                TensorAccessorArgs(*b2).append_to(partial_writer_compile);
                TensorAccessorArgs(*output).append_to(partial_writer_compile);
            } else if (dual_reader) {
                partial_writer_compile = bf16x3_split_a_selective_blow
                    ? std::vector<uint32_t>{
                          static_cast<uint32_t>(tt::CBIndex::c_5),
                          static_cast<uint32_t>(tt::CBIndex::c_3),
                          static_cast<uint32_t>(tt::CBIndex::c_2),
                          static_cast<uint32_t>(tt::CBIndex::c_16)}
                    : std::vector<uint32_t>{
                          static_cast<uint32_t>(tt::CBIndex::c_3),
                          static_cast<uint32_t>(tt::CBIndex::c_2),
                          static_cast<uint32_t>(tt::CBIndex::c_16)};
                if (bf16x3_split_a_selective_blow) {
                    TensorAccessorArgs(*a2).append_to(partial_writer_compile);
                }
                TensorAccessorArgs(*halo).append_to(partial_writer_compile);
                TensorAccessorArgs(*output).append_to(partial_writer_compile);
            } else {
                partial_writer_compile = {
                    static_cast<uint32_t>(tt::CBIndex::c_16)};
                if (single_stage) {
                    TensorAccessorArgs(*output).append_to(partial_writer_compile);
                } else {
                    TensorAccessorArgs(*partial).append_to(partial_writer_compile);
                }
            }
            std::vector<uint32_t> reduce_reader_compile = {
                static_cast<uint32_t>(tt::CBIndex::c_0)};
            TensorAccessorArgs(*partial).append_to(reduce_reader_compile);
            std::vector<uint32_t> reduce_writer_compile = {
                static_cast<uint32_t>(tt::CBIndex::c_16)};
            TensorAccessorArgs(*output).append_to(reduce_writer_compile);

            auto partial_reader = CreateKernel(
                partial_program,
                preexpanded_b
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_ahi_amid_preexpanded_bhi.cpp"
                : bf16x3_hifi2_exact_leading
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_a_bhi_hifi2_exact.cpp"
                    : bf16x3_corner_constant_selective_blow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_a_bhi_corner_constant.cpp"
                    : bf16x3_corner_reuse_selective_blow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_a_bhi_corner_reuse.cpp"
                    : bf16x3_split_a_selective_blow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_ahi_amid_bhi.cpp"
                    : bf16x3_dual
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_a_beven.cpp"
                    : bf16x3_affine_corner_fold
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_a_bhi_affine_fold.cpp"
                    : a_reuse_profile
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_a_reuse_profile.cpp"
                    : (bf16x2_dual || bf16x3_packlow ||
                       bf16x3_selective_blow)
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_a_bhi.cpp"
                    : one_stage_fpu
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_ordered.cpp"
                    : component_pass_sfpu
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_component_pass.cpp"
                    : "tt_metal/programming_examples/spmv_mac/kernels/brick_reader.cpp",
                cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = partial_reader_compile});
            auto partial_writer = CreateKernel(
                partial_program,
                preexpanded_b
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_alow_preexpanded_bmid_blow_writer.cpp"
                : bf16x3_hifi2_exact_leading
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_bmid_hifi2_exact_writer.cpp"
                    : bf16x3_split_a_selective_blow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_alow_bmid_writer.cpp"
                    : bf16x3_dual
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_bodd_writer.cpp"
                    : (bf16x2_dual || bf16x3_packlow ||
                       bf16x3_affine_corner_fold ||
                       bf16x3_corner_constant_selective_blow ||
                       bf16x3_corner_reuse_selective_blow ||
                       bf16x3_selective_blow)
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_reader_bmid_writer.cpp"
                    : single_stage
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_writer.cpp"
                    : "tt_metal/programming_examples/spmv_mac/kernels/brick_partial_writer.cpp",
                cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = partial_writer_compile});
            auto partial_compute = CreateKernel(
                partial_program,
                preexpanded_b
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_preexpanded_b.cpp"
                : bf16x3_hifi2_exact_leading
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_hifi2_exact.cpp"
                    : bf16x3_corner_constant_selective_blow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_corner_constant_selective_blow.cpp"
                    : bf16x3_corner_reuse_selective_blow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_corner_reuse_selective_blow.cpp"
                    : bf16x3_split_a_selective_blow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_split_a_selective_blow.cpp"
                    : bf16x3_affine_corner_fold
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_affine_corner_fold.cpp"
                    : bf16x3_selective_blow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_selective_blow.cpp"
                    : bf16x3_packlow
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_packlow.cpp"
                    : bf16x3_dual
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x3_dual.cpp"
                    : bf16x2_dual
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_bf16x2_dual.cpp"
                    : one_stage_fpu
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_one_stage.cpp"
                    : component_pass_sfpu
                    ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_component_pass_sfpu.cpp"
                    : (term_sfpu_accum
                           ? "tt_metal/programming_examples/spmv_mac/kernels/brick_compute_term_sfpu.cpp"
                           : "tt_metal/programming_examples/spmv_mac/kernels/brick_compute.cpp"),
                cores,
                ComputeConfig{
                    .math_fidelity = bf16x3_hifi2_exact_leading
                        ? MathFidelity::HiFi2
                        : MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true,
                    .math_approx_mode = false,
                    .compile_args = {
                        single_stage
                            ? term_count
                            : first_chunk_terms,
                        term_sfpu_accum ? sfpu_batch_tiles : 1u}});

            auto reduce_reader = CreateKernel(
                reduce_program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_partial_reader.cpp",
                cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = reduce_reader_compile});
            auto reduce_writer = CreateKernel(
                reduce_program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_writer.cpp",
                cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = reduce_writer_compile});
            auto reduce_compute = CreateKernel(
                reduce_program,
                "tt_metal/programming_examples/spmv_mac/kernels/brick_reduce.cpp",
                cores,
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true,
                    .math_approx_mode = false,
                    .compile_args = {reduce_mode}});

            uint32_t start_group = 0;
            uint32_t core_index = 0;
            for (auto assignment : {
                     std::make_pair(group_a, count_a),
                     std::make_pair(group_b, count_b)}) {
                for (const auto& range : assignment.first.ranges()) {
                    for (const auto& core : range) {
                        const uint32_t count = assignment.second;
                        std::vector<uint32_t> reader_runtime_args =
                            preexpanded_b
                            ? std::vector<uint32_t>{
                                  static_cast<uint32_t>(a0->address()),
                                  static_cast<uint32_t>(a1->address()),
                                  static_cast<uint32_t>(b0->address())}
                            : bf16x3_corner_constant_selective_blow
                            ? std::vector<uint32_t>{
                                  static_cast<uint32_t>(a0->address()),
                                  static_cast<uint32_t>(a1->address()),
                                  static_cast<uint32_t>(a2->address()),
                                  static_cast<uint32_t>(corner_a->address())}
                            : bf16x3_split_a_selective_blow
                            ? std::vector<uint32_t>{
                                  static_cast<uint32_t>(a0->address()),
                                  static_cast<uint32_t>(a1->address())}
                            : std::vector<uint32_t>{
                                  static_cast<uint32_t>(a0->address()),
                                  static_cast<uint32_t>(a1->address()),
                                  static_cast<uint32_t>(a2->address())};
                        if (dual_reader) {
                            reader_runtime_args.insert(
                                reader_runtime_args.end(),
                                {count, start_group, term_order_mode});
                        } else {
                            reader_runtime_args.insert(
                                reader_runtime_args.end(),
                                {
                                    static_cast<uint32_t>(halo->address()),
                                    count,
                                    start_group,
                                    core_index,
                                });
                        }
                        if (one_stage_fpu && !dual_reader) {
                            reader_runtime_args.push_back(term_order_mode);
                        }
                        SetRuntimeArgs(
                            partial_program,
                            partial_reader,
                            core,
                            reader_runtime_args);
                        SetRuntimeArgs(
                            partial_program,
                            partial_compute,
                            core,
                            (bf16x3_packlow || bf16x3_selective_blow ||
                             bf16x3_corner_constant_selective_blow ||
                             bf16x3_corner_reuse_selective_blow ||
                             bf16x3_split_a_selective_blow ||
                             bf16x3_affine_corner_fold ||
                             bf16x3_hifi2_exact_leading)
                                ? std::vector<uint32_t>{count, term_order_mode}
                                : std::vector<uint32_t>{count});
                        if (dual_reader) {
                            const std::vector<uint32_t> writer_runtime_args =
                                preexpanded_b
                                ? std::vector<uint32_t>{
                                      static_cast<uint32_t>(a2->address()),
                                      static_cast<uint32_t>(b1->address()),
                                      static_cast<uint32_t>(b2->address()),
                                      static_cast<uint32_t>(output->address()),
                                      count,
                                      start_group,
                                      term_order_mode}
                                : bf16x3_split_a_selective_blow
                                ? std::vector<uint32_t>{
                                      static_cast<uint32_t>(a2->address()),
                                      static_cast<uint32_t>(halo->address()),
                                      static_cast<uint32_t>(output->address()),
                                      count,
                                      start_group,
                                      core_index,
                                      term_order_mode}
                                : std::vector<uint32_t>{
                                      static_cast<uint32_t>(halo->address()),
                                      static_cast<uint32_t>(output->address()),
                                      count,
                                      start_group,
                                      core_index,
                                      term_order_mode};
                            SetRuntimeArgs(
                                partial_program,
                                partial_writer,
                                core,
                                writer_runtime_args);
                        } else {
                            SetRuntimeArgs(
                                partial_program,
                                partial_writer,
                                core,
                                {
                                    static_cast<uint32_t>(
                                        single_stage
                                            ? output->address()
                                            : partial->address()),
                                    count,
                                    start_group,
                                });
                        }
                        SetRuntimeArgs(
                            reduce_program,
                            reduce_reader,
                            core,
                            {
                                static_cast<uint32_t>(partial->address()),
                                count,
                                start_group,
                            });
                        SetRuntimeArgs(
                            reduce_program,
                            reduce_compute,
                            core,
                            {count});
                        SetRuntimeArgs(
                            reduce_program,
                            reduce_writer,
                            core,
                            {
                                static_cast<uint32_t>(output->address()),
                                count,
                                start_group,
                            });
                        start_group += count;
                        ++core_index;
                    }
                }
            }
            if (ncores != kCoresPerChip ||
                core_index != kCoresPerChip ||
                start_group != kGroupsPerChip) {
                throw std::runtime_error("unexpected per-core work split");
            }
            distributed::MeshCoordinate coordinate(0, chip);
            partial_workload.add_program(
                distributed::MeshCoordinateRange(coordinate, coordinate),
                std::move(partial_program));
            reduce_workload.add_program(
                distributed::MeshCoordinateRange(coordinate, coordinate),
                std::move(reduce_program));
        }
        const auto build_end = std::chrono::high_resolution_clock::now();
        std::printf(
            "PROGRAM_BUILD host_ms=%.3f stages=%u partial_tiles_per_chip=%u\n",
            elapsed_ms(build_start, build_end),
            single_stage ? 1u : 2u,
            kPartialTilesPerChip);

        auto source_reference = read_segment<float>(
            reference_path,
            0,
            static_cast<uint64_t>(kChips) * output_elements_per_chip);
        std::vector<float> reference(source_reference.size());
        for (uint32_t chip = 0; chip < kChips; ++chip) {
            const uint32_t source_chip = source_chip_for(chip);
            std::copy_n(
                source_reference.begin() +
                    static_cast<size_t>(source_chip) * output_elements_per_chip,
                output_elements_per_chip,
                reference.begin() + static_cast<size_t>(chip) * output_elements_per_chip);
        }
        const auto enqueue_apply = [&]() {
            distributed::EnqueueMeshWorkload(queue, partial_workload, false);
            if (!single_stage) {
                distributed::EnqueueMeshWorkload(queue, reduce_workload, false);
            }
            distributed::Finish(queue);
        };
        enqueue_apply();
        if (!single_stage &&
            std::getenv("BRICK_DIAG_PARTIALS") != nullptr) {
            print_partial_diagnostics(queue, partial);
        }
        auto warm_candidate = read_output(queue, output);
        print_target_outputs(warm_candidate);
        auto warm_candidate_reread = read_output(queue, output);
        std::printf(
            "OUTPUT_REREAD bitwise_mismatches=%llu\n",
            static_cast<unsigned long long>(
                bitwise_output_mismatches(warm_candidate, warm_candidate_reread)));
        const auto warm_metrics = compare(warm_candidate, reference);
        std::printf(
            "WARMUP correct=%d l2_rel=%.9e max_rel=%.9e nonfinite=%llu\n",
            warm_metrics.pass ? 1 : 0,
            warm_metrics.l2_relative,
            warm_metrics.max_relative,
            static_cast<unsigned long long>(warm_metrics.nonfinite));
        if (!warm_metrics.pass) {
            print_error_diagnostics(warm_candidate, reference);
            if (!allow_incorrect_profile) {
                device->close();
                return 2;
            }
            std::printf(
                "DIAGNOSTIC continuing_after_incorrect_warmup=1 "
                "gate_eligible=0\n");
        }

        std::vector<double> samples;
        bool all_correct = true;
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
            const auto start = std::chrono::high_resolution_clock::now();
            enqueue_apply();
            const auto end = std::chrono::high_resolution_clock::now();
            const double milliseconds = elapsed_ms(start, end);
            auto candidate = read_output(queue, output);
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
            R"(RESULT_JSON {"schema":"tt_gmg_brick_spmv_v1","samples_seconds":[)");
        for (size_t index = 0; index < samples.size(); ++index) {
            std::printf("%s%.9f", index == 0 ? "" : ",", samples[index]);
        }
        std::printf(
            R"(],"median_seconds":%.9f,"all_correct":%s,"g3_budget_seconds":0.003,"g3_pass":%s})"
            "\n",
            median_seconds,
            all_correct ? "true" : "false",
            g3 ? "true" : "false");
        device->close();
        return all_correct || allow_incorrect_profile ? 0 : 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "brick_spmv fatal: %s\n", error.what());
        return 1;
    }
}
