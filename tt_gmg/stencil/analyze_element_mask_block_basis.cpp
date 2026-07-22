// Audit the row236 assembled 3x3 block stencil against the eight incident
// hexahedron-presence bits encoded by its exact corner coefficients.
//
// This is host-only evidence.  It never opens a Tenstorrent device.  For a
// node-to-neighbour offset, only incident elements compatible with that
// offset can contribute.  The program groups every active coefficient block
// by that compatible element submask, measures within-group numerical spread,
// and checks whether each grouped block is the sum of its singleton-element
// blocks.  That tells us whether an exact/controlled matrix-free basis exists
// before any device kernel is designed.

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint32_t kChips = 8;
constexpr uint32_t kGroupsPerChip = 198;
constexpr uint32_t kTerms = 81;
constexpr uint32_t kComponents = 3;
constexpr uint32_t kLanes = 1024;
constexpr uint32_t kOffsets = 27;
constexpr uint32_t kEntries = 9;
constexpr uint32_t kCornerOffsets[8] = {0, 2, 6, 8, 18, 20, 24, 26};

struct Mapping {
    int fd = -1;
    size_t bytes = 0;
    const uint16_t* words = nullptr;

    explicit Mapping(const std::filesystem::path& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("open failed for " + path.string() + ": " + std::strerror(errno));
        }
        struct stat status {};
        if (::fstat(fd, &status) != 0) {
            throw std::runtime_error("fstat failed for " + path.string());
        }
        bytes = static_cast<size_t>(status.st_size);
        void* address = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (address == MAP_FAILED) {
            throw std::runtime_error("mmap failed for " + path.string());
        }
        words = static_cast<const uint16_t*>(address);
    }

    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;

    ~Mapping() {
        if (words != nullptr) {
            ::munmap(const_cast<uint16_t*>(words), bytes);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

inline size_t index_of(
    uint32_t chip_group,
    uint32_t term,
    uint32_t output_component,
    uint32_t lane) {
    return ((((static_cast<size_t>(chip_group) * kTerms + term) * kComponents +
              output_component) *
             kLanes) +
            lane);
}

inline bool nonzero_bf16(uint16_t bits) {
    return (bits & 0x7fffu) != 0;
}

inline float bf16_value(uint16_t bits) {
    uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

inline std::array<int, 3> offset_vector(uint32_t offset) {
    return {
        static_cast<int>(offset / 9) - 1,
        static_cast<int>((offset / 3) % 3) - 1,
        static_cast<int>(offset % 3) - 1,
    };
}

inline std::array<int, 3> corner_vector(uint32_t corner) {
    const uint32_t offset = kCornerOffsets[corner];
    return offset_vector(offset);
}

inline bool retains_blow(uint32_t term) {
    const uint32_t offset_id = term / 3;
    const uint32_t di = offset_id / 9;
    const uint32_t dj = (offset_id / 3) % 3;
    const uint32_t dk = offset_id % 3;
    const uint32_t manhattan =
        (di > 1 ? di - 1 : 1 - di) +
        (dj > 1 ? dj - 1 : 1 - dj) +
        (dk > 1 ? dk - 1 : 1 - dk);
    return manhattan <= 2 || offset_id == 6 || offset_id == 20;
}

template <typename T>
void write_binary(
    const std::filesystem::path& path,
    const T* data,
    size_t count) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("open for write failed: " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!output) {
        throw std::runtime_error("write failed: " + path.string());
    }
}

struct PatternStats {
    uint64_t count = 0;
    std::array<long double, kEntries> sum{};
    std::array<double, kEntries> minimum{};
    std::array<double, kEntries> maximum{};

    PatternStats() {
        minimum.fill(std::numeric_limits<double>::infinity());
        maximum.fill(-std::numeric_limits<double>::infinity());
    }

    void add(const std::array<double, kEntries>& matrix) {
        ++count;
        for (uint32_t entry = 0; entry < kEntries; ++entry) {
            sum[entry] += matrix[entry];
            minimum[entry] = std::min(minimum[entry], matrix[entry]);
            maximum[entry] = std::max(maximum[entry], matrix[entry]);
        }
    }

    double mean(uint32_t entry) const {
        return count == 0 ? 0.0 : static_cast<double>(sum[entry] / count);
    }
};

struct OffsetStats {
    std::array<int, 3> offset{};
    std::array<uint8_t, 8> compatible_corner{};
    uint32_t compatible_count = 0;
    std::vector<PatternStats> patterns;
};

struct QuantizedMatrix {
    std::array<int32_t, kEntries> values{};

    bool operator==(const QuantizedMatrix&) const = default;
};

struct QuantizedMatrixHash {
    size_t operator()(const QuantizedMatrix& matrix) const noexcept {
        size_t hash = 0xcbf29ce484222325ull;
        for (const int32_t value : matrix.values) {
            hash ^= static_cast<uint32_t>(value);
            hash *= 0x100000001b3ull;
        }
        return hash;
    }
};

struct QuantizedStencilSignature {
    std::array<int32_t, kOffsets * kEntries> values{};

    bool operator==(const QuantizedStencilSignature&) const = default;
};

struct QuantizedStencilSignatureHash {
    size_t operator()(const QuantizedStencilSignature& stencil) const noexcept {
        size_t hash = 0xcbf29ce484222325ull;
        for (const int32_t value : stencil.values) {
            hash ^= static_cast<uint32_t>(value);
            hash *= 0x100000001b3ull;
        }
        return hash;
    }
};

// The quantized/reconstructed matrix audit answers the numerical question,
// but a lossless on-device dictionary must preserve the three BF16 source
// words for every scalar.  Keep that representation separate so a small
// reconstructed-value cardinality is never mistaken for bitwise compression
// authority.
struct ExactSplitBlock {
    std::array<uint16_t, kEntries * 3> words{};

    bool operator==(const ExactSplitBlock&) const = default;
};

struct ExactSplitBlockHash {
    size_t operator()(const ExactSplitBlock& block) const noexcept {
        size_t hash = 0xcbf29ce484222325ull;
        for (const uint16_t word : block.words) {
            hash ^= word;
            hash *= 0x100000001b3ull;
        }
        return hash;
    }
};

constexpr std::array<double, 4> kQuantums = {
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path root =
            argc > 1 ? argv[1] : "/home/ttuser/ttgmg/device_v1";
        const std::filesystem::path export_root =
            argc > 2 ? argv[2] : std::filesystem::path{};
        std::array<Mapping, 3> splits = {
            Mapping(root / "operator_a0.bf16"),
            Mapping(root / "operator_a1.bf16"),
            Mapping(root / "operator_a2.bf16"),
        };
        const size_t expected_words =
            static_cast<size_t>(kChips) * kGroupsPerChip * kTerms *
            kComponents * kLanes;
        for (const auto& split : splits) {
            if (split.bytes != expected_words * sizeof(uint16_t)) {
                throw std::runtime_error("unexpected operator split size");
            }
        }

        std::array<OffsetStats, kOffsets> offsets;
        for (uint32_t oid = 0; oid < kOffsets; ++oid) {
            auto& stats = offsets[oid];
            stats.offset = offset_vector(oid);
            for (uint32_t corner = 0; corner < 8; ++corner) {
                const auto direction = corner_vector(corner);
                bool compatible = true;
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    compatible = compatible &&
                        (stats.offset[axis] == 0 ||
                         stats.offset[axis] == direction[axis]);
                }
                if (compatible) {
                    stats.compatible_corner[stats.compatible_count++] =
                        static_cast<uint8_t>(corner);
                }
            }
            stats.patterns.resize(size_t{1} << stats.compatible_count);
        }

        uint64_t active_lanes = 0;
        uint64_t corner_mask_histogram[256] = {};
        std::array<
            std::array<
                std::unordered_set<QuantizedMatrix, QuantizedMatrixHash>,
                kOffsets>,
            kQuantums.size()>
            quantized_by_offset;
        std::array<
            std::unordered_set<QuantizedMatrix, QuantizedMatrixHash>,
            kQuantums.size()>
            quantized_global;
        std::array<std::array<uint64_t, 10>, kOffsets>
            nonzero_entry_histogram{};
        std::array<
            std::unordered_set<ExactSplitBlock, ExactSplitBlockHash>,
            kOffsets>
            exact_by_offset;
        std::unordered_map<ExactSplitBlock, uint64_t, ExactSplitBlockHash>
            exact_global_frequency;
        std::array<std::unordered_set<int32_t>, kOffsets * kEntries>
            quantized_scalar_by_plane;
        std::unordered_set<int32_t> quantized_scalar_global;
        std::array<
            std::array<
                std::unordered_set<QuantizedMatrix, QuantizedMatrixHash>,
                256>,
            kOffsets>
            quantized_block_by_full_incident_mask;
        std::unordered_map<
            QuantizedStencilSignature,
            uint64_t,
            QuantizedStencilSignatureHash>
            quantized_stencil_frequency;
        std::array<
            std::unordered_set<
                QuantizedStencilSignature,
                QuantizedStencilSignatureHash>,
            256>
            quantized_stencils_by_incident_mask;
        std::unordered_map<
            QuantizedStencilSignature,
            uint16_t,
            QuantizedStencilSignatureHash>
            quantized_stencil_id;
        std::vector<uint16_t> stencil_id_by_lane(
            static_cast<size_t>(kChips) * kGroupsPerChip * kLanes,
            std::numeric_limits<uint16_t>::max());
        uint64_t structural_scalar_nonzeros = 0;
        const uint32_t chip_groups = kChips * kGroupsPerChip;
        for (uint32_t chip_group = 0; chip_group < chip_groups; ++chip_group) {
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                const size_t center_index = index_of(chip_group, 13 * 3, 0, lane);
                bool center_nonzero = false;
                for (uint32_t split = 0; split < 3; ++split) {
                    center_nonzero = center_nonzero ||
                        nonzero_bf16(splits[split].words[center_index]);
                }
                if (!center_nonzero) {
                    continue;
                }
                ++active_lanes;

                uint32_t element_mask = 0;
                for (uint32_t corner = 0; corner < 8; ++corner) {
                    const uint32_t term = kCornerOffsets[corner] * 3;
                    const size_t index = index_of(chip_group, term, 0, lane);
                    bool present = false;
                    for (uint32_t split = 0; split < 3; ++split) {
                        present = present ||
                            nonzero_bf16(splits[split].words[index]);
                    }
                    element_mask |= present ? (1u << corner) : 0u;
                }
                ++corner_mask_histogram[element_mask];

                QuantizedStencilSignature stencil_signature;
                for (uint32_t oid = 0; oid < kOffsets; ++oid) {
                    auto& stats = offsets[oid];
                    uint32_t submask = 0;
                    for (uint32_t local = 0;
                         local < stats.compatible_count;
                         ++local) {
                        const uint32_t corner = stats.compatible_corner[local];
                        submask |= ((element_mask >> corner) & 1u) << local;
                    }

                    std::array<double, kEntries> matrix{};
                    ExactSplitBlock exact_block;
                    for (uint32_t output = 0; output < 3; ++output) {
                        for (uint32_t input = 0; input < 3; ++input) {
                            const size_t index = index_of(
                                chip_group,
                                oid * 3 + input,
                                output,
                                lane);
                            double value = 0.0;
                            for (uint32_t split = 0; split < 3; ++split) {
                                const uint16_t word = splits[split].words[index];
                                value += bf16_value(word);
                                exact_block.words[
                                    (output * 3 + input) * 3 + split] = word;
                            }
                            matrix[output * 3 + input] = value;
                        }
                    }
                    exact_by_offset[oid].insert(exact_block);
                    ++exact_global_frequency[exact_block];
                    stats.patterns[submask].add(matrix);
                    uint32_t nonzero_entries = 0;
                    for (const double value : matrix) {
                        nonzero_entries += std::abs(value) > 1.0e-5;
                    }
                    for (uint32_t entry = 0; entry < kEntries; ++entry) {
                        const int32_t quantized_scalar = static_cast<int32_t>(
                            std::llround(matrix[entry] / kQuantums[0]));
                        quantized_scalar_by_plane[oid * kEntries + entry]
                            .insert(quantized_scalar);
                        quantized_scalar_global.insert(quantized_scalar);
                    }
                    structural_scalar_nonzeros += nonzero_entries;
                    ++nonzero_entry_histogram[oid][nonzero_entries];
                    for (uint32_t quantization = 0;
                         quantization < kQuantums.size();
                         ++quantization) {
                        QuantizedMatrix key;
                        for (uint32_t entry = 0;
                             entry < kEntries;
                             ++entry) {
                            key.values[entry] = static_cast<int32_t>(
                                std::llround(
                                    matrix[entry] /
                                    kQuantums[quantization]));
                        }
                        quantized_by_offset[quantization][oid].insert(key);
                        quantized_global[quantization].insert(key);
                        if (quantization == 0) {
                            for (uint32_t entry = 0;
                                 entry < kEntries;
                                 ++entry) {
                                stencil_signature.values[oid * kEntries + entry] =
                                    key.values[entry];
                            }
                            quantized_block_by_full_incident_mask[oid]
                                [element_mask]
                                    .insert(key);
                        }
                    }
                }
                ++quantized_stencil_frequency[stencil_signature];
                quantized_stencils_by_incident_mask[element_mask].insert(
                    stencil_signature);
                const auto [stencil_id_it, inserted] =
                    quantized_stencil_id.emplace(
                        stencil_signature,
                        static_cast<uint16_t>(quantized_stencil_id.size()));
                if (inserted && quantized_stencil_id.size() >
                        std::numeric_limits<uint16_t>::max()) {
                    throw std::runtime_error(
                        "canonical stencil id exceeds uint16 range");
                }
                stencil_id_by_lane[
                    static_cast<size_t>(chip_group) * kLanes + lane] =
                    stencil_id_it->second;
            }
        }

        std::printf("schema=tt_gmg_element_mask_block_basis_v1\n");
        std::printf("active_lanes=%llu\n", static_cast<unsigned long long>(active_lanes));
        uint32_t observed_node_masks = 0;
        for (uint32_t mask = 0; mask < 256; ++mask) {
            observed_node_masks += corner_mask_histogram[mask] != 0;
        }
        std::printf("observed_incident_element_masks=%u\n", observed_node_masks);

        uint64_t observed_patterns = 0;
        double global_max_span = 0.0;
        double global_max_additivity_error = 0.0;
        for (uint32_t oid = 0; oid < kOffsets; ++oid) {
            const auto& stats = offsets[oid];
            uint32_t observed = 0;
            double max_span = 0.0;
            double max_additivity_error = 0.0;
            for (uint32_t mask = 0; mask < stats.patterns.size(); ++mask) {
                const auto& pattern = stats.patterns[mask];
                if (pattern.count == 0) {
                    continue;
                }
                ++observed;
                for (uint32_t entry = 0; entry < kEntries; ++entry) {
                    max_span = std::max(
                        max_span,
                        pattern.maximum[entry] - pattern.minimum[entry]);
                    double singleton_sum = 0.0;
                    for (uint32_t bit = 0;
                         bit < stats.compatible_count;
                         ++bit) {
                        if ((mask & (1u << bit)) == 0) {
                            continue;
                        }
                        const auto& singleton = stats.patterns[1u << bit];
                        if (singleton.count != 0) {
                            singleton_sum += singleton.mean(entry);
                        }
                    }
                    max_additivity_error = std::max(
                        max_additivity_error,
                        std::abs(pattern.mean(entry) - singleton_sum));
                }
            }
            observed_patterns += observed;
            global_max_span = std::max(global_max_span, max_span);
            global_max_additivity_error = std::max(
                global_max_additivity_error,
                max_additivity_error);
            std::printf(
                "offset=%u vector=%d,%d,%d compatible_elements=%u "
                "observed_patterns=%u max_within_pattern_span=%.9e "
                "max_singleton_additivity_error=%.9e "
                "unique_q1e-7=%zu unique_q1e-6=%zu unique_q1e-5=%zu "
                "unique_q1e-4=%zu unique_exact_bf16x3=%zu nonzero_hist=",
                oid,
                stats.offset[0],
                stats.offset[1],
                stats.offset[2],
                stats.compatible_count,
                observed,
                max_span,
                max_additivity_error,
                quantized_by_offset[0][oid].size(),
                quantized_by_offset[1][oid].size(),
                quantized_by_offset[2][oid].size(),
                quantized_by_offset[3][oid].size(),
                exact_by_offset[oid].size());
            for (uint32_t count = 0; count <= 9; ++count) {
                std::printf(
                    "%s%u:%llu",
                    count == 0 ? "" : ",",
                    count,
                    static_cast<unsigned long long>(
                        nonzero_entry_histogram[oid][count]));
            }
            std::printf("\n");
        }
        std::printf("observed_offset_patterns=%llu\n", static_cast<unsigned long long>(observed_patterns));
        std::printf("global_max_within_pattern_span=%.9e\n", global_max_span);
        std::printf("global_max_singleton_additivity_error=%.9e\n", global_max_additivity_error);
        for (uint32_t quantization = 0;
             quantization < kQuantums.size();
             ++quantization) {
            std::printf(
                "global_unique_quantized quantum=%.1e count=%zu\n",
                kQuantums[quantization],
                quantized_global[quantization].size());
        }
        const uint64_t dense_structural_scalars =
            active_lanes * kOffsets * kEntries;
        const uint64_t zero_structural_scalars =
            dense_structural_scalars - structural_scalar_nonzeros;
        long double entropy_bits = 0.0L;
        uint64_t most_frequent_count = 0;
        for (const auto& [block, count] : exact_global_frequency) {
            (void)block;
            most_frequent_count = std::max(most_frequent_count, count);
            const long double probability =
                static_cast<long double>(count) /
                static_cast<long double>(active_lanes * kOffsets);
            entropy_bits -= probability * std::log2(probability);
        }
        std::printf(
            "global_unique_exact_bf16x3 count=%zu\n",
            exact_global_frequency.size());
        std::printf(
            "exact_dictionary_entropy_bits_per_active_block=%.9Lf "
            "most_frequent_count=%llu\n",
            entropy_bits,
            static_cast<unsigned long long>(most_frequent_count));
        std::printf(
            "structural_scalar_nonzeros=%llu dense_structural_scalars=%llu "
            "structural_scalar_zeros=%llu nonzero_fraction=%.9f "
            "zero_fraction=%.9f\n",
            static_cast<unsigned long long>(structural_scalar_nonzeros),
            static_cast<unsigned long long>(dense_structural_scalars),
            static_cast<unsigned long long>(zero_structural_scalars),
            static_cast<double>(structural_scalar_nonzeros) /
                static_cast<double>(dense_structural_scalars),
            static_cast<double>(zero_structural_scalars) /
                static_cast<double>(dense_structural_scalars));
        uint32_t scalar_plane_minimum = std::numeric_limits<uint32_t>::max();
        uint32_t scalar_plane_maximum = 0;
        uint64_t scalar_plane_sum = 0;
        std::array<uint32_t, 64> scalar_plane_histogram{};
        for (uint32_t oid = 0; oid < kOffsets; ++oid) {
            std::printf("quantized_scalar_plane_uniques offset=%u counts=", oid);
            for (uint32_t entry = 0; entry < kEntries; ++entry) {
                const uint32_t count = static_cast<uint32_t>(
                    quantized_scalar_by_plane[oid * kEntries + entry].size());
                scalar_plane_minimum = std::min(scalar_plane_minimum, count);
                scalar_plane_maximum = std::max(scalar_plane_maximum, count);
                scalar_plane_sum += count;
                ++scalar_plane_histogram[
                    std::min<uint32_t>(
                        count,
                        static_cast<uint32_t>(scalar_plane_histogram.size() - 1))];
                std::printf("%s%u", entry == 0 ? "" : ",", count);
            }
            std::printf("\n");
        }
        std::printf(
            "quantized_scalar_dictionary quantum=%.1e global_unique=%zu "
            "plane_min=%u plane_max=%u plane_mean=%.9f\n",
            kQuantums[0],
            quantized_scalar_global.size(),
            scalar_plane_minimum,
            scalar_plane_maximum,
            static_cast<double>(scalar_plane_sum) /
                static_cast<double>(kOffsets * kEntries));
        std::printf("quantized_scalar_plane_unique_hist=");
        bool first_histogram_entry = true;
        for (uint32_t count = 0;
             count < scalar_plane_histogram.size();
             ++count) {
            if (scalar_plane_histogram[count] == 0) {
                continue;
            }
            std::printf(
                "%s%u:%u",
                first_histogram_entry ? "" : ",",
                count,
                scalar_plane_histogram[count]);
            first_histogram_entry = false;
        }
        std::printf("\n");
        uint32_t full_mask_observed_pairs = 0;
        uint32_t full_mask_ambiguous_pairs = 0;
        uint32_t full_mask_max_block_types = 0;
        for (uint32_t oid = 0; oid < kOffsets; ++oid) {
            uint32_t observed_pairs = 0;
            uint32_t ambiguous_pairs = 0;
            uint32_t max_block_types = 0;
            for (uint32_t mask = 0; mask < 256; ++mask) {
                const uint32_t block_types = static_cast<uint32_t>(
                    quantized_block_by_full_incident_mask[oid][mask].size());
                observed_pairs += block_types != 0;
                ambiguous_pairs += block_types > 1;
                max_block_types = std::max(max_block_types, block_types);
            }
            full_mask_observed_pairs += observed_pairs;
            full_mask_ambiguous_pairs += ambiguous_pairs;
            full_mask_max_block_types = std::max(
                full_mask_max_block_types,
                max_block_types);
            std::printf(
                "full_incident_mask_predictor offset=%u observed_masks=%u "
                "ambiguous_masks=%u max_block_types_per_mask=%u\n",
                oid,
                observed_pairs,
                ambiguous_pairs,
                max_block_types);
        }
        std::printf(
            "full_incident_mask_predictor_summary quantum=%.1e "
            "observed_offset_mask_pairs=%u ambiguous_offset_mask_pairs=%u "
            "max_block_types_per_mask=%u deterministic=%u\n",
            kQuantums[0],
            full_mask_observed_pairs,
            full_mask_ambiguous_pairs,
            full_mask_max_block_types,
            full_mask_ambiguous_pairs == 0 ? 1u : 0u);
        uint32_t observed_signature_masks = 0;
        uint32_t ambiguous_signature_masks = 0;
        uint32_t max_signatures_per_mask = 0;
        for (uint32_t mask = 0; mask < 256; ++mask) {
            const uint32_t signatures = static_cast<uint32_t>(
                quantized_stencils_by_incident_mask[mask].size());
            observed_signature_masks += signatures != 0;
            ambiguous_signature_masks += signatures > 1;
            max_signatures_per_mask = std::max(
                max_signatures_per_mask,
                signatures);
        }
        long double stencil_entropy_bits = 0.0L;
        uint64_t most_frequent_stencil_count = 0;
        for (const auto& [stencil, count] : quantized_stencil_frequency) {
            (void)stencil;
            most_frequent_stencil_count = std::max(
                most_frequent_stencil_count,
                count);
            const long double probability =
                static_cast<long double>(count) /
                static_cast<long double>(active_lanes);
            stencil_entropy_bits -= probability * std::log2(probability);
        }
        std::printf(
            "canonical_stencil_dictionary quantum=%.1e unique=%zu "
            "entropy_bits_per_active_node=%.9Lf most_frequent_count=%llu\n",
            kQuantums[0],
            quantized_stencil_frequency.size(),
            stencil_entropy_bits,
            static_cast<unsigned long long>(most_frequent_stencil_count));
        std::printf(
            "canonical_stencil_by_incident_mask observed_masks=%u "
            "ambiguous_masks=%u max_signatures_per_mask=%u deterministic=%u\n",
            observed_signature_masks,
            ambiguous_signature_masks,
            max_signatures_per_mask,
            ambiguous_signature_masks == 0 ? 1u : 0u);
        const auto base_stencil_it = std::max_element(
            quantized_stencil_frequency.begin(),
            quantized_stencil_frequency.end(),
            [](const auto& left, const auto& right) {
                return left.second < right.second;
            });
        if (base_stencil_it == quantized_stencil_frequency.end()) {
            throw std::runtime_error("no canonical stencil signature found");
        }
        const auto base_id_it = quantized_stencil_id.find(
            base_stencil_it->first);
        if (base_id_it == quantized_stencil_id.end()) {
            throw std::runtime_error("canonical base stencil id is absent");
        }
        const uint16_t base_stencil_id = base_id_it->second;
        uint64_t correction_scalar_lanes = 0;
        uint64_t nonbase_nodes = 0;
        std::array<uint64_t, kOffsets * kEntries + 1>
            correction_count_histogram{};
        std::vector<uint64_t> stencil_frequencies;
        stencil_frequencies.reserve(quantized_stencil_frequency.size());
        for (const auto& [stencil, count] : quantized_stencil_frequency) {
            stencil_frequencies.push_back(count);
            uint32_t correction_count = 0;
            for (uint32_t scalar = 0;
                 scalar < kOffsets * kEntries;
                 ++scalar) {
                correction_count +=
                    stencil.values[scalar] !=
                    base_stencil_it->first.values[scalar];
            }
            correction_scalar_lanes += count * correction_count;
            correction_count_histogram[correction_count] += count;
            nonbase_nodes += correction_count == 0 ? 0 : count;
        }
        std::sort(
            stencil_frequencies.begin(),
            stencil_frequencies.end(),
            std::greater<uint64_t>());
        uint32_t base_nonzero_scalars = 0;
        std::unordered_set<int32_t> base_scalar_palette;
        uint32_t base_active_terms = 0;
        uint32_t base_active_blow_terms = 0;
        uint32_t base_active_no_blow_terms = 0;
        uint32_t base_product_count = 0;
        std::array<uint32_t, 4> base_nonzero_outputs_per_term_histogram{};
        for (const int32_t value : base_stencil_it->first.values) {
            base_nonzero_scalars += value != 0;
            base_scalar_palette.insert(value);
        }
        for (uint32_t term = 0; term < kTerms; ++term) {
            const uint32_t offset = term / kComponents;
            const uint32_t input = term % kComponents;
            uint32_t nonzero_outputs = 0;
            for (uint32_t output = 0; output < kComponents; ++output) {
                const uint32_t scalar =
                    offset * kEntries + output * kComponents + input;
                nonzero_outputs +=
                    base_stencil_it->first.values[scalar] != 0;
            }
            ++base_nonzero_outputs_per_term_histogram[nonzero_outputs];
            if (nonzero_outputs == 0) {
                continue;
            }
            ++base_active_terms;
            const bool retain_low = retains_blow(term);
            base_active_blow_terms += retain_low ? 1u : 0u;
            base_active_no_blow_terms += retain_low ? 0u : 1u;
            base_product_count += nonzero_outputs * (retain_low ? 6u : 5u);
        }
        const uint64_t dense_canonical_scalar_lanes =
            active_lanes * kOffsets * kEntries;
        uint32_t dense_blow_terms = 0;
        for (uint32_t term = 0; term < kTerms; ++term) {
            dense_blow_terms += retains_blow(term) ? 1u : 0u;
        }
        const uint32_t dense_no_blow_terms = kTerms - dense_blow_terms;
        const uint32_t dense_product_count =
            kComponents * (dense_blow_terms * 6u + dense_no_blow_terms * 5u);
        const uint64_t base_nodes = base_stencil_it->second;
        const uint64_t base_tiles = (base_nodes + kLanes - 1) / kLanes;
        const uint64_t nonbase_tiles = (nonbase_nodes + kLanes - 1) / kLanes;
        const uint64_t packed_tiles = base_tiles + nonbase_tiles;
        const uint64_t current_tiles = chip_groups;
        const uint64_t base_padding_lanes = base_tiles * kLanes - base_nodes;
        const uint64_t nonbase_padding_lanes =
            nonbase_tiles * kLanes - nonbase_nodes;
        std::printf(
            "canonical_base_stencil nodes=%llu node_fraction=%.9f "
            "nonzero_scalars=%u scalar_palette=%zu nonbase_nodes=%llu\n",
            static_cast<unsigned long long>(base_stencil_it->second),
            static_cast<double>(base_stencil_it->second) /
                static_cast<double>(active_lanes),
            base_nonzero_scalars,
            base_scalar_palette.size(),
            static_cast<unsigned long long>(nonbase_nodes));
        std::printf(
            "canonical_base_schedule active_terms=%u skipped_terms=%u "
            "retained_blow_terms=%u no_blow_terms=%u products_per_tile=%u "
            "dense_products_per_tile=%u product_reduction_fraction=%.9f "
            "nonzero_outputs_hist=0:%u,1:%u,2:%u,3:%u\n",
            base_active_terms,
            kTerms - base_active_terms,
            base_active_blow_terms,
            base_active_no_blow_terms,
            base_product_count,
            dense_product_count,
            1.0 - static_cast<double>(base_product_count) /
                static_cast<double>(dense_product_count),
            base_nonzero_outputs_per_term_histogram[0],
            base_nonzero_outputs_per_term_histogram[1],
            base_nonzero_outputs_per_term_histogram[2],
            base_nonzero_outputs_per_term_histogram[3]);
        std::printf(
            "canonical_repack_budget base_tiles=%llu nonbase_tiles=%llu "
            "packed_tiles=%llu current_tiles=%llu tile_reduction_fraction=%.9f "
            "base_padding_lanes=%llu nonbase_padding_lanes=%llu "
            "total_padding_lanes=%llu\n",
            static_cast<unsigned long long>(base_tiles),
            static_cast<unsigned long long>(nonbase_tiles),
            static_cast<unsigned long long>(packed_tiles),
            static_cast<unsigned long long>(current_tiles),
            1.0 - static_cast<double>(packed_tiles) /
                static_cast<double>(current_tiles),
            static_cast<unsigned long long>(base_padding_lanes),
            static_cast<unsigned long long>(nonbase_padding_lanes),
            static_cast<unsigned long long>(
                base_padding_lanes + nonbase_padding_lanes));
        std::printf("canonical_repack_per_chip=");
        const uint64_t base_tiles_per_chip = base_tiles / kChips;
        const uint64_t base_tiles_remainder = base_tiles % kChips;
        const uint64_t nonbase_tiles_per_chip = nonbase_tiles / kChips;
        const uint64_t nonbase_tiles_remainder = nonbase_tiles % kChips;
        for (uint32_t chip = 0; chip < kChips; ++chip) {
            const uint64_t chip_base =
                base_tiles_per_chip + (chip < base_tiles_remainder ? 1u : 0u);
            const uint64_t chip_nonbase =
                nonbase_tiles_per_chip +
                (chip < nonbase_tiles_remainder ? 1u : 0u);
            std::printf(
                "%s%u:%llu+%llu=%llu",
                chip == 0 ? "" : ",",
                chip,
                static_cast<unsigned long long>(chip_base),
                static_cast<unsigned long long>(chip_nonbase),
                static_cast<unsigned long long>(chip_base + chip_nonbase));
        }
        std::printf("\n");
        std::printf(
            "canonical_base_corrections scalar_lanes=%llu dense_scalar_lanes=%llu "
            "correction_fraction=%.9f base_match_fraction=%.9f\n",
            static_cast<unsigned long long>(correction_scalar_lanes),
            static_cast<unsigned long long>(dense_canonical_scalar_lanes),
            static_cast<double>(correction_scalar_lanes) /
                static_cast<double>(dense_canonical_scalar_lanes),
            1.0 - static_cast<double>(correction_scalar_lanes) /
                static_cast<double>(dense_canonical_scalar_lanes));
        std::printf("canonical_stencil_top_frequencies=");
        for (uint32_t rank = 0;
             rank < std::min<size_t>(10, stencil_frequencies.size());
             ++rank) {
            std::printf(
                "%s%llu",
                rank == 0 ? "" : ",",
                static_cast<unsigned long long>(stencil_frequencies[rank]));
        }
        std::printf("\ncanonical_correction_count_hist=");
        bool first_correction_histogram_entry = true;
        for (uint32_t corrections = 0;
             corrections < correction_count_histogram.size();
             ++corrections) {
            if (correction_count_histogram[corrections] == 0) {
                continue;
            }
            std::printf(
                "%s%u:%llu",
                first_correction_histogram_entry ? "" : ",",
                corrections,
                static_cast<unsigned long long>(
                    correction_count_histogram[corrections]));
            first_correction_histogram_entry = false;
        }
        std::printf("\n");
        uint32_t allbase_groups = 0;
        uint64_t allbase_group_active_nodes = 0;
        std::array<uint32_t, kLanes + 1> boundary_lanes_per_group_histogram{};
        for (uint32_t chip_group = 0;
             chip_group < chip_groups;
             ++chip_group) {
            uint32_t active_in_group = 0;
            uint32_t boundary_in_group = 0;
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                const uint16_t stencil_id = stencil_id_by_lane[
                    static_cast<size_t>(chip_group) * kLanes + lane];
                if (stencil_id == std::numeric_limits<uint16_t>::max()) {
                    continue;
                }
                ++active_in_group;
                boundary_in_group += stencil_id != base_stencil_id;
            }
            ++boundary_lanes_per_group_histogram[boundary_in_group];
            if (boundary_in_group == 0) {
                ++allbase_groups;
                allbase_group_active_nodes += active_in_group;
            }
        }
        std::printf(
            "canonical_base_group_coverage groups=%u allbase_groups=%u "
            "allbase_group_fraction=%.9f allbase_group_active_nodes=%llu "
            "active_node_fraction=%.9f\n",
            chip_groups,
            allbase_groups,
            static_cast<double>(allbase_groups) /
                static_cast<double>(chip_groups),
            static_cast<unsigned long long>(allbase_group_active_nodes),
            static_cast<double>(allbase_group_active_nodes) /
                static_cast<double>(active_lanes));
        std::printf("canonical_boundary_lanes_per_group_hist=");
        bool first_group_histogram_entry = true;
        for (uint32_t boundary_lanes = 0;
             boundary_lanes < boundary_lanes_per_group_histogram.size();
             ++boundary_lanes) {
            if (boundary_lanes_per_group_histogram[boundary_lanes] == 0) {
                continue;
            }
            std::printf(
                "%s%u:%u",
                first_group_histogram_entry ? "" : ",",
                boundary_lanes,
                boundary_lanes_per_group_histogram[boundary_lanes]);
            first_group_histogram_entry = false;
        }
        std::printf("\n");
        if (!export_root.empty()) {
            std::filesystem::create_directories(export_root);
            std::vector<uint8_t> base_lane_mask(stencil_id_by_lane.size(), 0);
            uint64_t exported_base_lanes = 0;
            for (size_t lane = 0; lane < stencil_id_by_lane.size(); ++lane) {
                const bool is_base = stencil_id_by_lane[lane] == base_stencil_id;
                base_lane_mask[lane] = is_base ? 1u : 0u;
                exported_base_lanes += is_base ? 1u : 0u;
            }
            if (exported_base_lanes != base_nodes) {
                throw std::runtime_error("exported base-lane count mismatch");
            }
            write_binary(
                export_root / "canonical_base_lane_mask.u8",
                base_lane_mask.data(),
                base_lane_mask.size());
            write_binary(
                export_root / "canonical_base_signature_q1e7.i32",
                base_stencil_it->first.values.data(),
                base_stencil_it->first.values.size());
            std::ofstream metadata(
                export_root / "canonical_base_metadata.json",
                std::ios::trunc);
            if (!metadata) {
                throw std::runtime_error("open canonical metadata failed");
            }
            metadata
                << "{\n"
                << "  \"schema\": \"tt_gmg_canonical_base_export_v1\",\n"
                << "  \"quantum\": 1e-7,\n"
                << "  \"chips\": " << kChips << ",\n"
                << "  \"groups_per_chip\": " << kGroupsPerChip << ",\n"
                << "  \"lanes_per_group\": " << kLanes << ",\n"
                << "  \"active_lanes\": " << active_lanes << ",\n"
                << "  \"base_lanes\": " << base_nodes << ",\n"
                << "  \"nonbase_lanes\": " << nonbase_nodes << ",\n"
                << "  \"base_stencil_id\": " << base_stencil_id << ",\n"
                << "  \"base_nonzero_scalars\": "
                << base_nonzero_scalars << ",\n"
                << "  \"base_scalar_palette_size_including_zero\": "
                << base_scalar_palette.size() << ",\n"
                << "  \"base_active_terms\": " << base_active_terms << ",\n"
                << "  \"base_products_per_tile\": "
                << base_product_count << ",\n"
                << "  \"base_tiles\": " << base_tiles << ",\n"
                << "  \"nonbase_tiles\": " << nonbase_tiles << ",\n"
                << "  \"packed_tiles\": " << packed_tiles << "\n"
                << "}\n";
            if (!metadata) {
                throw std::runtime_error("write canonical metadata failed");
            }
            std::printf(
                "canonical_export root=%s base_mask_bytes=%zu signature_bytes=%zu\n",
                export_root.c_str(),
                base_lane_mask.size(),
                base_stencil_it->first.values.size() * sizeof(int32_t));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "element-mask block-basis audit failed: %s\n", error.what());
        return 1;
    }
}
