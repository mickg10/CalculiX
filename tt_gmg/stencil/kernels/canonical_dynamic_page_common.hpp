#pragma once

#include <cstdint>

namespace canonical_dynamic_page {

constexpr uint32_t kOffsets = 27;
constexpr uint32_t kComponents = 3;
constexpr uint32_t kTerms = kOffsets * kComponents;
constexpr uint32_t kTermsPerPublish = 3;
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kVectorPagesPerComponent = 180;
constexpr uint32_t kMaxGroupPages = 62;
constexpr uint32_t kPageTableBytes = 64;
constexpr uint32_t kPageCountByte = 63;
constexpr uint16_t kLaneSentinel = 65535;

// Component-major streaming lets the data-movement cores stage only one
// component's vector pages at a time.  Offsets remain reverse ordered to keep
// the Run66 cancellation order inside each component pass deterministic.
inline uint32_t ordered_term(uint32_t stream_term) {
    const uint32_t component = stream_term / kOffsets;
    const uint32_t reverse_offset = kOffsets - 1 - stream_term % kOffsets;
    return reverse_offset * kComponents + component;
}

inline bool retains_blow(uint32_t term) {
    const uint32_t offset_id = term / kComponents;
    const uint32_t di = offset_id / 9;
    const uint32_t dj = (offset_id / 3) % 3;
    const uint32_t dk = offset_id % 3;
    const uint32_t manhattan =
        (di > 1 ? di - 1 : 1 - di) +
        (dj > 1 ? dj - 1 : 1 - dj) +
        (dk > 1 ? dk - 1 : 1 - dk);
    return manhattan <= 2 || offset_id == 6 || offset_id == 20;
}

// A component-major batch can retain 1, 2, or 3 low-split tiles.  The low
// stream CB is exactly kTermsPerPublish pages, so advancing it by the compact
// count can cross the FIFO end (for example 1 then 3).  Publish a complete CB
// cycle whenever a batch has any low tiles; only the first low_count pages are
// materialized and consumed by the MAC schedule.
inline uint32_t low_publish_count(uint32_t low_count) {
    return low_count == 0 ? 0 : kTermsPerPublish;
}

// Pack four gathered BF16 words per loop.  The direct branch is the exact
// aligned consecutive case measured by the host feasibility audit; absent
// quartets become one zero store.  All other quartets are lossless scalar
// L1 gathers followed by one aligned 64-bit destination store.
inline void gather_b_tile(
    uint32_t destination,
    tt_l1_ptr uint16_t* staged,
    tt_l1_ptr uint16_t* lane_index) {
    tt_l1_ptr uint64_t* output = (tt_l1_ptr uint64_t*)destination;
    for (uint32_t word = 0; word < 256; ++word) {
        const uint32_t lane = word * 4;
        const uint16_t i0 = lane_index[lane + 0];
        const uint16_t i1 = lane_index[lane + 1];
        const uint16_t i2 = lane_index[lane + 2];
        const uint16_t i3 = lane_index[lane + 3];
        uint64_t packed = 0;
        if (i0 == kLaneSentinel && i1 == kLaneSentinel &&
            i2 == kLaneSentinel && i3 == kLaneSentinel) {
            packed = 0;
        } else if (
            i0 != kLaneSentinel && (i0 & 3u) == 0 &&
            i1 == static_cast<uint16_t>(i0 + 1) &&
            i2 == static_cast<uint16_t>(i0 + 2) &&
            i3 == static_cast<uint16_t>(i0 + 3)) {
            packed = ((tt_l1_ptr uint64_t*)staged)[i0 / 4];
        } else {
            const uint64_t v0 =
                i0 == kLaneSentinel ? 0 : static_cast<uint64_t>(staged[i0]);
            const uint64_t v1 =
                i1 == kLaneSentinel ? 0 : static_cast<uint64_t>(staged[i1]);
            const uint64_t v2 =
                i2 == kLaneSentinel ? 0 : static_cast<uint64_t>(staged[i2]);
            const uint64_t v3 =
                i3 == kLaneSentinel ? 0 : static_cast<uint64_t>(staged[i3]);
            packed = v0 | (v1 << 16) | (v2 << 32) | (v3 << 48);
        }
        output[word] = packed;
    }
}

}  // namespace canonical_dynamic_page
