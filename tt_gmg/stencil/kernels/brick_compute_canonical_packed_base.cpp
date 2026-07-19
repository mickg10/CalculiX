// Run52/run64 arithmetic for q1e-7 canonical base-stencil packed groups.
//
// All 81 shifted-B terms still stream.  The 33 BF16x3 palette tiles remain
// resident for the core's full assignment, and exact-zero coefficient planes
// skip their five or six mathematically zero tile products.  The host oracle
// proves this sparse schedule bitwise identical to the dense zero-product
// schedule for random, smooth, constant, and boundary vectors.
#include "api/compute/eltwise_binary.h"
#include "tt_metal/programming_examples/spmv_mac/kernels/canonical_base_palette_schedule.hpp"
#include <cstdint>

namespace {

inline uint32_t ordered_term(uint32_t stream_term, uint32_t order_mode) {
    constexpr uint32_t terms = 81;
    if (order_mode == 3) {
        const uint32_t component = stream_term / 27;
        const uint32_t reverse_offset = 26 - stream_term % 27;
        return reverse_offset * 3 + component;
    }
    if (order_mode == 1) {
        if (stream_term < 3) {
            return 39 + stream_term;
        }
        const uint32_t remainder = stream_term - 3;
        return remainder < 39 ? remainder : remainder + 3;
    }
    return order_mode == 2 ? terms - 1 - stream_term : stream_term;
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

}  // namespace

template <ckernel::MathFidelity math_fidelity>
ALWI void mac_term(
    uint32_t cb_a,
    uint32_t cb_b,
    uint32_t ai,
    uint32_t bi,
    uint32_t idst,
    bool first) {
    UNPACK((llk_unpack_AB(cb_a, cb_b, ai, bi)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          math_fidelity,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, first)));
}

void kernel_main() {
    const uint32_t n_groups = get_arg_val<uint32_t>(0);
    const uint32_t order_mode = get_arg_val<uint32_t>(1);
    constexpr auto cb_palette = tt::CBIndex::c_0;
    constexpr auto cb_bhi = tt::CBIndex::c_1;
    constexpr auto cb_bmid = tt::CBIndex::c_3;
    constexpr auto cb_blow = tt::CBIndex::c_4;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t accumulate_terms = get_compile_time_arg_val(0);
    constexpr uint32_t terms = canonical_packed_base::kTerms;
    constexpr uint32_t terms_per_publish = 3;
    static_assert(terms % terms_per_publish == 0);
    static_assert(canonical_packed_base::kPaletteTiles == 33);
    static_assert(canonical_packed_base::kNonzeroScalarPlanes == 153);
    static_assert(canonical_packed_base::kRun52ProductsPerGroup == 864);

    cb_wait_front(cb_palette, canonical_packed_base::kPaletteTiles);
    binary_op_init_common(cb_palette, cb_bhi, cb_out);
    mul_tiles_init(cb_palette, cb_bhi);
    for (uint32_t group = 0; group < n_groups; ++group) {
        bool initialized[canonical_packed_base::kOutputComponents] = {
            false, false, false};
        tile_regs_acquire();
        for (uint32_t stream_batch = 0;
             stream_batch < terms / terms_per_publish;
             ++stream_batch) {
            uint32_t low_count = 0;
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                low_count += retains_blow(
                    ordered_term(stream_term, order_mode)) ? 1u : 0u;
            }
            const uint32_t low_publish_count =
                order_mode == 3 && low_count != 0
                    ? terms_per_publish
                    : low_count;
            cb_wait_front(cb_bhi, terms_per_publish);
            cb_wait_front(cb_bmid, terms_per_publish);
            if (low_publish_count != 0) {
                cb_wait_front(cb_blow, low_publish_count);
            }
            uint32_t low_index = 0;
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                const uint32_t term = ordered_term(stream_term, order_mode);
                const bool retain_low = retains_blow(term);
                if (stream_term < accumulate_terms) {
                    for (uint32_t output_component = 0;
                         output_component <
                             canonical_packed_base::kOutputComponents;
                         ++output_component) {
                        const uint32_t palette_index =
                            canonical_packed_base::kPaletteIndex
                                [term][output_component];
                        if (palette_index ==
                            canonical_packed_base::kZeroPaletteIndex) {
                            continue;
                        }
                        const uint32_t a_hi = palette_index;
                        const uint32_t a_mid =
                            canonical_packed_base::kPaletteSize + palette_index;
                        const uint32_t a_low =
                            2 * canonical_packed_base::kPaletteSize +
                            palette_index;
                        const bool first = !initialized[output_component];
                        initialized[output_component] = true;
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_palette, cb_bhi, a_hi, batch_lane,
                            output_component, first);
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_palette, cb_bmid, a_hi, batch_lane,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_palette, cb_bhi, a_mid, batch_lane,
                            output_component, false);
                        if (retain_low) {
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_palette, cb_blow, a_hi, low_index,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_palette, cb_bmid, a_mid, batch_lane,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_palette, cb_bhi, a_low, batch_lane,
                                output_component, false);
                        } else {
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_palette, cb_bhi, a_low, batch_lane,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_palette, cb_bmid, a_mid, batch_lane,
                                output_component, false);
                        }
                    }
                }
                low_index += retain_low ? 1u : 0u;
            }
            cb_pop_front(cb_bhi, terms_per_publish);
            cb_pop_front(cb_bmid, terms_per_publish);
            if (low_publish_count != 0) {
                cb_pop_front(cb_blow, low_publish_count);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 3);
        tile_regs_wait();
        pack_tile(0, cb_out, 0);
        pack_tile(1, cb_out, 1);
        pack_tile(2, cb_out, 2);
        cb_push_back(cb_out, 3);
        tile_regs_release();
    }
}
