// Qualifying run52 arithmetic with split A readers and direct shifted x pages.
// This removes only the scalar halo materializer; term order and products are
// unchanged from the already host/device-correct selective-b-low schedule.
#include "api/compute/eltwise_binary.h"
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
    constexpr auto cb_a_hi_mid = tt::CBIndex::c_0;
    constexpr auto cb_bhi = tt::CBIndex::c_1;
    constexpr auto cb_bmid = tt::CBIndex::c_3;
    constexpr auto cb_blow = tt::CBIndex::c_4;
    constexpr auto cb_a_low = tt::CBIndex::c_5;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t accumulate_terms = get_compile_time_arg_val(0);
    constexpr uint32_t terms = 81;
    constexpr uint32_t terms_per_publish = 3;
    constexpr uint32_t a_hi_mid_tiles_per_term = 6;
    constexpr uint32_t a_low_tiles_per_term = 3;
    static_assert(terms % terms_per_publish == 0);

    binary_op_init_common(cb_a_hi_mid, cb_bhi, cb_out);
    mul_tiles_init(cb_a_hi_mid, cb_bhi);
    for (uint32_t group = 0; group < n_groups; ++group) {
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
            cb_wait_front(
                cb_a_hi_mid,
                terms_per_publish * a_hi_mid_tiles_per_term);
            cb_wait_front(
                cb_a_low,
                terms_per_publish * a_low_tiles_per_term);
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
                    const bool first = stream_term == 0;
                    for (uint32_t output_component = 0;
                         output_component < 3;
                         ++output_component) {
                        const uint32_t a_hi =
                            batch_lane * a_hi_mid_tiles_per_term +
                            output_component;
                        const uint32_t a_mid =
                            batch_lane * a_hi_mid_tiles_per_term + 3 +
                            output_component;
                        const uint32_t a_low =
                            batch_lane * a_low_tiles_per_term +
                            output_component;
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_a_hi_mid, cb_bhi, a_hi, batch_lane,
                            output_component, first);
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_a_hi_mid, cb_bmid, a_hi, batch_lane,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_a_hi_mid, cb_bhi, a_mid, batch_lane,
                            output_component, false);
                        if (retain_low) {
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a_hi_mid, cb_blow, a_hi, low_index,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a_hi_mid, cb_bmid, a_mid, batch_lane,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a_low, cb_bhi, a_low, batch_lane,
                                output_component, false);
                        } else {
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a_low, cb_bhi, a_low, batch_lane,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a_hi_mid, cb_bmid, a_mid, batch_lane,
                                output_component, false);
                        }
                    }
                }
                low_index += retain_low ? 1u : 0u;
            }
            cb_pop_front(
                cb_a_hi_mid,
                terms_per_publish * a_hi_mid_tiles_per_term);
            cb_pop_front(
                cb_a_low,
                terms_per_publish * a_low_tiles_per_term);
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
