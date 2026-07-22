// Fixed-HiFi2 bf16x3 accumulation.  Degree-0/1 products are reconstructed
// exactly from top/residual B tile pairs; degree-2 products use one HiFi2
// call.  That is 18 fidelity phases instead of run41's 24, with no runtime
// math-MOP transition.
#include "api/compute/eltwise_binary.h"
#include <cstdint>

namespace {

inline uint32_t ordered_term(uint32_t stream_term, uint32_t order_mode) {
    constexpr uint32_t terms = 81;
    if (order_mode == 1) {
        if (stream_term < 3) {
            return 39 + stream_term;
        }
        const uint32_t remainder = stream_term - 3;
        return remainder < 39 ? remainder : remainder + 3;
    }
    return order_mode == 2 ? terms - 1 - stream_term : stream_term;
}

#ifdef TRISC_PACK
inline void materialize_low_tile(
    uint32_t destination,
    uint32_t halo_l1,
    uint32_t group_slot,
    uint32_t component,
    uint32_t di,
    uint32_t dj,
    uint32_t dk) {
    constexpr uint32_t plane_words = 8192 / 4;
    tt_l1_ptr uint64_t* halo_words = (tt_l1_ptr uint64_t*)halo_l1;
    tt_l1_ptr uint64_t* b_words = (tt_l1_ptr uint64_t*)destination;
    const uint32_t plane_word0 =
        (group_slot * 9 + component * 3 + 2) * plane_words;
    constexpr uint32_t source_word_offsets[16] = {
        0, 3, 6, 9, 18, 21, 24, 27, 36, 39, 42, 45, 54, 57, 60, 63};
    for (uint32_t brick = 0; brick < 16; ++brick) {
        tt_l1_ptr uint64_t* source =
            halo_words + plane_word0 + brick * (432 / 4) +
            (di * 6 + dj) * 3 + dk;
        tt_l1_ptr uint64_t* destination_words = b_words + brick * 16;
        for (uint32_t word = 0; word < 16; ++word) {
            destination_words[word] = source[source_word_offsets[word]];
        }
    }
}
#endif

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
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_bhi = tt::CBIndex::c_1;
    constexpr auto cb_halo = tt::CBIndex::c_2;
    constexpr auto cb_bmid = tt::CBIndex::c_3;
    constexpr auto cb_blow = tt::CBIndex::c_4;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t accumulate_terms = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t terms = 81;
    constexpr uint32_t terms_per_publish = 3;
    constexpr uint32_t halo_tiles_per_group = 72;

    const uint32_t halo_tiles = n_groups * halo_tiles_per_group;
    cb_wait_front(cb_halo, halo_tiles);
    const uint32_t halo_l1 = get_tile_address(cb_halo, 0);

    binary_op_init_common(cb_a, cb_bhi, cb_out);
    mul_tiles_init(cb_a, cb_bhi);
    for (uint32_t group = 0; group < n_groups; ++group) {
        tile_regs_acquire();
        for (uint32_t stream_batch = 0;
             stream_batch < terms / terms_per_publish;
             ++stream_batch) {
            cb_reserve_back(cb_blow, terms_per_publish);
            PACK({
                const uint32_t destination =
                    get_local_cb_interface(static_cast<uint32_t>(cb_blow))
                        .fifo_wr_ptr
                    << 4;
                for (uint32_t batch_lane = 0;
                     batch_lane < terms_per_publish;
                     ++batch_lane) {
                    const uint32_t stream_term =
                        stream_batch * terms_per_publish + batch_lane;
                    const uint32_t term = ordered_term(stream_term, order_mode);
                    const uint32_t offset_id = term / 3;
                    const uint32_t component = term - offset_id * 3;
                    const uint32_t di = offset_id / 9;
                    const uint32_t dj = (offset_id / 3) % 3;
                    const uint32_t dk = offset_id % 3;
                    materialize_low_tile(
                        destination + batch_lane * tile_bytes,
                        halo_l1,
                        group,
                        component,
                        di,
                        dj,
                        dk);
                }
            });
            cb_push_back(cb_blow, terms_per_publish);

            cb_wait_front(cb_a, terms_per_publish * 9);
            cb_wait_front(cb_bhi, terms_per_publish * 2);
            cb_wait_front(cb_bmid, terms_per_publish * 2);
            cb_wait_front(cb_blow, terms_per_publish);
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                if (stream_term < accumulate_terms) {
                    const bool first = stream_term == 0;
                    const uint32_t b_hi_top = batch_lane * 2;
                    const uint32_t b_hi_residual = b_hi_top + 1;
                    const uint32_t b_mid_top = batch_lane * 2;
                    const uint32_t b_mid_residual = b_mid_top + 1;
                    for (uint32_t output_component = 0;
                         output_component < 3;
                         ++output_component) {
                        const uint32_t a_hi =
                            batch_lane * 9 + output_component;
                        const uint32_t a_mid =
                            batch_lane * 9 + 3 + output_component;
                        const uint32_t a_low =
                            batch_lane * 9 + 6 + output_component;
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_bhi, a_hi, b_hi_top,
                            output_component, first);
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_bhi, a_hi, b_hi_residual,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_bmid, a_hi, b_mid_top,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_bmid, a_hi, b_mid_residual,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_bhi, a_mid, b_hi_top,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_bhi, a_mid, b_hi_residual,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_blow, a_hi, batch_lane,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_bmid, a_mid, b_mid_top,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi2>(
                            cb_a, cb_bhi, a_low, b_hi_top,
                            output_component, false);
                    }
                }
            }
            cb_pop_front(cb_a, terms_per_publish * 9);
            cb_pop_front(cb_bhi, terms_per_publish * 2);
            cb_pop_front(cb_bmid, terms_per_publish * 2);
            cb_pop_front(cb_blow, terms_per_publish);
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
