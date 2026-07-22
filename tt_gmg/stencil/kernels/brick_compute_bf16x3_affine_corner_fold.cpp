// Affine-moment-preserving corner compression for the bf16x3 brick apply.
//
// The host folds every corner A-low coefficient w into the three signed
// face coefficients (+w each) and the center coefficient (-2w), then rounds
// those adjusted low coefficients back to BF16.  That identity preserves the
// corner action for affine x before re-rounding.  This kernel consequently
// reads A-low from DRAM only for non-corner offsets.  Corner A-low CB slots
// contain local zeros, allowing this kernel to retain run52's exact per-term
// MAC cadence and x-low policy: all non-corners plus corner pair 6/20.
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

inline uint32_t manhattan_offset(uint32_t term) {
    const uint32_t offset_id = term / 3;
    const uint32_t di = offset_id / 9;
    const uint32_t dj = (offset_id / 3) % 3;
    const uint32_t dk = offset_id % 3;
    return
        (di > 1 ? di - 1 : 1 - di) +
        (dj > 1 ? dj - 1 : 1 - dj) +
        (dk > 1 ? dk - 1 : 1 - dk);
}

inline bool retains_blow(uint32_t term) {
    const uint32_t offset_id = term / 3;
    return manhattan_offset(term) <= 2 || offset_id == 6 || offset_id == 20;
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
    for (uint32_t brick = 0; brick < 16; ++brick) {
        tt_l1_ptr uint64_t* source =
            halo_words + plane_word0 + brick * (432 / 4) +
            (di * 6 + dj) * 3 + dk;
        tt_l1_ptr uint64_t* destination_words = b_words + brick * 16;

        destination_words[0] = source[0];
        destination_words[1] = source[3];
        destination_words[2] = source[6];
        destination_words[3] = source[9];
        destination_words[4] = source[18];
        destination_words[5] = source[21];
        destination_words[6] = source[24];
        destination_words[7] = source[27];
        destination_words[8] = source[36];
        destination_words[9] = source[39];
        destination_words[10] = source[42];
        destination_words[11] = source[45];
        destination_words[12] = source[54];
        destination_words[13] = source[57];
        destination_words[14] = source[60];
        destination_words[15] = source[63];
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
    constexpr uint32_t a_tiles_per_publish = terms_per_publish * 9;
    constexpr uint32_t halo_tiles_per_group = 72;
    static_assert(terms % terms_per_publish == 0);

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
            uint32_t low_count = 0;
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                const uint32_t term = ordered_term(stream_term, order_mode);
                low_count += retains_blow(term) ? 1u : 0u;
            }
            if (low_count != 0) {
                cb_reserve_back(cb_blow, low_count);
                PACK({
                    const uint32_t destination =
                        get_local_cb_interface(static_cast<uint32_t>(cb_blow))
                            .fifo_wr_ptr
                        << 4;
                    uint32_t low_index = 0;
                    for (uint32_t batch_lane = 0;
                         batch_lane < terms_per_publish;
                         ++batch_lane) {
                        const uint32_t stream_term =
                            stream_batch * terms_per_publish + batch_lane;
                        const uint32_t term = ordered_term(stream_term, order_mode);
                        if (!retains_blow(term)) {
                            continue;
                        }
                        const uint32_t offset_id = term / 3;
                        const uint32_t component = term - offset_id * 3;
                        const uint32_t di = offset_id / 9;
                        const uint32_t dj = (offset_id / 3) % 3;
                        const uint32_t dk = offset_id % 3;
                        materialize_low_tile(
                            destination + low_index * tile_bytes,
                            halo_l1,
                            group,
                            component,
                            di,
                            dj,
                            dk);
                        ++low_index;
                    }
                });
                cb_push_back(cb_blow, low_count);
            }

            cb_wait_front(cb_a, a_tiles_per_publish);
            cb_wait_front(cb_bhi, terms_per_publish);
            cb_wait_front(cb_bmid, terms_per_publish);
            if (low_count != 0) {
                cb_wait_front(cb_blow, low_count);
            }
            uint32_t low_index = 0;
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                const uint32_t term = ordered_term(stream_term, order_mode);
                const bool retain_b_low = retains_blow(term);
                const uint32_t a_index = batch_lane * 9;
                if (stream_term < accumulate_terms) {
                    const bool first = stream_term == 0;
                    for (uint32_t output_component = 0;
                         output_component < 3;
                         ++output_component) {
                        const uint32_t a_hi = a_index + output_component;
                        const uint32_t a_mid = a_index + 3 + output_component;
                        const uint32_t a_low = a_index + 6 + output_component;
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_a, cb_bhi, a_hi, batch_lane,
                            output_component, first);
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_a, cb_bmid, a_hi, batch_lane,
                            output_component, false);
                        mac_term<ckernel::MathFidelity::HiFi4>(
                            cb_a, cb_bhi, a_mid, batch_lane,
                            output_component, false);
                        if (retain_b_low) {
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a, cb_blow, a_hi, low_index,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a, cb_bmid, a_mid, batch_lane,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a, cb_bhi, a_low, batch_lane,
                                output_component, false);
                        } else {
                            // Match run52's five-product T5 order exactly.
                            // For a folded corner, a_low is a local zero tile.
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a, cb_bhi, a_low, batch_lane,
                                output_component, false);
                            mac_term<ckernel::MathFidelity::HiFi4>(
                                cb_a, cb_bmid, a_mid, batch_lane,
                                output_component, false);
                        }
                    }
                }
                low_index += retain_b_low ? 1u : 0u;
            }
            cb_pop_front(cb_a, a_tiles_per_publish);
            cb_pop_front(cb_bhi, terms_per_publish);
            cb_pop_front(cb_bmid, terms_per_publish);
            if (low_count != 0) {
                cb_pop_front(cb_blow, low_count);
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
