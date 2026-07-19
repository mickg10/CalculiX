// Lossless corner compression without a second A buffer or variable CB
// cadence.  Every corner entry is one of three exact BF16x3 coefficient
// tiles already present in the original A splits.  Read those nine
// representative tiles into the first nine pages of the normal 27-page A
// publish; the remaining pages are padding and are never consumed.
#include "api/dataflow/dataflow_api.h"
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

inline bool is_corner(uint32_t offset_id) {
    const uint32_t di = offset_id / 9;
    const uint32_t dj = (offset_id / 3) % 3;
    const uint32_t dk = offset_id % 3;
    return di != 1 && dj != 1 && dk != 1;
}

// Return a page containing the requested exact coefficient type.  Type 1 is
// diagonal.  Type 2 uses a same-sign off-diagonal pair, and type 0 a
// different-sign pair.  An all-equal-sign corner has no type-0 entries, so
// its unused type-0 slot may safely alias the diagonal representative.
inline uint32_t representative_page(
    uint32_t local_group,
    uint32_t offset_id,
    uint32_t type) {
    uint32_t input_component = 0;
    uint32_t output_component = 0;
    if (type != 1) {
        const bool signs[3] = {
            offset_id / 9 == 2,
            (offset_id / 3) % 3 == 2,
            offset_id % 3 == 2,
        };
        const bool seek_equal = type == 2;
        if ((signs[0] == signs[1]) == seek_equal) {
            output_component = 1;
        } else if ((signs[0] == signs[2]) == seek_equal) {
            output_component = 2;
        } else if (seek_equal) {
            input_component = 1;
            output_component = 2;
        }
    }
    const uint32_t term = offset_id * 3 + input_component;
    return (local_group * 81 + term) * 3 + output_component;
}

inline void materialize_b_tile(
    uint32_t destination,
    tt_l1_ptr uint64_t* halo_words,
    uint32_t group_slot,
    uint32_t plane,
    uint32_t di,
    uint32_t dj,
    uint32_t dk) {
    tt_l1_ptr uint64_t* b_words = (tt_l1_ptr uint64_t*)destination;
    const uint32_t plane_word0 =
        (group_slot * 9 + plane) * (8192 / 4);
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

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t a0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t start_group = get_arg_val<uint32_t>(arg++);
    const uint32_t order_mode = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bhi = get_compile_time_arg_val(1);
    constexpr uint32_t cb_halo = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t terms = 81;
    constexpr uint32_t terms_per_publish = 3;
    constexpr uint32_t a_tiles_per_publish = terms_per_publish * 9;
    constexpr uint32_t halo_tiles_per_group = 72;
    static_assert(terms % terms_per_publish == 0);

    constexpr auto g0 = TensorAccessorArgs<3>();
    const auto a0 = TensorAccessor(g0, a0_addr, tile_bytes);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto a1 = TensorAccessor(g1, a1_addr, tile_bytes);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto a2 = TensorAccessor(g2, a2_addr, tile_bytes);

    const uint32_t halo_tiles = n_groups * halo_tiles_per_group;
    cb_wait_front(cb_halo, halo_tiles);
    tt_l1_ptr uint64_t* halo_words =
        (tt_l1_ptr uint64_t*)get_read_ptr(cb_halo);

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t local_group = start_group + group_slot;
        for (uint32_t stream_batch = 0;
             stream_batch < terms / terms_per_publish;
             ++stream_batch) {
            const uint32_t first_term =
                ordered_term(stream_batch * terms_per_publish, order_mode);
            const uint32_t offset_id = first_term / 3;
            const bool corner_batch = is_corner(offset_id);

            cb_reserve_back(cb_a, a_tiles_per_publish);
            cb_reserve_back(cb_bhi, terms_per_publish);
            const uint32_t pa = get_write_ptr(cb_a);
            const uint32_t pb = get_write_ptr(cb_bhi);

            if (corner_batch) {
                for (uint32_t type = 0; type < 3; ++type) {
                    const uint32_t page =
                        representative_page(local_group, offset_id, type);
                    noc_async_read_page(
                        page, a0, pa + (0 * 3 + type) * tile_bytes);
                    noc_async_read_page(
                        page, a1, pa + (1 * 3 + type) * tile_bytes);
                    noc_async_read_page(
                        page, a2, pa + (2 * 3 + type) * tile_bytes);
                }
            }

            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                const uint32_t term = ordered_term(stream_term, order_mode);
                const uint32_t term_offset = term / 3;
                const uint32_t component = term - term_offset * 3;
                const uint32_t di = term_offset / 9;
                const uint32_t dj = (term_offset / 3) % 3;
                const uint32_t dk = term_offset % 3;
                if (!corner_batch) {
                    for (uint32_t output_component = 0;
                         output_component < 3;
                         ++output_component) {
                        const uint32_t a_page =
                            (local_group * terms + term) * 3 + output_component;
                        const uint32_t destination =
                            pa + (batch_lane * 9 + output_component) * tile_bytes;
                        noc_async_read_page(
                            a_page, a0, destination + 0 * 3 * tile_bytes);
                        noc_async_read_page(
                            a_page, a1, destination + 1 * 3 * tile_bytes);
                        noc_async_read_page(
                            a_page, a2, destination + 2 * 3 * tile_bytes);
                    }
                }
                materialize_b_tile(
                    pb + batch_lane * tile_bytes,
                    halo_words,
                    group_slot,
                    component * 3,
                    di,
                    dj,
                    dk);
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, a_tiles_per_publish);
            cb_push_back(cb_bhi, terms_per_publish);
        }
    }
}
