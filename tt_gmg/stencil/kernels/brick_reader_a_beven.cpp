// Stream A for every term and materialize all three x splits for even stream
// terms.  The companion RISC-1 kernel handles odd terms.  Keeping the split
// loop innermost shares halo address arithmetic across the three bf16 levels.
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

inline void materialize_b3(
    uint32_t destination,
    tt_l1_ptr uint64_t* halo_words,
    uint32_t group_slot,
    uint32_t component,
    uint32_t di,
    uint32_t dj,
    uint32_t dk) {
    constexpr uint32_t tile_words = 2048 / sizeof(uint64_t);
    constexpr uint32_t plane_words = 8192 / 4;
    tt_l1_ptr uint64_t* b0 = (tt_l1_ptr uint64_t*)destination;
    tt_l1_ptr uint64_t* b1 = b0 + tile_words;
    tt_l1_ptr uint64_t* b2 = b1 + tile_words;
    const uint32_t plane0_word =
        (group_slot * 9 + component * 3) * plane_words;
    for (uint32_t brick = 0; brick < 16; ++brick) {
        const uint32_t brick_word0 = plane0_word + brick * (432 / 4);
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t halo_i = i + di;
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t destination_word = brick * 16 + i * 4 + j;
                const uint32_t source_word =
                    brick_word0 + (halo_i * 6 + j + dj) * 3 + dk;
                b0[destination_word] = halo_words[source_word];
                b1[destination_word] = halo_words[source_word + plane_words];
                b2[destination_word] = halo_words[source_word + 2 * plane_words];
            }
        }
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
    constexpr uint32_t cb_beven = get_compile_time_arg_val(1);
    constexpr uint32_t cb_halo = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t terms = 81;
    constexpr uint32_t halo_tiles_per_group = 72;

    constexpr auto g0 = TensorAccessorArgs<3>();
    const auto a0 = TensorAccessor(g0, a0_addr);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto a1 = TensorAccessor(g1, a1_addr);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto a2 = TensorAccessor(g2, a2_addr);

    const uint32_t halo_tiles = n_groups * halo_tiles_per_group;
    cb_wait_front(cb_halo, halo_tiles);
    tt_l1_ptr uint64_t* halo_words =
        (tt_l1_ptr uint64_t*)get_read_ptr(cb_halo);

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t local_group = start_group + group_slot;
        for (uint32_t stream_term = 0; stream_term < terms; ++stream_term) {
            const uint32_t term = ordered_term(stream_term, order_mode);
            const uint32_t offset_id = term / 3;
            const uint32_t component = term - offset_id * 3;
            const uint32_t di = offset_id / 9;
            const uint32_t dj = (offset_id / 3) % 3;
            const uint32_t dk = offset_id % 3;

            cb_reserve_back(cb_a, 9);
            const uint32_t pa = get_write_ptr(cb_a);
            const uint32_t a_page = (local_group * terms + term) * 3;
            for (uint32_t output_component = 0;
                 output_component < 3;
                 ++output_component) {
                const uint32_t page = a_page + output_component;
                const uint32_t destination =
                    pa + output_component * 3 * tile_bytes;
                noc_async_read_page(page, a0, destination + 0 * tile_bytes);
                noc_async_read_page(page, a1, destination + 1 * tile_bytes);
                noc_async_read_page(page, a2, destination + 2 * tile_bytes);
            }
            if ((stream_term & 1u) == 0) {
                cb_reserve_back(cb_beven, 3);
                materialize_b3(
                    get_write_ptr(cb_beven),
                    halo_words,
                    group_slot,
                    component,
                    di,
                    dj,
                    dk);
                cb_push_back(cb_beven, 3);
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, 9);
        }
    }
}
