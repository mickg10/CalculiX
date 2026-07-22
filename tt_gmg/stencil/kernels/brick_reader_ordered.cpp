// Single-pass reader with an explicit term permutation.  Center-first seeds
// the FPU destination with the dominant self block before neighbor cancellation.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t a0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t halo_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t start_group = get_arg_val<uint32_t>(arg++);
    const uint32_t core_index = get_arg_val<uint32_t>(arg++);
    const uint32_t order_mode = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_halo = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t terms = 81;
    constexpr uint32_t halo_tiles_per_group = 72;
    constexpr uint32_t halo_tiles_per_core = 288;

    constexpr auto g0 = TensorAccessorArgs<3>();
    const auto a0 = TensorAccessor(g0, a0_addr);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto a1 = TensorAccessor(g1, a1_addr);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto a2 = TensorAccessor(g2, a2_addr);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();
    const auto halo = TensorAccessor(g3, halo_addr);

    const uint32_t halo_tiles = n_groups * halo_tiles_per_group;
    cb_reserve_back(cb_halo, halo_tiles);
    const uint32_t halo_l1 = get_write_ptr(cb_halo);
    const uint32_t halo_page0 = core_index * halo_tiles_per_core;
    for (uint32_t page = 0; page < halo_tiles; ++page) {
        noc_async_read_page(halo_page0 + page, halo, halo_l1 + page * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_halo, halo_tiles);
    tt_l1_ptr uint64_t* halo_words = (tt_l1_ptr uint64_t*)halo_l1;

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t local_group = start_group + group_slot;
        for (uint32_t stream_term = 0; stream_term < terms; ++stream_term) {
            uint32_t term = stream_term;
            if (order_mode == 1) {
                if (stream_term < 3) {
                    term = 39 + stream_term;
                } else {
                    const uint32_t remainder = stream_term - 3;
                    term = remainder < 39 ? remainder : remainder + 3;
                }
            } else if (order_mode == 2) {
                term = terms - 1 - stream_term;
            }
            const uint32_t offset_id = term / 3;
            const uint32_t component = term - offset_id * 3;
            const uint32_t di = offset_id / 9;
            const uint32_t dj = (offset_id / 3) % 3;
            const uint32_t dk = offset_id % 3;

            cb_reserve_back(cb_a, 9);
            cb_reserve_back(cb_b, 3);
            const uint32_t pa = get_write_ptr(cb_a);
            const uint32_t pb = get_write_ptr(cb_b);
            const uint32_t a_page = (local_group * terms + term) * 3;
            for (uint32_t output_component = 0;
                 output_component < 3;
                 ++output_component) {
                const uint32_t page = a_page + output_component;
                const uint32_t destination = pa + output_component * 3 * tile_bytes;
                noc_async_read_page(page, a0, destination + 0 * tile_bytes);
                noc_async_read_page(page, a1, destination + 1 * tile_bytes);
                noc_async_read_page(page, a2, destination + 2 * tile_bytes);
            }

            for (uint32_t split = 0; split < 3; ++split) {
                tt_l1_ptr uint64_t* b_words =
                    (tt_l1_ptr uint64_t*)(pb + split * tile_bytes);
                const uint32_t plane = component * 3 + split;
                const uint32_t plane_word0 =
                    (group_slot * 9 + plane) * (8192 / 4);
                for (uint32_t brick = 0; brick < 16; ++brick) {
                    const uint32_t brick_word0 = plane_word0 + brick * (432 / 4);
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t halo_i = i + di;
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint32_t halo_j = j + dj;
                            const uint32_t source_word =
                                brick_word0 + (halo_i * 6 + halo_j) * 3 + dk;
                            const uint32_t destination_word = brick * 16 + i * 4 + j;
                            b_words[destination_word] = halo_words[source_word];
                        }
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, 9);
            cb_push_back(cb_b, 3);
        }
    }
    cb_pop_front(cb_halo, halo_tiles);
}
