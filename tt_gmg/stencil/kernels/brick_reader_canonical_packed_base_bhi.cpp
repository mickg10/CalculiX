// RISC-0 reader for the q1e-7 canonical base-stencil packed groups.
//
// The 11-value BF16x3 coefficient palette is loaded once per core and stays
// resident in c_0.  Only direct packed shifted-B high pages stream per group.
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

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t palette0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t palette1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t palette2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t start_group = get_arg_val<uint32_t>(arg++);
    const uint32_t order_mode = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_palette = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bhi = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t terms = 81;
    constexpr uint32_t terms_per_publish = 3;
    constexpr uint32_t palette_size = 11;
    constexpr uint32_t palette_splits = 3;
    constexpr uint32_t palette_tiles = palette_size * palette_splits;
    static_assert(terms % terms_per_publish == 0);

    constexpr auto g0 = TensorAccessorArgs<2>();
    const auto palette0 = TensorAccessor(g0, palette0_addr, tile_bytes);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto palette1 = TensorAccessor(g1, palette1_addr, tile_bytes);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto palette2 = TensorAccessor(g2, palette2_addr, tile_bytes);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();
    const auto b0 = TensorAccessor(g3, b0_addr, tile_bytes);

    cb_reserve_back(cb_palette, palette_tiles);
    const uint32_t palette_l1 = get_write_ptr(cb_palette);
    for (uint32_t page = 0; page < palette_size; ++page) {
        noc_async_read_page(
            page,
            palette0,
            palette_l1 + (0 * palette_size + page) * tile_bytes);
        noc_async_read_page(
            page,
            palette1,
            palette_l1 + (1 * palette_size + page) * tile_bytes);
        noc_async_read_page(
            page,
            palette2,
            palette_l1 + (2 * palette_size + page) * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_palette, palette_tiles);

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t packed_group = start_group + group_slot;
        for (uint32_t stream_batch = 0;
             stream_batch < terms / terms_per_publish;
             ++stream_batch) {
            cb_reserve_back(cb_bhi, terms_per_publish);
            const uint32_t destination = get_write_ptr(cb_bhi);
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                const uint32_t term = ordered_term(stream_term, order_mode);
                const uint32_t page = packed_group * terms + term;
                noc_async_read_page(
                    page,
                    b0,
                    destination + batch_lane * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_bhi, terms_per_publish);
        }
    }
}
