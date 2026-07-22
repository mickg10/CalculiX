// RISC-0 half of the shifted-resident-vector brick path.
//
// A-hi/A-mid and the already shifted x-hi tile pages are direct NOC reads.
// No scalar 6x6x6-halo materialization remains in the timed apply.
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
    const uint32_t a0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t start_group = get_arg_val<uint32_t>(arg++);
    const uint32_t order_mode = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_a_hi_mid = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bhi = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t terms = 81;
    constexpr uint32_t terms_per_publish = 3;
    constexpr uint32_t a_tiles_per_term = 6;
    static_assert(terms % terms_per_publish == 0);

    constexpr auto g0 = TensorAccessorArgs<2>();
    const auto a0 = TensorAccessor(g0, a0_addr, tile_bytes);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto a1 = TensorAccessor(g1, a1_addr, tile_bytes);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto b0 = TensorAccessor(g2, b0_addr, tile_bytes);

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t local_group = start_group + group_slot;
        for (uint32_t stream_batch = 0;
             stream_batch < terms / terms_per_publish;
             ++stream_batch) {
            cb_reserve_back(
                cb_a_hi_mid,
                terms_per_publish * a_tiles_per_term);
            cb_reserve_back(cb_bhi, terms_per_publish);
            const uint32_t pa = get_write_ptr(cb_a_hi_mid);
            const uint32_t pb = get_write_ptr(cb_bhi);
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                const uint32_t term = ordered_term(stream_term, order_mode);
                const uint32_t a_page0 =
                    (local_group * terms + term) * 3;
                const uint32_t b_page = local_group * terms + term;
                const uint32_t a_index = batch_lane * a_tiles_per_term;
                for (uint32_t output_component = 0;
                     output_component < 3;
                     ++output_component) {
                    noc_async_read_page(
                        a_page0 + output_component,
                        a0,
                        pa + (a_index + output_component) * tile_bytes);
                    noc_async_read_page(
                        a_page0 + output_component,
                        a1,
                        pa + (a_index + 3 + output_component) * tile_bytes);
                }
                noc_async_read_page(
                    b_page,
                    b0,
                    pb + batch_lane * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(
                cb_a_hi_mid,
                terms_per_publish * a_tiles_per_term);
            cb_push_back(cb_bhi, terms_per_publish);
        }
    }
}
