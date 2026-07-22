// RISC-1 reader/writer for compact exact fallback A and packed B/output.
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

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t a2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t a_start_group = get_arg_val<uint32_t>(arg++);
    const uint32_t packed_start_group = get_arg_val<uint32_t>(arg++);
    const uint32_t order_mode = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_a_low = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bmid = get_compile_time_arg_val(1);
    constexpr uint32_t cb_blow = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t terms = 81;
    constexpr uint32_t terms_per_publish = 3;
    constexpr uint32_t a_tiles_per_term = 3;
    static_assert(terms % terms_per_publish == 0);

    constexpr auto g0 = TensorAccessorArgs<4>();
    const auto a2 = TensorAccessor(g0, a2_addr, tile_bytes);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto b1 = TensorAccessor(g1, b1_addr, tile_bytes);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto b2 = TensorAccessor(g2, b2_addr, tile_bytes);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(g3, output_addr, 4096);

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t a_group = a_start_group + group_slot;
        const uint32_t packed_group = packed_start_group + group_slot;
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
            cb_reserve_back(cb_a_low, terms_per_publish * a_tiles_per_term);
            cb_reserve_back(cb_bmid, terms_per_publish);
            if (low_count != 0) {
                cb_reserve_back(cb_blow, low_count);
            }
            const uint32_t a_l1 = get_write_ptr(cb_a_low);
            const uint32_t mid_l1 = get_write_ptr(cb_bmid);
            const uint32_t low_l1 =
                low_count == 0 ? 0 : get_write_ptr(cb_blow);
            uint32_t low_index = 0;
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                const uint32_t term = ordered_term(stream_term, order_mode);
                const uint32_t a_page0 = (a_group * terms + term) * 3;
                const uint32_t b_page = packed_group * terms + term;
                const uint32_t a_index = batch_lane * a_tiles_per_term;
                for (uint32_t output_component = 0;
                     output_component < 3;
                     ++output_component) {
                    noc_async_read_page(
                        a_page0 + output_component,
                        a2,
                        a_l1 + (a_index + output_component) * tile_bytes);
                }
                noc_async_read_page(
                    b_page,
                    b1,
                    mid_l1 + batch_lane * tile_bytes);
                if (retains_blow(term)) {
                    noc_async_read_page(
                        b_page,
                        b2,
                        low_l1 + low_index * tile_bytes);
                    ++low_index;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_a_low, terms_per_publish * a_tiles_per_term);
            cb_push_back(cb_bmid, terms_per_publish);
            if (low_count != 0) {
                cb_push_back(cb_blow, low_count);
            }
        }
    }

    for (uint32_t group = 0; group < n_groups; ++group) {
        cb_wait_front(cb_out, 3);
        const uint32_t source = get_read_ptr(cb_out);
        const uint32_t page0 = (packed_start_group + group) * 3;
        noc_async_write_page(page0 + 0, output, source + 0 * 4096);
        noc_async_write_page(page0 + 1, output, source + 1 * 4096);
        noc_async_write_page(page0 + 2, output, source + 2 * 4096);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 3);
    }
}
