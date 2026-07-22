// Load the resident halo, materialize the middle x split, then drain output.
// This occupies RISC-1 during the apply instead of leaving the writer idle.
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

inline void materialize_b_tile(
    uint32_t destination,
    tt_l1_ptr uint64_t* halo_words,
    uint32_t group_slot,
    uint32_t plane,
    uint32_t di,
    uint32_t dj,
    uint32_t dk) {
    tt_l1_ptr uint64_t* b_words =
        (tt_l1_ptr uint64_t*)destination;
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
    const uint32_t halo_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t start_group = get_arg_val<uint32_t>(arg++);
    const uint32_t core_index = get_arg_val<uint32_t>(arg++);
    const uint32_t order_mode = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_bmid = get_compile_time_arg_val(0);
    constexpr uint32_t cb_halo = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t terms = 81;
    constexpr uint32_t terms_per_publish = 3;
    constexpr uint32_t halo_tiles_per_group = 72;
    constexpr uint32_t halo_tiles_per_core = 288;
    static_assert(terms % 3 == 0);

    constexpr auto g0 = TensorAccessorArgs<3>();
    const auto halo = TensorAccessor(g0, halo_addr);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(g1, output_addr, 4096);

    const uint32_t halo_tiles = n_groups * halo_tiles_per_group;
    cb_reserve_back(cb_halo, halo_tiles);
    const uint32_t halo_l1 = get_write_ptr(cb_halo);
    const uint32_t halo_page0 = core_index * halo_tiles_per_core;
    for (uint32_t page = 0; page < halo_tiles; ++page) {
        noc_async_read_page(
            halo_page0 + page,
            halo,
            halo_l1 + page * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_halo, halo_tiles);
    tt_l1_ptr uint64_t* halo_words = (tt_l1_ptr uint64_t*)halo_l1;

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        for (uint32_t stream_batch = 0;
             stream_batch < terms / terms_per_publish;
             ++stream_batch) {
            cb_reserve_back(cb_bmid, terms_per_publish);
            const uint32_t destination = get_write_ptr(cb_bmid);
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
                materialize_b_tile(
                    destination + batch_lane * tile_bytes,
                    halo_words,
                    group_slot,
                    component * 3 + 1,
                    di,
                    dj,
                    dk);
            }
            cb_push_back(cb_bmid, terms_per_publish);
        }
    }

    for (uint32_t group = 0; group < n_groups; ++group) {
        cb_wait_front(cb_out, 3);
        const uint32_t source = get_read_ptr(cb_out);
        const uint32_t page0 = (start_group + group) * 3;
        noc_async_write_page(page0 + 0, output, source + 0 * 4096);
        noc_async_write_page(page0 + 1, output, source + 1 * 4096);
        noc_async_write_page(page0 + 2, output, source + 2 * 4096);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 3);
    }
    cb_pop_front(cb_halo, halo_tiles);
}
