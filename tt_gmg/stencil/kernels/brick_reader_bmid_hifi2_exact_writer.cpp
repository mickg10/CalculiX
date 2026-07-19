// Load the resident halo, materialize top/residual tiles for x_mid, and drain
// output.  This is the RISC-1 half of the fixed-HiFi2 exact-leading mode.
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

inline uint16_t bf16_residual_bits(uint16_t bits) {
    if ((bits & 1u) == 0u) {
        return 0;
    }
    const uint16_t exponent = (bits >> 7) & 0xffu;
    if (exponent == 0xffu) {
        return 0;
    }
    const uint16_t sign = bits & 0x8000u;
    if (exponent >= 8u) {
        return sign | static_cast<uint16_t>((exponent - 7u) << 7);
    }
    const uint16_t subnormal = exponent == 0u
        ? 1u
        : static_cast<uint16_t>(1u << (exponent - 1u));
    return sign | subnormal;
}

inline void store_split_word(
    uint64_t source,
    tt_l1_ptr uint64_t* top_words,
    tt_l1_ptr uint64_t* residual_words,
    uint32_t index) {
    uint64_t top = source & 0xfffefffefffefffeULL;
    uint64_t residual = 0;
    for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t shift = lane * 16;
        const uint16_t bits = static_cast<uint16_t>(source >> shift);
        if (((bits >> 7) & 0xffu) == 0xffu) {
            top |= static_cast<uint64_t>(bits & 1u) << shift;
        }
        residual |= static_cast<uint64_t>(bf16_residual_bits(bits)) << shift;
    }
    top_words[index] = top;
    residual_words[index] = residual;
}

inline void materialize_b_split_tiles(
    uint32_t top_destination,
    uint32_t residual_destination,
    tt_l1_ptr uint64_t* halo_words,
    uint32_t group_slot,
    uint32_t plane,
    uint32_t di,
    uint32_t dj,
    uint32_t dk) {
    tt_l1_ptr uint64_t* top_words = (tt_l1_ptr uint64_t*)top_destination;
    tt_l1_ptr uint64_t* residual_words =
        (tt_l1_ptr uint64_t*)residual_destination;
    const uint32_t plane_word0 = (group_slot * 9 + plane) * (8192 / 4);
    constexpr uint32_t source_word_offsets[16] = {
        0, 3, 6, 9, 18, 21, 24, 27, 36, 39, 42, 45, 54, 57, 60, 63};
    for (uint32_t brick = 0; brick < 16; ++brick) {
        tt_l1_ptr uint64_t* source =
            halo_words + plane_word0 + brick * (432 / 4) +
            (di * 6 + dj) * 3 + dk;
        const uint32_t destination_word0 = brick * 16;
        for (uint32_t word = 0; word < 16; ++word) {
            store_split_word(
                source[source_word_offsets[word]],
                top_words,
                residual_words,
                destination_word0 + word);
        }
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
            cb_reserve_back(cb_bmid, terms_per_publish * 2);
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
                const uint32_t pair =
                    destination + batch_lane * 2 * tile_bytes;
                materialize_b_split_tiles(
                    pair,
                    pair + tile_bytes,
                    halo_words,
                    group_slot,
                    component * 3 + 1,
                    di,
                    dj,
                    dk);
            }
            cb_push_back(cb_bmid, terms_per_publish * 2);
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
