// Stream all A splits and materialize top/residual tiles for x_hi.  The pair
// lets the compute kernel reconstruct degree-0/1 products with one fixed
// HiFi2 MOP; run41's default HiFi4 reader remains a separate kernel.
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
    constexpr uint32_t halo_tiles_per_group = 72;

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
            cb_reserve_back(cb_a, terms_per_publish * 9);
            cb_reserve_back(cb_bhi, terms_per_publish * 2);
            const uint32_t pa = get_write_ptr(cb_a);
            const uint32_t pb = get_write_ptr(cb_bhi);
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
                for (uint32_t output_component = 0;
                     output_component < 3;
                     ++output_component) {
                    const uint32_t a_page =
                        (local_group * terms + term) * 3 + output_component;
                    const uint32_t destination =
                        pa + (batch_lane * 9 + output_component) * tile_bytes;
                    noc_async_read_page(a_page, a0, destination + 0 * 3 * tile_bytes);
                    noc_async_read_page(a_page, a1, destination + 1 * 3 * tile_bytes);
                    noc_async_read_page(a_page, a2, destination + 2 * 3 * tile_bytes);
                }
                const uint32_t pair = pb + batch_lane * 2 * tile_bytes;
                materialize_b_split_tiles(
                    pair,
                    pair + tile_bytes,
                    halo_words,
                    group_slot,
                    component * 3,
                    di,
                    dj,
                    dk);
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, terms_per_publish * 9);
            cb_push_back(cb_bhi, terms_per_publish * 2);
        }
    }
}
