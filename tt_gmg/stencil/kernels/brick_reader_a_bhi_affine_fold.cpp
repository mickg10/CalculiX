// Stream A-hi/A-mid for every term and the moment-folded A-low split only
// for non-corner offsets.  The host folds each omitted corner A-low tile
// into its signed faces and center before upload.  B-hi materialization is
// unchanged.  Keep the proven fixed 27-tile circular-buffer cadence: variable
// 18-27 tile reservations can strand the producer at a ring wrap even when
// total free space is sufficient.  Runs 55 and 56 also showed that merely
// omitting the corner NOC commands reproducibly corrupts two isolated output
// lanes, independent of whether the unused slots are ignored or locally
// zeroed.  Preserve the run52 nine-command cadence with a 32-byte probe read
// for each omitted A-low page, then zero the corresponding tile after the NOC
// barrier.  This keeps almost all of the intended bandwidth reduction while
// the compute kernel consumes zeros with run52's exact per-term MAC cadence.
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

inline bool retains_alow(uint32_t term) {
    const uint32_t offset_id = term / 3;
    const uint32_t di = offset_id / 9;
    const uint32_t dj = (offset_id / 3) % 3;
    const uint32_t dk = offset_id % 3;
    const uint32_t manhattan =
        (di > 1 ? di - 1 : 1 - di) +
        (dj > 1 ? dj - 1 : 1 - dj) +
        (dk > 1 ? dk - 1 : 1 - dk);
    return manhattan <= 2;
}

inline void zero_bf16_tile(uint32_t destination) {
    constexpr uint32_t tile_bytes = 2048;
    tt_l1_ptr uint64_t* words = (tt_l1_ptr uint64_t*)destination;
    for (uint32_t index = 0; index < tile_bytes / sizeof(uint64_t); ++index) {
        words[index] = 0;
    }
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
    static_assert(terms % terms_per_publish == 0);

    constexpr auto g0 = TensorAccessorArgs<3>();
    const auto a0 = TensorAccessor(g0, a0_addr, tile_bytes);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto a1 = TensorAccessor(g1, a1_addr, tile_bytes);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto a2 = TensorAccessor(g2, a2_addr, tile_bytes);

    const uint32_t halo_tiles = n_groups * halo_tiles_per_group;
    cb_wait_front(cb_halo, halo_tiles);
    const uint32_t halo_l1 = get_read_ptr(cb_halo);
    tt_l1_ptr uint64_t* halo_words = (tt_l1_ptr uint64_t*)halo_l1;

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t local_group = start_group + group_slot;
        for (uint32_t stream_batch = 0;
             stream_batch < terms / terms_per_publish;
             ++stream_batch) {
            cb_reserve_back(cb_a, terms_per_publish * 9);
            cb_reserve_back(cb_bhi, terms_per_publish);
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
                const bool retain_low = retains_alow(term);
                const uint32_t a_index = batch_lane * 9;
                for (uint32_t output_component = 0;
                     output_component < 3;
                     ++output_component) {
                    const uint32_t a_page =
                        (local_group * terms + term) * 3 + output_component;
                    noc_async_read_page(
                        a_page,
                        a0,
                        pa + (a_index + output_component) * tile_bytes);
                    noc_async_read_page(
                        a_page,
                        a1,
                        pa + (a_index + 3 + output_component) * tile_bytes);
                    const uint32_t low_destination =
                        pa + (a_index + 6 + output_component) * tile_bytes;
                    if (retain_low) {
                        noc_async_read_page(
                            a_page,
                            a2,
                            low_destination);
                    } else {
                        // Preserve the proven NOC command cadence without
                        // transferring the unused 2 KiB corner page.
                        noc_async_read(a2.get_noc_addr(a_page), low_destination, 32);
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
            for (uint32_t batch_lane = 0;
                 batch_lane < terms_per_publish;
                 ++batch_lane) {
                const uint32_t stream_term =
                    stream_batch * terms_per_publish + batch_lane;
                const uint32_t term = ordered_term(stream_term, order_mode);
                if (!retains_alow(term)) {
                    const uint32_t a_index = batch_lane * 9;
                    for (uint32_t output_component = 0;
                         output_component < 3;
                         ++output_component) {
                        zero_bf16_tile(
                            pa + (a_index + 6 + output_component) * tile_bytes);
                    }
                }
            }
            cb_push_back(cb_a, terms_per_publish * 9);
            cb_push_back(cb_bhi, terms_per_publish);
        }
    }
}
