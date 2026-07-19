// Default-off changing-x RISC-0 reader for canonical base groups.
//
// One vector component is staged at a time from the exact page table.  The
// lane plan then reconstructs the 27 high-split term tiles in component-major
// order while the coefficient palette remains resident.
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/programming_examples/spmv_mac/kernels/canonical_dynamic_page_common.hpp"
#include <cstdint>

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t palette0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t palette1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t palette2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t vector0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lane_map_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t start_group = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_palette = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bhi = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t cb_page_table = get_compile_time_arg_val(3);
    constexpr uint32_t cb_lane_map = get_compile_time_arg_val(4);
    constexpr uint32_t palette_size = 11;
    constexpr uint32_t palette_tiles = 33;

    constexpr auto g0 = TensorAccessorArgs<5>();
    const auto palette0 = TensorAccessor(
        g0, palette0_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto palette1 = TensorAccessor(
        g1, palette1_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto palette2 = TensorAccessor(
        g2, palette2_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();
    const auto vector0 = TensorAccessor(
        g3, vector0_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g4 = TensorAccessorArgs<g3.next_compile_time_args_offset()>();
    const auto page_table = TensorAccessor(
        g4, page_table_addr, canonical_dynamic_page::kPageTableBytes);
    constexpr auto g5 = TensorAccessorArgs<g4.next_compile_time_args_offset()>();
    const auto lane_map = TensorAccessor(
        g5, lane_map_addr, canonical_dynamic_page::kTileBytes);

    // c_2 is dedicated L1 scratch, not a producer/consumer stream.  Reserve
    // its entire allocation once so get_write_ptr is valid, then never push
    // it: page_count varies by group, and advancing a 62-page FIFO by
    // arbitrary counts can cross its end.
    cb_reserve_back(cb_stage, canonical_dynamic_page::kMaxGroupPages);
    const uint32_t stage_l1 = get_write_ptr(cb_stage);
    tt_l1_ptr uint16_t* staged = (tt_l1_ptr uint16_t*)stage_l1;

    cb_reserve_back(cb_palette, palette_tiles);
    const uint32_t palette_l1 = get_write_ptr(cb_palette);
    for (uint32_t page = 0; page < palette_size; ++page) {
        noc_async_read_page(
            page, palette0, palette_l1 + page * canonical_dynamic_page::kTileBytes);
        noc_async_read_page(
            page,
            palette1,
            palette_l1 + (palette_size + page) * canonical_dynamic_page::kTileBytes);
        noc_async_read_page(
            page,
            palette2,
            palette_l1 + (2 * palette_size + page) * canonical_dynamic_page::kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_palette, palette_tiles);

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t packed_group = start_group + group_slot;
        cb_reserve_back(cb_page_table, 1);
        const uint32_t table_l1 = get_write_ptr(cb_page_table);
        noc_async_read_page(packed_group, page_table, table_l1);
        noc_async_read_barrier();
        cb_push_back(cb_page_table, 1);
        cb_wait_front(cb_page_table, 1);
        tt_l1_ptr uint8_t* table = (tt_l1_ptr uint8_t*)get_read_ptr(cb_page_table);
        const uint32_t page_count = table[canonical_dynamic_page::kPageCountByte];

        for (uint32_t component = 0;
             component < canonical_dynamic_page::kComponents;
             ++component) {
            for (uint32_t page_slot = 0; page_slot < page_count; ++page_slot) {
                const uint32_t vector_page =
                    component * canonical_dynamic_page::kVectorPagesPerComponent +
                    table[page_slot];
                noc_async_read_page(
                    vector_page,
                    vector0,
                    stage_l1 + page_slot * canonical_dynamic_page::kTileBytes);
            }
            noc_async_read_barrier();

            for (uint32_t offset_batch = 0;
                 offset_batch < canonical_dynamic_page::kOffsets;
                 offset_batch += canonical_dynamic_page::kTermsPerPublish) {
                cb_reserve_back(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
                const uint32_t map_l1 = get_write_ptr(cb_lane_map);
                for (uint32_t batch_lane = 0;
                     batch_lane < canonical_dynamic_page::kTermsPerPublish;
                     ++batch_lane) {
                    const uint32_t stream_term =
                        component * canonical_dynamic_page::kOffsets +
                        offset_batch + batch_lane;
                    const uint32_t term =
                        canonical_dynamic_page::ordered_term(stream_term);
                    const uint32_t offset = term / canonical_dynamic_page::kComponents;
                    noc_async_read_page(
                        packed_group * canonical_dynamic_page::kOffsets + offset,
                        lane_map,
                        map_l1 + batch_lane * canonical_dynamic_page::kTileBytes);
                }
                noc_async_read_barrier();
                cb_push_back(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
                cb_wait_front(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
                cb_reserve_back(cb_bhi, canonical_dynamic_page::kTermsPerPublish);
                const uint32_t destination = get_write_ptr(cb_bhi);
                tt_l1_ptr uint16_t* maps =
                    (tt_l1_ptr uint16_t*)get_read_ptr(cb_lane_map);
                for (uint32_t batch_lane = 0;
                     batch_lane < canonical_dynamic_page::kTermsPerPublish;
                     ++batch_lane) {
                    canonical_dynamic_page::gather_b_tile(
                        destination + batch_lane * canonical_dynamic_page::kTileBytes,
                        staged,
                        maps + batch_lane * 1024);
                }
                cb_push_back(cb_bhi, canonical_dynamic_page::kTermsPerPublish);
                cb_pop_front(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
            }
        }
        cb_pop_front(cb_page_table, 1);
    }
}
