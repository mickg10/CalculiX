// Default-off changing-x RISC-1 reader/writer for canonical base groups.
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/programming_examples/spmv_mac/kernels/canonical_dynamic_page_common.hpp"
#include <cstdint>

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t vector1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t vector2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lane_map_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t start_group = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_bmid = get_compile_time_arg_val(0);
    constexpr uint32_t cb_blow = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stage_mid = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stage_low = get_compile_time_arg_val(4);
    constexpr uint32_t cb_page_table = get_compile_time_arg_val(5);
    constexpr uint32_t cb_lane_map = get_compile_time_arg_val(6);

    constexpr auto g0 = TensorAccessorArgs<7>();
    const auto vector1 = TensorAccessor(
        g0, vector1_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto vector2 = TensorAccessor(
        g1, vector2_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto page_table = TensorAccessor(
        g2, page_table_addr, canonical_dynamic_page::kPageTableBytes);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();
    const auto lane_map = TensorAccessor(
        g3, lane_map_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g4 = TensorAccessorArgs<g3.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(g4, output_addr, 4096);

    // c_6/c_7 are fixed-address L1 scratch.  Reserve each complete allocation
    // once so get_write_ptr is valid, then never push either CB.  They must
    // not advance as variable-count FIFOs because that can cross page 62.
    cb_reserve_back(cb_stage_mid, canonical_dynamic_page::kMaxGroupPages);
    cb_reserve_back(cb_stage_low, canonical_dynamic_page::kMaxGroupPages);
    const uint32_t mid_stage_l1 = get_write_ptr(cb_stage_mid);
    const uint32_t low_stage_l1 = get_write_ptr(cb_stage_low);
    tt_l1_ptr uint16_t* staged_mid =
        (tt_l1_ptr uint16_t*)mid_stage_l1;
    tt_l1_ptr uint16_t* staged_low =
        (tt_l1_ptr uint16_t*)low_stage_l1;

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
                    vector1,
                    mid_stage_l1 + page_slot * canonical_dynamic_page::kTileBytes);
                noc_async_read_page(
                    vector_page,
                    vector2,
                    low_stage_l1 + page_slot * canonical_dynamic_page::kTileBytes);
            }
            noc_async_read_barrier();

            for (uint32_t offset_batch = 0;
                 offset_batch < canonical_dynamic_page::kOffsets;
                 offset_batch += canonical_dynamic_page::kTermsPerPublish) {
                cb_reserve_back(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
                const uint32_t map_l1 = get_write_ptr(cb_lane_map);
                uint32_t low_count = 0;
                for (uint32_t batch_lane = 0;
                     batch_lane < canonical_dynamic_page::kTermsPerPublish;
                     ++batch_lane) {
                    const uint32_t stream_term =
                        component * canonical_dynamic_page::kOffsets +
                        offset_batch + batch_lane;
                    const uint32_t term =
                        canonical_dynamic_page::ordered_term(stream_term);
                    const uint32_t offset = term / canonical_dynamic_page::kComponents;
                    low_count += canonical_dynamic_page::retains_blow(term) ? 1u : 0u;
                    noc_async_read_page(
                        packed_group * canonical_dynamic_page::kOffsets + offset,
                        lane_map,
                        map_l1 + batch_lane * canonical_dynamic_page::kTileBytes);
                }
                noc_async_read_barrier();
                const uint32_t low_publish_count =
                    canonical_dynamic_page::low_publish_count(low_count);
                cb_push_back(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
                cb_wait_front(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
                cb_reserve_back(cb_bmid, canonical_dynamic_page::kTermsPerPublish);
                if (low_publish_count != 0) {
                    cb_reserve_back(cb_blow, low_publish_count);
                }
                const uint32_t mid_destination = get_write_ptr(cb_bmid);
                const uint32_t low_destination =
                    low_count == 0 ? 0 : get_write_ptr(cb_blow);
                tt_l1_ptr uint16_t* maps =
                    (tt_l1_ptr uint16_t*)get_read_ptr(cb_lane_map);
                uint32_t low_index = 0;
                for (uint32_t batch_lane = 0;
                     batch_lane < canonical_dynamic_page::kTermsPerPublish;
                     ++batch_lane) {
                    const uint32_t stream_term =
                        component * canonical_dynamic_page::kOffsets +
                        offset_batch + batch_lane;
                    const uint32_t term =
                        canonical_dynamic_page::ordered_term(stream_term);
                    canonical_dynamic_page::gather_b_tile(
                        mid_destination + batch_lane * canonical_dynamic_page::kTileBytes,
                        staged_mid,
                        maps + batch_lane * 1024);
                    if (canonical_dynamic_page::retains_blow(term)) {
                        canonical_dynamic_page::gather_b_tile(
                            low_destination + low_index * canonical_dynamic_page::kTileBytes,
                            staged_low,
                            maps + batch_lane * 1024);
                        ++low_index;
                    }
                }
                cb_push_back(cb_bmid, canonical_dynamic_page::kTermsPerPublish);
                if (low_publish_count != 0) {
                    cb_push_back(cb_blow, low_publish_count);
                }
                cb_pop_front(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
            }
        }
        cb_pop_front(cb_page_table, 1);
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
}
