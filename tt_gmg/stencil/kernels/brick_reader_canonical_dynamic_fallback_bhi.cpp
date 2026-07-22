// Default-off changing-x RISC-0 reader for exact fallback groups.
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/programming_examples/spmv_mac/kernels/canonical_dynamic_page_common.hpp"
#include <cstdint>

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t a0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t vector0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lane_map_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t a_start_group = get_arg_val<uint32_t>(arg++);
    const uint32_t packed_start_group = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_a_hi_mid = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bhi = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t cb_page_table = get_compile_time_arg_val(3);
    constexpr uint32_t cb_lane_map = get_compile_time_arg_val(4);
    constexpr uint32_t a_tiles_per_term = 6;

    constexpr auto g0 = TensorAccessorArgs<5>();
    const auto a0 = TensorAccessor(g0, a0_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();
    const auto a1 = TensorAccessor(g1, a1_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();
    const auto vector0 = TensorAccessor(
        g2, vector0_addr, canonical_dynamic_page::kTileBytes);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();
    const auto page_table = TensorAccessor(
        g3, page_table_addr, canonical_dynamic_page::kPageTableBytes);
    constexpr auto g4 = TensorAccessorArgs<g3.next_compile_time_args_offset()>();
    const auto lane_map = TensorAccessor(
        g4, lane_map_addr, canonical_dynamic_page::kTileBytes);

    // Dedicated fixed-address L1 scratch.  Reserve all 62 pages once so
    // get_write_ptr is valid, then never push the CB or advance it by the
    // variable per-group page_count.
    cb_reserve_back(cb_stage, canonical_dynamic_page::kMaxGroupPages);
    const uint32_t stage_l1 = get_write_ptr(cb_stage);
    tt_l1_ptr uint16_t* staged = (tt_l1_ptr uint16_t*)stage_l1;

    for (uint32_t group_slot = 0; group_slot < n_groups; ++group_slot) {
        const uint32_t a_group = a_start_group + group_slot;
        const uint32_t packed_group = packed_start_group + group_slot;
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
                    cb_a_hi_mid,
                    canonical_dynamic_page::kTermsPerPublish * a_tiles_per_term);
                cb_reserve_back(
                    cb_lane_map, canonical_dynamic_page::kTermsPerPublish);
                const uint32_t a_l1 = get_write_ptr(cb_a_hi_mid);
                const uint32_t map_l1 = get_write_ptr(cb_lane_map);
                for (uint32_t batch_lane = 0;
                     batch_lane < canonical_dynamic_page::kTermsPerPublish;
                     ++batch_lane) {
                    const uint32_t stream_term =
                        component * canonical_dynamic_page::kOffsets +
                        offset_batch + batch_lane;
                    const uint32_t term =
                        canonical_dynamic_page::ordered_term(stream_term);
                    const uint32_t a_page0 = (a_group * canonical_dynamic_page::kTerms + term) * 3;
                    const uint32_t a_index = batch_lane * a_tiles_per_term;
                    for (uint32_t output_component = 0;
                         output_component < canonical_dynamic_page::kComponents;
                         ++output_component) {
                        noc_async_read_page(
                            a_page0 + output_component,
                            a0,
                            a_l1 + (a_index + output_component) * canonical_dynamic_page::kTileBytes);
                        noc_async_read_page(
                            a_page0 + output_component,
                            a1,
                            a_l1 + (a_index + 3 + output_component) * canonical_dynamic_page::kTileBytes);
                    }
                    const uint32_t offset = term / canonical_dynamic_page::kComponents;
                    noc_async_read_page(
                        packed_group * canonical_dynamic_page::kOffsets + offset,
                        lane_map,
                        map_l1 + batch_lane * canonical_dynamic_page::kTileBytes);
                }
                noc_async_read_barrier();
                cb_push_back(
                    cb_a_hi_mid,
                    canonical_dynamic_page::kTermsPerPublish * a_tiles_per_term);
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
