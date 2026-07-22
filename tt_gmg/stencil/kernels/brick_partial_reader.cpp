// Read nine fp32 chunk/component partial tiles per brick group.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_groups = get_arg_val<uint32_t>(1);
    const uint32_t start_group = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t partial_tiles_per_group = 9;
    constexpr auto input_args = TensorAccessorArgs<1>();
    const auto input = TensorAccessor(input_args, input_addr, 4096);

    for (uint32_t group = 0; group < n_groups; ++group) {
        cb_reserve_back(cb_partial, partial_tiles_per_group);
        const uint32_t destination = get_write_ptr(cb_partial);
        const uint32_t page0 =
            (start_group + group) * partial_tiles_per_group;
        for (uint32_t page = 0; page < partial_tiles_per_group; ++page) {
            noc_async_read_page(page0 + page, input, destination + page * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_partial, partial_tiles_per_group);
    }
}
