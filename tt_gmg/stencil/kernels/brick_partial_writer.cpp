// Write nine fp32 chunk/component partial tiles per brick group.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_groups = get_arg_val<uint32_t>(1);
    const uint32_t start_group = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t partial_tiles_per_group = 9;
    constexpr auto output_args = TensorAccessorArgs<1>();
    const auto output = TensorAccessor(output_args, output_addr, 4096);

    for (uint32_t group = 0; group < n_groups; ++group) {
        cb_wait_front(cb_out, partial_tiles_per_group);
        const uint32_t source = get_read_ptr(cb_out);
        const uint32_t page0 =
            (start_group + group) * partial_tiles_per_group;
        for (uint32_t page = 0; page < partial_tiles_per_group; ++page) {
            noc_async_write_page(page0 + page, output, source + page * 4096);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, partial_tiles_per_group);
    }
}
