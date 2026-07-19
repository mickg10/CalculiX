// Write three fp32 component tiles per brick group.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_groups = get_arg_val<uint32_t>(1);
    const uint32_t start_group = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();
    const auto output = TensorAccessor(output_args, output_addr, 4096);

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
