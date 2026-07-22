// reduce_writer.cpp — write the single fp32 output tile (row sums in column 0) back to DRAM.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>
void kernel_main() {
    const uint32_t dst = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto args = TensorAccessorArgs<1>();
    const auto A = TensorAccessor(args, dst, 32u * 32u * 4u);
    cb_wait_front(cb_out, 1);
    noc_async_write_page(0, A, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
