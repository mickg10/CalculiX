// mac_writer.cpp — data-movement (RISCV_1) kernel: write finished output tiles from cb_out to DRAM.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t c_addr       = get_arg_val<uint32_t>(0);
    const uint32_t n_out        = get_arg_val<uint32_t>(1);
    const uint32_t start_out_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto c_args = TensorAccessorArgs<1>();
    const auto c = TensorAccessor(c_args, c_addr);

    for (uint32_t t = 0; t < n_out; ++t) {
        cb_wait_front(cb_out, 1);
        const uint32_t r = get_read_ptr(cb_out);
        // DIAG (temporary): overwrite output tile with LOCAL out-tile index so host readback shows which cores ran.
        { volatile float* fp = (volatile float*)r; for (uint32_t i = 0; i < 1024; ++i) fp[i] = (float)(start_out_id + t); }
        noc_async_write_page(start_out_id + t, c, r);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
