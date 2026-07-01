// reduce_reader.cpp — seed a bf16 1.0 reduce-scaler tile into cb_sc, then stream nt fp32 product tiles into cb_in.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>
void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0), nt = get_arg_val<uint32_t>(1), scaler = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0), cb_sc = get_compile_time_arg_val(1);
    // inlined wh_generate_reduce_scaler: fill first face row with `scaler`, replicate to the 4 faces
    cb_reserve_back(cb_sc, 1);
    uint32_t sw = get_write_ptr(cb_sc);
    volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sw);
    for (int i = 0; i < 512; ++i) sp[i] = 0;
    for (int j = 0; j < 8; ++j) sp[j] = scaler;
    uint64_t na = get_noc_addr(sw);
    noc_async_read_one_packet_set_state(na, 32);
    noc_async_read_one_packet_with_state(na, sw + (1 << 9));
    noc_async_read_one_packet_with_state(na, sw + (2 << 9));
    noc_async_read_one_packet_with_state(na, sw + (3 << 9));
    noc_async_read_barrier();
    cb_push_back(cb_sc, 1);
    constexpr auto args = TensorAccessorArgs<2>();
    const auto A = TensorAccessor(args, src, 32u * 32u * 4u);   // fp32 tile page = 4096 B
    for (uint32_t i = 0; i < nt; ++i) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_page(i, A, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
