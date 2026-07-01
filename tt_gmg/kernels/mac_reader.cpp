// mac_reader.cpp — deliver the 3 unique a-terms [ah,am,al] into cb_a and 3 unique b-terms [bh,bm,bl] into cb_b
// per k (each unique tile read ONCE -> no repeated same-page NoC reads). Buffers: A0=ah A1=am A2=al A3=bh A4=bm A5=bl.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a0 = get_arg_val<uint32_t>(0), a1 = get_arg_val<uint32_t>(1), a2 = get_arg_val<uint32_t>(2);
    const uint32_t a3 = get_arg_val<uint32_t>(3), a4 = get_arg_val<uint32_t>(4), a5 = get_arg_val<uint32_t>(5);
    const uint32_t n_out = get_arg_val<uint32_t>(6), K = get_arg_val<uint32_t>(7), start_out_id = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0), cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t TB = 2048;   // bf16 32x32 tile bytes

    constexpr auto g0 = TensorAccessorArgs<2>();                                   const auto Aah = TensorAccessor(g0, a0);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();  const auto Aam = TensorAccessor(g1, a1);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();  const auto Aal = TensorAccessor(g2, a2);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();  const auto Abh = TensorAccessor(g3, a3);
    constexpr auto g4 = TensorAccessorArgs<g3.next_compile_time_args_offset()>();  const auto Abm = TensorAccessor(g4, a4);
    constexpr auto g5 = TensorAccessorArgs<g4.next_compile_time_args_offset()>();  const auto Abl = TensorAccessor(g5, a5);

    for (uint32_t t = 0; t < n_out; ++t) {
        const uint32_t base = (start_out_id + t) * K;
        for (uint32_t k = 0; k < K; ++k) {
            cb_reserve_back(cb_a, 3); cb_reserve_back(cb_b, 3);
            const uint32_t pa = get_write_ptr(cb_a), pb = get_write_ptr(cb_b), p = base + k;
            noc_async_read_page(p, Aah, pa + 0*TB); noc_async_read_page(p, Aam, pa + 1*TB); noc_async_read_page(p, Aal, pa + 2*TB);
            noc_async_read_page(p, Abh, pb + 0*TB); noc_async_read_page(p, Abm, pb + 1*TB); noc_async_read_page(p, Abl, pb + 2*TB);
            noc_async_read_barrier();
            cb_push_back(cb_a, 3); cb_push_back(cb_b, 3);
        }
    }
}
