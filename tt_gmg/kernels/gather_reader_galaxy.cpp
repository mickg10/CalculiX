// gather_reader.cpp — NON-REDUNDANT per-core fused-SpMV reader for the FULL multi-chip sharded gather.
// Each core does its npc tiles (split_work_to_cores). GLOBAL index g_start drives node0/nbr/x (replicated, full);
// LOCAL index local_start drives the sharded a-page (a-shard on this chip). This is the proven spmv_mac reader
// (one staged window per core) generalized so per-chip tile_base != 0 works: the proven version only tested
// tile_base=0 (chip-local subset), where local==global. Args: [a_ah,a_am,a_al, a_xh,a_xm,a_xl, a_nbr,
// n_out(=npc), K, g_start(GLOBAL), xwin_tile_lo, xwin_ntiles, node_lo, n_nodes, local_start(LOCAL a-page)].
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t ar = 0;
    const uint32_t a_ah = get_arg_val<uint32_t>(ar++), a_am = get_arg_val<uint32_t>(ar++), a_al = get_arg_val<uint32_t>(ar++);
    const uint32_t a_xh = get_arg_val<uint32_t>(ar++), a_xm = get_arg_val<uint32_t>(ar++), a_xl = get_arg_val<uint32_t>(ar++);
    const uint32_t a_nbr = get_arg_val<uint32_t>(ar++);
    const uint32_t n_out = get_arg_val<uint32_t>(ar++), K = get_arg_val<uint32_t>(ar++), g_start = get_arg_val<uint32_t>(ar++);
    const uint32_t xwin_tile_lo = get_arg_val<uint32_t>(ar++), xwin_ntiles = get_arg_val<uint32_t>(ar++);
    const uint32_t node_lo = get_arg_val<uint32_t>(ar++), n_nodes = get_arg_val<uint32_t>(ar++);
    const uint32_t local_start = get_arg_val<uint32_t>(ar++);   // LOCAL shard tile offset (a-page); a is SHARDED

    constexpr uint32_t cb_a = get_compile_time_arg_val(0), cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_xh = get_compile_time_arg_val(2), cb_xm = get_compile_time_arg_val(3), cb_xl = get_compile_time_arg_val(4);
    constexpr uint32_t cb_nbr = get_compile_time_arg_val(5);
    constexpr uint32_t TB = 2048;
    const uint32_t xwin_elem_lo = xwin_tile_lo * 1024;

    constexpr auto g0 = TensorAccessorArgs<6>();                                   const auto Aah = TensorAccessor(g0, a_ah, TB);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();  const auto Aam = TensorAccessor(g1, a_am, TB);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();  const auto Aal = TensorAccessor(g2, a_al, TB);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();  const auto Xh  = TensorAccessor(g3, a_xh, TB);
    constexpr auto g4 = TensorAccessorArgs<g3.next_compile_time_args_offset()>();  const auto Xm  = TensorAccessor(g4, a_xm, TB);
    constexpr auto g5 = TensorAccessorArgs<g4.next_compile_time_args_offset()>();  const auto Xl  = TensorAccessor(g5, a_xl, TB);
    constexpr auto g6 = TensorAccessorArgs<g5.next_compile_time_args_offset()>();  const auto Nbr = TensorAccessor(g6, a_nbr, 4096);

    // ---- stage x-window (bf16 tiles) + nbr node-slice (int32) resident in L1 (filled once for this core) ----
    cb_reserve_back(cb_xh, xwin_ntiles); cb_reserve_back(cb_xm, xwin_ntiles); cb_reserve_back(cb_xl, xwin_ntiles);
    const uint32_t xh0 = get_write_ptr(cb_xh), xm0 = get_write_ptr(cb_xm), xl0 = get_write_ptr(cb_xl);
    for (uint32_t j = 0; j < xwin_ntiles; ++j) {
        const uint32_t p = xwin_tile_lo + j;
        noc_async_read_page(p, Xh, xh0 + j * TB); noc_async_read_page(p, Xm, xm0 + j * TB); noc_async_read_page(p, Xl, xl0 + j * TB);
    }
    constexpr uint32_t NBTB = 4096;
    const uint32_t slice_int_lo = node_lo * 27;
    const uint32_t page_lo = slice_int_lo / 1024;
    const uint32_t npages = (slice_int_lo + n_nodes * 27 + 1023) / 1024 - page_lo;
    cb_reserve_back(cb_nbr, npages);
    const uint32_t nbr0 = get_write_ptr(cb_nbr);
    for (uint32_t j = 0; j < npages; ++j) noc_async_read_page(page_lo + j, Nbr, nbr0 + j * NBTB);
    noc_async_read_barrier();
    cb_push_back(cb_xh, xwin_ntiles); cb_push_back(cb_xm, xwin_ntiles); cb_push_back(cb_xl, xwin_ntiles); cb_push_back(cb_nbr, npages);

    tt_l1_ptr uint16_t* XH = (tt_l1_ptr uint16_t*)xh0;
    tt_l1_ptr uint16_t* XM = (tt_l1_ptr uint16_t*)xm0;
    tt_l1_ptr uint16_t* XL = (tt_l1_ptr uint16_t*)xl0;
    tt_l1_ptr int32_t*  NB = (tt_l1_ptr int32_t*)nbr0;
    const uint32_t nbr_base = slice_int_lo - page_lo * 1024;

    for (uint32_t t = 0; t < n_out; ++t) {
        const uint32_t e0 = (g_start + t) * 1024;            // GLOBAL out-element base -> node0/nbr/x
        const uint32_t base = (local_start + t) * K;         // LOCAL shard a-page (a is SHARDED on this chip)
        const uint32_t node0 = e0 / 3, rr0 = e0 - node0 * 3;
        // OFFSET-GROUPED: process the 3 components c=0,1,2 of one DIA offset oo together (k = oo*3 + c, so k
        // steps 0..80 in groups of 3). The 3 components share ONE nbr node nn = nbr[oo,node], and their source x
        // elements 3*nn+{0,1,2} are CONTIGUOUS in the window. Wins vs per-k: (1) ONE noc_async_read_barrier per 3
        // k's (243 barriers/core, not 729 -> kills the DRAM-latency stall that dominated the 44ms), (2) the NB[]
        // lookup + (node-node_lo)*27 index math happen ONCE per (node,oo) not per (node,k), (3) contiguous reads.
        // K=81=27*3 is structural (27-pt stencil x 3 dof), so groups of 3 tile exactly. cb_a/cb_b depth>=9.
        for (uint32_t og = 0; og < K; og += 3) {
            const uint32_t oo = og / 3;
            cb_reserve_back(cb_a, 9); cb_reserve_back(cb_b, 9);
            const uint32_t pa = get_write_ptr(cb_a), pb = get_write_ptr(cb_b);
            for (uint32_t kk = 0; kk < 3; ++kk) { const uint32_t p = base + og + kk;
                noc_async_read_page(p, Aah, pa + (kk * 3 + 0) * TB);
                noc_async_read_page(p, Aam, pa + (kk * 3 + 1) * TB);
                noc_async_read_page(p, Aal, pa + (kk * 3 + 2) * TB); }
            tt_l1_ptr uint16_t* BH0 = (tt_l1_ptr uint16_t*)(pb + 0 * TB); tt_l1_ptr uint16_t* BM0 = (tt_l1_ptr uint16_t*)(pb + 1 * TB); tt_l1_ptr uint16_t* BL0 = (tt_l1_ptr uint16_t*)(pb + 2 * TB);
            tt_l1_ptr uint16_t* BH1 = (tt_l1_ptr uint16_t*)(pb + 3 * TB); tt_l1_ptr uint16_t* BM1 = (tt_l1_ptr uint16_t*)(pb + 4 * TB); tt_l1_ptr uint16_t* BL1 = (tt_l1_ptr uint16_t*)(pb + 5 * TB);
            tt_l1_ptr uint16_t* BH2 = (tt_l1_ptr uint16_t*)(pb + 6 * TB); tt_l1_ptr uint16_t* BM2 = (tt_l1_ptr uint16_t*)(pb + 7 * TB); tt_l1_ptr uint16_t* BL2 = (tt_l1_ptr uint16_t*)(pb + 8 * TB);
            uint32_t node = node0, i = 0, rr = rr0;
            while (i < 1024) {
                const int32_t nn = (node >= node_lo && (node - node_lo) < n_nodes) ? NB[nbr_base + (node - node_lo) * 27 + oo] : -1;
                uint16_t h0,m0,l0,h1,m1,l1,h2,m2,l2;
                if (nn < 0) { h0=m0=l0=h1=m1=l1=h2=m2=l2=0; }
                else { const uint32_t s0 = (uint32_t)(3 * nn) - xwin_elem_lo;
                       h0 = XH[s0];     m0 = XM[s0];     l0 = XL[s0];
                       h1 = XH[s0 + 1]; m1 = XM[s0 + 1]; l1 = XL[s0 + 1];
                       h2 = XH[s0 + 2]; m2 = XM[s0 + 2]; l2 = XL[s0 + 2]; }
                for (; rr < 3 && i < 1024; ++rr, ++i) {
                    BH0[i]=h0; BM0[i]=m0; BL0[i]=l0;
                    BH1[i]=h1; BM1[i]=m1; BL1[i]=l1;
                    BH2[i]=h2; BM2[i]=m2; BL2[i]=l2; }
                rr = 0; node++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, 9); cb_push_back(cb_b, 9);
        }
    }
}
