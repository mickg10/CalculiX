// gather_reader_matmul.cpp — reader for the matmul-diagonal fp32-accumulate DIA MAC.
// Emits, per 32-element output group: KT tiles of A ([32 elem, 32 k], read from host-transposed resident a) and
// KT tiles of B ([32 k, 32 elem], gathered: B[k,e] = x[3*nbr[k_off,node(e)]+c]). Also stages an identity tile once.
// A is [element, k]-major in DRAM (host transposed a_mm[group][kt][32*32], row=elem, col=k). B is built here by
// gathering each k's b for the 32 elements into COLUMN e, ROW k. The compute does diag(A@B) with fp32 K-accumulate.
// Args: [a_mm, a_xh,a_xm,a_xl, a_nbr, n_grp, KT, g_elem0(GLOBAL first element), xwin_tile_lo, xwin_ntiles,
//        node_lo, n_nodes, id_addr]. Single-bf16 b (plain matmul-diagonal: 11-bit products = 4.69e-4, within the
//        eig-deflation tolerance). a_mm holds a single-bf16 coefficient per (elem,k).
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t ar = 0;
    const uint32_t a_mm = get_arg_val<uint32_t>(ar++);
    const uint32_t a_xh = get_arg_val<uint32_t>(ar++), a_xm = get_arg_val<uint32_t>(ar++), a_xl = get_arg_val<uint32_t>(ar++);
    const uint32_t a_nbr = get_arg_val<uint32_t>(ar++);
    const uint32_t n_grp = get_arg_val<uint32_t>(ar++), KT = get_arg_val<uint32_t>(ar++), g_elem0 = get_arg_val<uint32_t>(ar++);
    const uint32_t xwin_tile_lo = get_arg_val<uint32_t>(ar++), xwin_ntiles = get_arg_val<uint32_t>(ar++);
    const uint32_t node_lo = get_arg_val<uint32_t>(ar++), n_nodes = get_arg_val<uint32_t>(ar++);
    const uint32_t id_addr = get_arg_val<uint32_t>(ar++);

    constexpr uint32_t cb_A = get_compile_time_arg_val(0), cb_B = get_compile_time_arg_val(1);
    constexpr uint32_t cb_xh = get_compile_time_arg_val(2), cb_xm = get_compile_time_arg_val(3), cb_xl = get_compile_time_arg_val(4);
    constexpr uint32_t cb_nbr = get_compile_time_arg_val(5), cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t TB = 2048, NBTB = 4096;
    const uint32_t xwin_elem_lo = xwin_tile_lo * 1024;

    constexpr auto g0 = TensorAccessorArgs<7>();                                   const auto Amm = TensorAccessor(g0, a_mm, TB);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();  const auto Xh  = TensorAccessor(g1, a_xh, TB);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();  const auto Xm  = TensorAccessor(g2, a_xm, TB);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();  const auto Xl  = TensorAccessor(g3, a_xl, TB);
    constexpr auto g4 = TensorAccessorArgs<g3.next_compile_time_args_offset()>();  const auto Nbr = TensorAccessor(g4, a_nbr, NBTB);

    // ---- stage identity mask (1 tile) + x-window + nbr slice resident, once ----
    cb_reserve_back(cb_id, 1); noc_async_read_page(0, TensorAccessor(TensorAccessorArgs<7>(), id_addr, TB), get_write_ptr(cb_id));
    cb_reserve_back(cb_xh, xwin_ntiles); cb_reserve_back(cb_xm, xwin_ntiles); cb_reserve_back(cb_xl, xwin_ntiles);
    const uint32_t xh0 = get_write_ptr(cb_xh), xm0 = get_write_ptr(cb_xm), xl0 = get_write_ptr(cb_xl);
    for (uint32_t j = 0; j < xwin_ntiles; ++j) { const uint32_t p = xwin_tile_lo + j;
        noc_async_read_page(p, Xh, xh0 + j*TB); noc_async_read_page(p, Xm, xm0 + j*TB); noc_async_read_page(p, Xl, xl0 + j*TB); }
    const uint32_t slice_int_lo = node_lo * 27, page_lo = slice_int_lo / 1024;
    const uint32_t npages = (slice_int_lo + n_nodes*27 + 1023)/1024 - page_lo;
    cb_reserve_back(cb_nbr, npages); const uint32_t nbr0 = get_write_ptr(cb_nbr);
    for (uint32_t j = 0; j < npages; ++j) noc_async_read_page(page_lo + j, Nbr, nbr0 + j*NBTB);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1); cb_push_back(cb_xh, xwin_ntiles); cb_push_back(cb_xm, xwin_ntiles); cb_push_back(cb_xl, xwin_ntiles); cb_push_back(cb_nbr, npages);
    tt_l1_ptr uint16_t* XH=(tt_l1_ptr uint16_t*)xh0; tt_l1_ptr int32_t* NB=(tt_l1_ptr int32_t*)nbr0;
    const uint32_t nbr_base = slice_int_lo - page_lo*1024;

    // per group g: A from a_mm (KT tiles, already [elem,k]); B gathered ([k,elem]).  K (real terms) = KT*32 padded.
    for (uint32_t g = 0; g < n_grp; ++g) {
        for (uint32_t kt = 0; kt < KT; ++kt) {
            cb_reserve_back(cb_A, 1); noc_async_read_page(g*KT + kt, Amm, get_write_ptr(cb_A));  // A tile [32 elem,32 k]
            cb_reserve_back(cb_B, 1); const uint32_t pb = get_write_ptr(cb_B);
            tt_l1_ptr uint16_t* B = (tt_l1_ptr uint16_t*)pb;                                      // B tile [32 k,32 elem]
            for (uint32_t kk = 0; kk < 32; ++kk) { const uint32_t k = kt*32 + kk;                 // DIA term index = o*3+c
                const uint32_t oo = k/3, cc = k%3;                                                // offset, component
                for (uint32_t e = 0; e < 32; ++e) { const uint32_t ge = g_elem0 + g*32 + e;       // GLOBAL output element
                    const uint32_t node = ge/3;                                                   // output node (r=ge%3 unused: cf holds it)
                    int32_t nn = (k < 81 && node>=node_lo && (node-node_lo)<n_nodes) ? NB[nbr_base + (node-node_lo)*27 + oo] : -1;
                    // B[k,e] -> TILED offset (row=kk, col=e): 4x16x16 faces
                    const uint32_t off = ((kk/16)*2 + (e/16))*256 + (kk%16)*16 + (e%16);
                    B[off] = (nn<0) ? 0 : XH[(uint32_t)(3*nn + cc) - xwin_elem_lo];                // B[k,e]=x[3*nbr[o,node]+c]; k>=81 pad 0
                } }
            noc_async_read_barrier();
            cb_push_back(cb_A, 1); cb_push_back(cb_B, 1);
        }
    }
}
