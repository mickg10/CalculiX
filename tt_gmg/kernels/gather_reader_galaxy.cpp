// gather_reader.cpp — REDUNDANT per-tile-window fused-SpMV reader. Every core computes ALL n_out(=n_local) tiles
// of its chip's shard; last-writer-wins to the sharded c is benign (all cores produce identical values), so on a
// runtime that dispatches only a few cores/chip, whichever DO run still fill the whole shard -> full gather.
// Each tile's x-window is computed ON-DEVICE from that tile's nbr slice (min/max neighbor element) and staged PER
// TILE, so L1 holds only one tile's window (~40 tiles) + one tile's nbr (~12 pages), not the whole span. Args:
// [a_ah,a_am,a_al, a_xh,a_xm,a_xl, a_nbr, n_out(=n_local), K, tile_base(GLOBAL out-tile of this chip's shard),
//  0,0, NBn(node cap), 0].  a is REPLICATED (global page), c is SHARDED (writer uses LOCAL start=0..n_out).
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t ar = 0;
    const uint32_t a_ah = get_arg_val<uint32_t>(ar++), a_am = get_arg_val<uint32_t>(ar++), a_al = get_arg_val<uint32_t>(ar++);
    const uint32_t a_xh = get_arg_val<uint32_t>(ar++), a_xm = get_arg_val<uint32_t>(ar++), a_xl = get_arg_val<uint32_t>(ar++);
    const uint32_t a_nbr = get_arg_val<uint32_t>(ar++);
    const uint32_t n_out = get_arg_val<uint32_t>(ar++), K = get_arg_val<uint32_t>(ar++), tile_base = get_arg_val<uint32_t>(ar++);
    (void)get_arg_val<uint32_t>(ar++); (void)get_arg_val<uint32_t>(ar++);   // (old xwin_tile_lo/ntiles: per-tile computed now)
    const uint32_t NBn = get_arg_val<uint32_t>(ar++); (void)get_arg_val<uint32_t>(ar++);  // node cap; (old n_nodes unused)

    constexpr uint32_t cb_a = get_compile_time_arg_val(0), cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_xh = get_compile_time_arg_val(2), cb_xm = get_compile_time_arg_val(3), cb_xl = get_compile_time_arg_val(4);
    constexpr uint32_t cb_nbr = get_compile_time_arg_val(5);
    constexpr uint32_t TB = 2048;                             // bf16 32x32 tile bytes
    constexpr uint32_t NBTB = 4096;                           // int32 page (1024 int32)

    constexpr auto g0 = TensorAccessorArgs<6>();                                   const auto Aah = TensorAccessor(g0, a_ah, TB);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();  const auto Aam = TensorAccessor(g1, a_am, TB);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();  const auto Aal = TensorAccessor(g2, a_al, TB);
    constexpr auto g3 = TensorAccessorArgs<g2.next_compile_time_args_offset()>();  const auto Xh  = TensorAccessor(g3, a_xh, TB);
    constexpr auto g4 = TensorAccessorArgs<g3.next_compile_time_args_offset()>();  const auto Xm  = TensorAccessor(g4, a_xm, TB);
    constexpr auto g5 = TensorAccessorArgs<g4.next_compile_time_args_offset()>();  const auto Xl  = TensorAccessor(g5, a_xl, TB);
    constexpr auto g6 = TensorAccessorArgs<g5.next_compile_time_args_offset()>();  const auto Nbr = TensorAccessor(g6, a_nbr, 4096);

    for (uint32_t t = 0; t < n_out; ++t) {
        const uint32_t gt = tile_base + t;                    // GLOBAL out-tile
        const uint32_t e0 = gt * 1024;
        uint32_t node0 = e0 / 3, rr0 = e0 - node0 * 3;
        // padding tiles beyond the real problem: a is zero-padded there so output is ignored; clamp reads in-bounds.
        if (node0 >= NBn) { node0 = (NBn > 342) ? (NBn - 342) : 0; rr0 = 0; }
        uint32_t node_hi = node0 + 341; if (node_hi >= NBn) node_hi = NBn - 1;
        const uint32_t nn_t = (node_hi >= node0) ? (node_hi - node0 + 1) : 1;

        // ---- stage THIS tile's nbr slice ----
        const uint32_t slice_int_lo = node0 * 27;
        const uint32_t page_lo = slice_int_lo / 1024;
        const uint32_t npages = (slice_int_lo + nn_t * 27 + 1023) / 1024 - page_lo;
        cb_reserve_back(cb_nbr, npages);
        const uint32_t nbr0 = get_write_ptr(cb_nbr);
        for (uint32_t j = 0; j < npages; ++j) noc_async_read_page(page_lo + j, Nbr, nbr0 + j * NBTB);
        noc_async_read_barrier();
        cb_push_back(cb_nbr, npages);
        tt_l1_ptr int32_t* NB = (tt_l1_ptr int32_t*)nbr0;
        const uint32_t nbr_base = slice_int_lo - page_lo * 1024;

        // ---- compute THIS tile's x-window from nbr min/max ----
        int32_t emin = 0x7fffffff, emax = -1;
        for (uint32_t nd = 0; nd < nn_t; ++nd) for (uint32_t o = 0; o < 27; ++o) {
            const int32_t nn = NB[nbr_base + nd * 27 + o];
            if (nn >= 0) { const int32_t e = 3 * nn; if (e < emin) emin = e; if (e + 2 > emax) emax = e + 2; }
        }
        if (emax < 0) { emin = 0; emax = 0; }
        const uint32_t xwin_tile_lo = (uint32_t)emin / 1024;
        const uint32_t xwin_ntiles = (uint32_t)emax / 1024 - xwin_tile_lo + 1;
        const uint32_t xwin_elem_lo = xwin_tile_lo * 1024;

        // ---- stage THIS tile's x-window (replicated x, global page) ----
        cb_reserve_back(cb_xh, xwin_ntiles); cb_reserve_back(cb_xm, xwin_ntiles); cb_reserve_back(cb_xl, xwin_ntiles);
        const uint32_t xh0 = get_write_ptr(cb_xh), xm0 = get_write_ptr(cb_xm), xl0 = get_write_ptr(cb_xl);
        for (uint32_t j = 0; j < xwin_ntiles; ++j) {
            const uint32_t p = xwin_tile_lo + j;
            noc_async_read_page(p, Xh, xh0 + j * TB); noc_async_read_page(p, Xm, xm0 + j * TB); noc_async_read_page(p, Xl, xl0 + j * TB);
        }
        noc_async_read_barrier();
        cb_push_back(cb_xh, xwin_ntiles); cb_push_back(cb_xm, xwin_ntiles); cb_push_back(cb_xl, xwin_ntiles);
        tt_l1_ptr uint16_t* XH = (tt_l1_ptr uint16_t*)xh0;
        tt_l1_ptr uint16_t* XM = (tt_l1_ptr uint16_t*)xm0;
        tt_l1_ptr uint16_t* XL = (tt_l1_ptr uint16_t*)xl0;

        // ---- stream a[gt,k] (replicated, GLOBAL page) + gather b[k] per k ----
        const uint32_t base = gt * K;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t oo = k / 3, c = k % 3;
            cb_reserve_back(cb_a, 3); cb_reserve_back(cb_b, 3);
            const uint32_t pa = get_write_ptr(cb_a), pb = get_write_ptr(cb_b), p = base + k;
            noc_async_read_page(p, Aah, pa + 0 * TB); noc_async_read_page(p, Aam, pa + 1 * TB); noc_async_read_page(p, Aal, pa + 2 * TB);
            tt_l1_ptr uint16_t* BH = (tt_l1_ptr uint16_t*)(pb + 0 * TB);
            tt_l1_ptr uint16_t* BM = (tt_l1_ptr uint16_t*)(pb + 1 * TB);
            tt_l1_ptr uint16_t* BL = (tt_l1_ptr uint16_t*)(pb + 2 * TB);
            uint32_t node = node0, rr = rr0;
            for (uint32_t i = 0; i < 1024; ++i) {
                const int32_t nn = ((node - node0) < nn_t) ? NB[nbr_base + (node - node0) * 27 + oo] : -1;
                if (nn < 0) { BH[i] = 0; BM[i] = 0; BL[i] = 0; }
                else { const uint32_t s = (uint32_t)(3 * nn + c) - xwin_elem_lo; BH[i] = XH[s]; BM[i] = XM[s]; BL[i] = XL[s]; }
                if (++rr == 3) { rr = 0; node++; }
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, 3); cb_push_back(cb_b, 3);
        }
        // ---- free THIS tile's nbr + x-window for the next tile ----
        cb_pop_front(cb_nbr, npages);
        cb_pop_front(cb_xh, xwin_ntiles); cb_pop_front(cb_xm, xwin_ntiles); cb_pop_front(cb_xl, xwin_ntiles);
    }
}
