// gather_reader_deint.cpp — DE-INTERLEAVED fused-SpMV reader (G3 throughput fix, GATHER_DESIGN lines 143-151).
// STATUS: layout proven bit-exact in make_deint_op.py (de-interleaved == interleaved == BCSR, rel 0.0);
//         this device kernel is PENDING JIT/validation on the next TT window (no tt-metal headers off-box).
//
// Why: the interleaved reader (gather_reader.cpp) gathers 3 values/node with a per-element e/3 div ->
// scalar-RISC latency-bound (~60 ms/8-chip). De-interleaving stores x as 3 node-contiguous planes
// X_c[node] (each bf16x3: xh_c/xm_c/xl_c) and the operator as A[(o*3+r)*3+c, node] (243 planes). Then:
//   * a tile = 1024 NODES (not 3*node+r elements) -> node = start_node + i, NO div, NO r-carry.
//   * gather for a fixed (o,c) = ONE value/node: g_oc[i] = X_c[ nbr[o, node_i] ].
//   * the 81 (o,c) gathers for a node-block are REUSED across the 3 output r-planes -> 3x fewer gathers.
// Compute/writer UNCHANGED (element-wise DST-accum over 81 terms/output-tile); we emit, per output tile
// (r, T): 81 pairs a=A[(o*3+r)*3+c, T], b=g_oc[T] (from the per-T gather cache).
//
// Layout (make_deint_op.py --write -> row236_deint_op.bin):  A[243, nb] row-major (page=1024 f32->bf16x3 tiles),
//   X[3, NBpad] planes, nbr[27, NBpad] (repacked node-major nbr2[node*27+o] on host for contiguous core slices).
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t ar = 0;
    // A planes: base page-buffers for the 243-plane operator (streamed per (r,o,c)); one accessor, page = plane*n_out + T.
    const uint32_t a_ah = get_arg_val<uint32_t>(ar++), a_am = get_arg_val<uint32_t>(ar++), a_al = get_arg_val<uint32_t>(ar++);
    // de-interleaved x: 3 planes x 3 levels = 9 buffers (xh/xm/xl for c=0,1,2).
    const uint32_t a_xh0 = get_arg_val<uint32_t>(ar++), a_xm0 = get_arg_val<uint32_t>(ar++), a_xl0 = get_arg_val<uint32_t>(ar++);
    const uint32_t a_xh1 = get_arg_val<uint32_t>(ar++), a_xm1 = get_arg_val<uint32_t>(ar++), a_xl1 = get_arg_val<uint32_t>(ar++);
    const uint32_t a_xh2 = get_arg_val<uint32_t>(ar++), a_xm2 = get_arg_val<uint32_t>(ar++), a_xl2 = get_arg_val<uint32_t>(ar++);
    const uint32_t a_nbr = get_arg_val<uint32_t>(ar++);
    const uint32_t n_out = get_arg_val<uint32_t>(ar++);          // node-tiles this core owns (Y planes tiled over NBpad)
    const uint32_t start_tile = get_arg_val<uint32_t>(ar++);     // first node-tile (global) this core owns
    const uint32_t nnode_tiles = get_arg_val<uint32_t>(ar++);    // NBpad/1024 (per-plane tile count, for A page math)
    const uint32_t xwin_tile_lo = get_arg_val<uint32_t>(ar++), xwin_ntiles = get_arg_val<uint32_t>(ar++);
    const uint32_t node_lo = get_arg_val<uint32_t>(ar++), n_nodes = get_arg_val<uint32_t>(ar++);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0), cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_xh0 = get_compile_time_arg_val(2), cb_xm0 = get_compile_time_arg_val(3), cb_xl0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_xh1 = get_compile_time_arg_val(5), cb_xm1 = get_compile_time_arg_val(6), cb_xl1 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_xh2 = get_compile_time_arg_val(8), cb_xm2 = get_compile_time_arg_val(9), cb_xl2 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_nbr = get_compile_time_arg_val(11);
    constexpr uint32_t cb_bc = get_compile_time_arg_val(12);     // per-T gather cache: 81*3 bf16 tiles (h/m/l for each o,c)
    constexpr uint32_t TB = 2048;                                // bf16 32x32 tile bytes
    constexpr uint32_t K3 = 81;                                  // (o,c) gather terms cached per node-block
    const uint32_t xwin_elem_lo = xwin_tile_lo * 1024;

    constexpr auto g0 = TensorAccessorArgs<13>();                                  const auto Aah = TensorAccessor(g0, a_ah);
    constexpr auto g1 = TensorAccessorArgs<g0.next_compile_time_args_offset()>();  const auto Aam = TensorAccessor(g1, a_am);
    constexpr auto g2 = TensorAccessorArgs<g1.next_compile_time_args_offset()>();  const auto Aal = TensorAccessor(g2, a_al);
    constexpr auto x0h = TensorAccessorArgs<g2.next_compile_time_args_offset()>(); const auto Xh0 = TensorAccessor(x0h, a_xh0);
    constexpr auto x0m = TensorAccessorArgs<x0h.next_compile_time_args_offset()>();const auto Xm0 = TensorAccessor(x0m, a_xm0);
    constexpr auto x0l = TensorAccessorArgs<x0m.next_compile_time_args_offset()>();const auto Xl0 = TensorAccessor(x0l, a_xl0);
    constexpr auto x1h = TensorAccessorArgs<x0l.next_compile_time_args_offset()>();const auto Xh1 = TensorAccessor(x1h, a_xh1);
    constexpr auto x1m = TensorAccessorArgs<x1h.next_compile_time_args_offset()>();const auto Xm1 = TensorAccessor(x1m, a_xm1);
    constexpr auto x1l = TensorAccessorArgs<x1m.next_compile_time_args_offset()>();const auto Xl1 = TensorAccessor(x1l, a_xl1);
    constexpr auto x2h = TensorAccessorArgs<x1l.next_compile_time_args_offset()>();const auto Xh2 = TensorAccessor(x2h, a_xh2);
    constexpr auto x2m = TensorAccessorArgs<x2h.next_compile_time_args_offset()>();const auto Xm2 = TensorAccessor(x2m, a_xm2);
    constexpr auto x2l = TensorAccessorArgs<x2m.next_compile_time_args_offset()>();const auto Xl2 = TensorAccessor(x2l, a_xl2);
    constexpr auto gnb = TensorAccessorArgs<x2l.next_compile_time_args_offset()>();const auto Nbr = TensorAccessor(gnb, a_nbr);

    // ---- stage the x-window (9 planes) + nbr node-slice resident in L1 (once for this core) ----
    const uint32_t XW = xwin_ntiles;
    cb_reserve_back(cb_xh0, XW); cb_reserve_back(cb_xm0, XW); cb_reserve_back(cb_xl0, XW);
    cb_reserve_back(cb_xh1, XW); cb_reserve_back(cb_xm1, XW); cb_reserve_back(cb_xl1, XW);
    cb_reserve_back(cb_xh2, XW); cb_reserve_back(cb_xm2, XW); cb_reserve_back(cb_xl2, XW);
    const uint32_t ph0 = get_write_ptr(cb_xh0), pm0 = get_write_ptr(cb_xm0), pl0 = get_write_ptr(cb_xl0);
    const uint32_t ph1 = get_write_ptr(cb_xh1), pm1 = get_write_ptr(cb_xm1), pl1 = get_write_ptr(cb_xl1);
    const uint32_t ph2 = get_write_ptr(cb_xh2), pm2 = get_write_ptr(cb_xm2), pl2 = get_write_ptr(cb_xl2);
    for (uint32_t j = 0; j < XW; ++j) {
        const uint32_t p = xwin_tile_lo + j;
        noc_async_read_page(p, Xh0, ph0 + j*TB); noc_async_read_page(p, Xm0, pm0 + j*TB); noc_async_read_page(p, Xl0, pl0 + j*TB);
        noc_async_read_page(p, Xh1, ph1 + j*TB); noc_async_read_page(p, Xm1, pm1 + j*TB); noc_async_read_page(p, Xl1, pl1 + j*TB);
        noc_async_read_page(p, Xh2, ph2 + j*TB); noc_async_read_page(p, Xm2, pm2 + j*TB); noc_async_read_page(p, Xl2, pl2 + j*TB);
    }
    constexpr uint32_t NBTB = 4096;                              // int32 page (1024 int32)
    const uint32_t slice_int_lo = node_lo * 27;
    const uint32_t page_lo = slice_int_lo / 1024;
    const uint32_t npages = (slice_int_lo + n_nodes * 27 + 1023) / 1024 - page_lo;
    cb_reserve_back(cb_nbr, npages);
    const uint32_t nbr0 = get_write_ptr(cb_nbr);
    for (uint32_t j = 0; j < npages; ++j) noc_async_read_page(page_lo + j, Nbr, nbr0 + j*NBTB);
    noc_async_read_barrier();
    cb_push_back(cb_xh0, XW); cb_push_back(cb_xm0, XW); cb_push_back(cb_xl0, XW);
    cb_push_back(cb_xh1, XW); cb_push_back(cb_xm1, XW); cb_push_back(cb_xl1, XW);
    cb_push_back(cb_xh2, XW); cb_push_back(cb_xm2, XW); cb_push_back(cb_xl2, XW);
    cb_push_back(cb_nbr, npages);

    tt_l1_ptr uint16_t* XH[3] = {(tt_l1_ptr uint16_t*)ph0,(tt_l1_ptr uint16_t*)ph1,(tt_l1_ptr uint16_t*)ph2};
    tt_l1_ptr uint16_t* XM[3] = {(tt_l1_ptr uint16_t*)pm0,(tt_l1_ptr uint16_t*)pm1,(tt_l1_ptr uint16_t*)pm2};
    tt_l1_ptr uint16_t* XL[3] = {(tt_l1_ptr uint16_t*)pl0,(tt_l1_ptr uint16_t*)pl1,(tt_l1_ptr uint16_t*)pl2};
    tt_l1_ptr int32_t*  NB    = (tt_l1_ptr int32_t*)nbr0;
    const uint32_t nbr_base   = slice_int_lo - page_lo * 1024;
    const uint32_t bc0 = get_write_ptr(cb_bc);                   // gather cache base (persistent scratch, 81*3 tiles)

    for (uint32_t T = 0; T < n_out; ++T) {
        const uint32_t gtile = start_tile + T;                  // global node-tile
        const uint32_t node0 = gtile * 1024;                    // FIRST NODE of the tile -- no div, node = node0 + i
        // (1) gather all 81 (o,c) once for this node-block into the cache (reused across r=0,1,2).
        for (uint32_t oc = 0; oc < K3; ++oc) {
            const uint32_t oo = oc / 3, c = oc % 3;
            tt_l1_ptr uint16_t* BH = (tt_l1_ptr uint16_t*)(bc0 + (oc*3+0)*TB);
            tt_l1_ptr uint16_t* BM = (tt_l1_ptr uint16_t*)(bc0 + (oc*3+1)*TB);
            tt_l1_ptr uint16_t* BL = (tt_l1_ptr uint16_t*)(bc0 + (oc*3+2)*TB);
            for (uint32_t i = 0; i < 1024; ++i) {
                const uint32_t node = node0 + i;
                const int32_t nn = (node >= node_lo && (node - node_lo) < n_nodes)
                                 ? NB[nbr_base + (node - node_lo) * 27 + oo] : -1;
                if (nn < 0) { BH[i] = 0; BM[i] = 0; BL[i] = 0; }
                else { const uint32_t s = (uint32_t)nn - xwin_elem_lo;   // de-interleaved: index by NODE, plane c chosen by buffer
                       BH[i] = XH[c][s]; BM[i] = XM[c][s]; BL[i] = XL[c][s]; }
            }
        }
        // (2) emit 3 output tiles (r=0,1,2), each 81 (a,b) pairs: a=A[(o*3+r)*3+c, gtile], b=cached g_oc.
        for (uint32_t r = 0; r < 3; ++r) {
            for (uint32_t oc = 0; oc < K3; ++oc) {
                const uint32_t oc_dummy = oc; (void)oc_dummy;
                const uint32_t otile = gtile * 3 + r;                       // shardable output-tile (spmv_mac-style)
                const uint32_t apage = otile * 81 + oc;                    // a_deint[output_tile][81]: 81 contiguous pages/tile
                (void)nnode_tiles;                                          // a is now output-tile-major (host repacks)
                cb_reserve_back(cb_a, 3); cb_reserve_back(cb_b, 3);
                const uint32_t pa = get_write_ptr(cb_a), pb = get_write_ptr(cb_b);
                noc_async_read_page(apage, Aah, pa + 0*TB); noc_async_read_page(apage, Aam, pa + 1*TB); noc_async_read_page(apage, Aal, pa + 2*TB);
                // copy cached gather (h/m/l) for this (o,c) into cb_b
                tt_l1_ptr uint32_t* SB = (tt_l1_ptr uint32_t*)(bc0 + (oc*3+0)*TB);
                tt_l1_ptr uint32_t* DB = (tt_l1_ptr uint32_t*)pb;
                for (uint32_t w = 0; w < (3*TB)/4; ++w) DB[w] = SB[w];     // 3 tiles (h,m,l are contiguous in the cache)
                noc_async_read_barrier();
                cb_push_back(cb_a, 3); cb_push_back(cb_b, 3);
            }
        }
    }
}
