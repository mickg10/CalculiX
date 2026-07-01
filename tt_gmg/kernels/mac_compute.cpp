// mac_compute.cpp — bf16x3 COMPENSATED MAC. cb_a holds [ah,am,al], cb_b holds [bh,bm,bl] per k (3 tiles each,
// each unique tile read ONCE by the reader -> no repeated same-page NoC reads). The 6 cross-terms with
// level-sum<=2 are selected by (a_idx,b_idx): (0,0)=ah*bh (0,1)=ah*bm (1,0)=am*bh (0,2)=ah*bl (1,1)=am*bm (2,0)=al*bh,
// accumulated in fp32 dest -> ~fp32. c_16 = fp32 output.
#include "api/compute/eltwise_binary.h"
#include <cstdint>

void kernel_main() {
    const uint32_t n_out = get_arg_val<uint32_t>(0);
    const uint32_t K     = get_arg_val<uint32_t>(1);
    constexpr auto cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_a, cb_b, cb_out);

    for (uint32_t t = 0; t < n_out; ++t) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            cb_wait_front(cb_a, 3); cb_wait_front(cb_b, 3);
            if (k == 0) {
                mul_tiles_init(cb_a, cb_b);        mul_tiles(cb_a, cb_b, 0, 0, 0);   // seed dst = ah*bh
                mul_tiles_init(cb_a, cb_b, 1, 0);                                     // accumulate mode
            } else {
                mul_tiles(cb_a, cb_b, 0, 0, 0);                                       // ah*bh
            }
            mul_tiles(cb_a, cb_b, 0, 1, 0);   // ah*bm
            mul_tiles(cb_a, cb_b, 1, 0, 0);   // am*bh
            mul_tiles(cb_a, cb_b, 0, 2, 0);   // ah*bl
            mul_tiles(cb_a, cb_b, 1, 1, 0);   // am*bm
            mul_tiles(cb_a, cb_b, 2, 0, 0);   // al*bh
            cb_pop_front(cb_a, 3); cb_pop_front(cb_b, 3);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1); tile_regs_wait(); pack_tile(0, cb_out); cb_push_back(cb_out, 1); tile_regs_release();
    }
}
