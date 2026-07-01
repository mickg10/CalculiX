// mac_compute.cpp — bf16x3 COMPENSATED MAC via a SINGLE cb_a/cb_b pair (no operand-CB switching).
// The reader interleaves, per k, the 6 cross-term tile-pairs (ah*bh, ah*bm, am*bh, ah*bl, am*bm, al*bh)
// into cb_a (c_0) and cb_b (c_1). Compute accumulates all 6*K products in fp32 dest -> ~fp32 accuracy,
// required for the near-singular row236 operator (single-pass bf16 gives ~150% error). c_16 = output.
#include "api/compute/eltwise_binary.h"
#include <cstdint>

void kernel_main() {
    const uint32_t n_out = get_arg_val<uint32_t>(0);
    const uint32_t K     = get_arg_val<uint32_t>(1);
    constexpr auto cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
    constexpr uint32_t T = 6;   // cross-terms per k

    binary_op_init_common(cb_a, cb_b, cb_out);

    for (uint32_t t = 0; t < n_out; ++t) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            cb_wait_front(cb_a, T); cb_wait_front(cb_b, T);
            if (k == 0) {
                mul_tiles_init(cb_a, cb_b);          mul_tiles(cb_a, cb_b, 0, 0, 0);           // seed dst
                mul_tiles_init(cb_a, cb_b, 1, 0);    for (uint32_t j=1;j<T;++j) mul_tiles(cb_a, cb_b, j, j, 0);
            } else {
                for (uint32_t j = 0; j < T; ++j) mul_tiles(cb_a, cb_b, j, j, 0);               // accumulate
            }
            cb_pop_front(cb_a, T); cb_pop_front(cb_b, T);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1); tile_regs_wait(); pack_tile(0, cb_out); cb_push_back(cb_out, 1); tile_regs_release();
    }
}
