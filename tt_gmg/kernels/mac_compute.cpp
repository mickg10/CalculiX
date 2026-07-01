// mac_compute.cpp — bf16x3 COMPENSATED MAC with TRUE fp32 accumulate.
// Public mul_tiles hardcodes clear_fp32_dst_acc=true (wipes the fp32 accumulator every call), so eltwise
// accumulation is bf16 -> fails under cancellation. We call the LLK directly with clear_fp32_dst_acc=false
// for every term after the first, so the 6*K cross-term products accumulate in the fp32 dst register.
// cb_a=[ah,am,al], cb_b=[bh,bm,bl] per k; 6 cross-terms (level-sum<=2). c_16 = fp32 output.
#include "api/compute/eltwise_binary.h"
#include <cstdint>

// one MAC term: dst[0] (+)= cb_a[ai] (elementwise*) cb_b[bi].  first=true seeds (clears fp32 acc), else accumulates.
ALWI void mac_term(uint32_t cb_a, uint32_t cb_b, uint32_t ai, uint32_t bi, bool first) {
    UNPACK((llk_unpack_AB(cb_a, cb_b, ai, bi)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_ACCUM_MODE, MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(cb_a, cb_b, 0 /*idst*/, first /*clear_fp32_dst_acc*/)));
}

void kernel_main() {
    const uint32_t n_out = get_arg_val<uint32_t>(0);
    const uint32_t K     = get_arg_val<uint32_t>(1);
    constexpr auto cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_a, cb_b, cb_out);
    mul_tiles_init(cb_a, cb_b, 1 /*acc_to_dest*/, 0 /*call_line, disambiguates overload*/);

    for (uint32_t t = 0; t < n_out; ++t) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            cb_wait_front(cb_a, 3); cb_wait_front(cb_b, 3);
            bool first = (k == 0);
            mac_term(cb_a, cb_b, 0, 0, first);   // ah*bh (first term of the tile seeds/clears fp32 acc)
            mac_term(cb_a, cb_b, 0, 1, false);   // ah*bm
            mac_term(cb_a, cb_b, 1, 0, false);   // am*bh
            mac_term(cb_a, cb_b, 0, 2, false);   // ah*bl
            mac_term(cb_a, cb_b, 1, 1, false);   // am*bm
            mac_term(cb_a, cb_b, 2, 0, false);   // al*bh
            cb_pop_front(cb_a, 3); cb_pop_front(cb_b, 3);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1); tile_regs_wait(); pack_tile(0, cb_out); cb_push_back(cb_out, 1); tile_regs_release();
    }
}
