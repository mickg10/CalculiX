// mac_compute_matmul.cpp — fp32-ACCUMULATE DIA MAC via matmul-diagonal (Stage B: matmul_tiles into fp32 dest acc).
// Replaces the eltwise-LLK MAC (bf16-accumulate -> diverges on the smoother's extreme-cancellation vectors,
// |Ax|<<|x|). matmul_tiles is the ONLY Metalium primitive that accumulates DST+=C in fp32 (strategy line 111).
// For 32 output elements e and K terms k, y[e] = sum_k a_k[e] b_k[e] = diag(A @ B) where A[e,k]=a_k[e],
// B[k,e]=b_k[e]. The K-reduction (matmul inner dim) accumulates in fp32 -> exact cancellation. Feeding the bf16x3
// SPLIT levels (8-bit, survive matmul's 11-bit input rounding) as A/B gives products >4.69e-4-class -> within the
// eig-deflation tolerance -> converges. Diagonal is extracted by masking with an identity tile then row-reduce
// (that reduce has ONE nonzero/row -> no cancellation -> bf16 reduce is safe).
// CBs: cb_A=c_0 (KT tiles [32 elem,32 k]), cb_B=c_1 (KT tiles [32 k,32 elem]), cb_id=c_2 (identity mask),
//      cb_c=c_24 (fp32 matmul scratch), cb_out=c_16 (fp32 y). Args: n_out (32-elem groups), KT (k-tiles).
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include <cstdint>

namespace NAMESPACE {
void MAIN {
    const uint32_t n_out = get_arg_val<uint32_t>(0);   // number of 32-element output groups
    const uint32_t KT    = get_arg_val<uint32_t>(1);   // number of k-tiles (ceil(6K/32))
    constexpr auto cb_A = tt::CBIndex::c_0, cb_B = tt::CBIndex::c_1, cb_id = tt::CBIndex::c_2;
    constexpr auto cb_c = tt::CBIndex::c_24, cb_out = tt::CBIndex::c_16;

    mm_init(cb_A, cb_B, cb_c);
    cb_wait_front(cb_id, 1);                            // identity mask resident (staged once by reader)

    for (uint32_t g = 0; g < n_out; ++g) {
        // ---- C = A @ B, fp32-accumulated over the KT k-tiles ----
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < KT; ++kt) {
            cb_wait_front(cb_A, 1); cb_wait_front(cb_B, 1);
            matmul_tiles(cb_A, cb_B, 0, 0, 0, false);  // dst[0] += A_kt @ B_kt  (fp32 dest accumulate)
            cb_pop_front(cb_A, 1); cb_pop_front(cb_B, 1);
        }
        tile_regs_commit();
        cb_reserve_back(cb_c, 1); tile_regs_wait(); pack_tile(0, cb_c); cb_push_back(cb_c, 1); tile_regs_release();

        // ---- diag(C): mask with identity then row-reduce (one nonzero per row -> safe bf16 reduce) ----
        cb_wait_front(cb_c, 1);
        tile_regs_acquire();
        mul_tiles_init(cb_c, cb_id);
        mul_tiles(cb_c, cb_id, 0, 0, 0);               // dst[0] = C .* I  (only diagonal survives)
        tile_regs_commit();
        cb_reserve_back(cb_c, 1); tile_regs_wait(); pack_tile(0, cb_c); cb_push_back(cb_c, 1); tile_regs_release();
        cb_pop_front(cb_c, 1);

        cb_wait_front(cb_c, 1);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_c, cb_id, cb_out);
        tile_regs_acquire();
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_c, cb_id, 0, 0, 0);
        reduce_uninit();
        tile_regs_commit();
        cb_reserve_back(cb_out, 1); tile_regs_wait(); pack_tile(0, cb_out); cb_push_back(cb_out, 1); tile_regs_release();
        cb_pop_front(cb_c, 1);
    }
}
}
