// reduce_compute.cpp — fp32-accumulate REDUCE_ROW over nt fp32 tiles.
// enforce_fp32_accumulation=true (requires DST_ACCUM_MODE / fp32_dest_acc_en) => full-fp32 reduction,
// which the near-singular GMG operator needs (ttnn.sum / matmul / packer_l1_acc all failed cancellation).
#include "api/compute/reduce.h"
#include <cstdint>
void kernel_main() {
    const uint32_t nt = get_arg_val<uint32_t>(0);
    constexpr auto cb_in = tt::CBIndex::c_0, cb_sc = tt::CBIndex::c_2, cb_out = tt::CBIndex::c_16;
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_in, cb_sc, cb_out);
    cb_wait_front(cb_sc, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < nt; ++i) {
        cb_wait_front(cb_in, 1);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_in, cb_sc, 0, 0, 0);
        cb_pop_front(cb_in, 1);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    reduce_uninit();
}
