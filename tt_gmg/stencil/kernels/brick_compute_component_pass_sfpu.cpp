// Accumulate one output component at a time.  Three reconstructed terms live
// in the four-tile fp32 destination file (main + 3 scratch), then one SFPU pass
// folds and clears the scratch tiles without invoking the FPU sign-crossing bug.
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include <cstdint>

#ifdef TRISC_MATH
inline void accumulate_three_and_clear_face(
    const uint32_t,
    const uint32_t,
    const uint32_t) {
    constexpr uint32_t vectors_per_tile = 32;
    for (uint32_t vector = 0; vector < 8; ++vector) {
        sfpi::vFloat accumulator = sfpi::dst_reg[vector];
        accumulator = accumulator + sfpi::dst_reg[1 * vectors_per_tile + vector];
        accumulator = accumulator + sfpi::dst_reg[2 * vectors_per_tile + vector];
        accumulator = accumulator + sfpi::dst_reg[3 * vectors_per_tile + vector];
        sfpi::dst_reg[vector] = accumulator;
        sfpi::dst_reg[1 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[2 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[3 * vectors_per_tile + vector] = 0.0f;
    }
}
#endif

ALWI void accumulate_three_and_clear() {
    MATH((_llk_math_eltwise_binary_sfpu_params_(
        accumulate_three_and_clear_face, 0, 0, 0)));
}

ALWI void reconstruct_term(
    uint32_t cb_a,
    uint32_t cb_b,
    uint32_t idst) {
    UNPACK((llk_unpack_AB(cb_a, cb_b, 0, 0)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, 0, 1)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, 1, 0)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, 0, 2)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, 1, 1)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, 2, 0)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(cb_a, cb_b, idst, false)));
}

void kernel_main() {
    const uint32_t n_groups = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t accumulate_terms = get_compile_time_arg_val(0);
    constexpr uint32_t terms = 81;

    binary_op_init_common(cb_a, cb_b, cb_out);
    mul_tiles_init(cb_a, cb_b);

    for (uint32_t group = 0; group < n_groups; ++group) {
        for (uint32_t output_component = 0;
             output_component < 3;
             ++output_component) {
            tile_regs_acquire();
            fill_tile_init();
            for (uint32_t destination = 0; destination < 4; ++destination) {
                fill_tile(destination, 0.0f);
            }
            add_binary_tile_init();
            uint32_t pending = 0;
            for (uint32_t term = 0; term < terms; ++term) {
                cb_wait_front(cb_a, 3);
                cb_wait_front(cb_b, 3);
                if (term < accumulate_terms) {
                    reconstruct_term(cb_a, cb_b, 1 + pending);
                    ++pending;
                    if (pending == 3) {
                        accumulate_three_and_clear();
                        pending = 0;
                    }
                }
                cb_pop_front(cb_a, 3);
                cb_pop_front(cb_b, 3);
            }
            if (pending != 0) {
                accumulate_three_and_clear();
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out, 0);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    }
}
