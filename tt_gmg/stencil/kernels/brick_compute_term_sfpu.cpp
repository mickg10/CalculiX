// Reconstruct each bf16x3 term in the FPU, then sum terms in full-fp32 SFPU
// destination registers. This avoids Wormhole ELWMUL's cross-term
// negative-to-positive cancellation defect and avoids Float32 CB unpack
// truncation in a second reduction stage.
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include <cstdint>

#ifdef TRISC_MATH
inline void accumulate_and_clear_batch_start0_face(
    const uint32_t,
    const uint32_t,
    const uint32_t) {
    constexpr uint32_t vectors_per_tile = 32;
    const uint32_t accumulator_0_base = 0 * vectors_per_tile;
    const uint32_t accumulator_1_base = 1 * vectors_per_tile;
    const uint32_t accumulator_2_base = 2 * vectors_per_tile;
    for (uint32_t vector = 0; vector < 8; ++vector) {
        sfpi::vFloat accumulator_0 =
            sfpi::dst_reg[accumulator_0_base + vector];
        sfpi::vFloat accumulator_1 =
            sfpi::dst_reg[accumulator_1_base + vector];
        sfpi::vFloat accumulator_2 =
            sfpi::dst_reg[accumulator_2_base + vector];
        accumulator_0 = accumulator_0 + sfpi::dst_reg[3 * vectors_per_tile + vector];
        accumulator_1 = accumulator_1 + sfpi::dst_reg[4 * vectors_per_tile + vector];
        accumulator_2 = accumulator_2 + sfpi::dst_reg[5 * vectors_per_tile + vector];
        accumulator_0 = accumulator_0 + sfpi::dst_reg[6 * vectors_per_tile + vector];
        accumulator_1 = accumulator_1 + sfpi::dst_reg[7 * vectors_per_tile + vector];
        sfpi::dst_reg[accumulator_0_base + vector] = accumulator_0;
        sfpi::dst_reg[accumulator_1_base + vector] = accumulator_1;
        sfpi::dst_reg[accumulator_2_base + vector] = accumulator_2;
        sfpi::dst_reg[3 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[4 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[5 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[6 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[7 * vectors_per_tile + vector] = 0.0f;
    }
}

inline void accumulate_and_clear_batch_start1_face(
    const uint32_t,
    const uint32_t,
    const uint32_t) {
    constexpr uint32_t vectors_per_tile = 32;
    for (uint32_t vector = 0; vector < 8; ++vector) {
        sfpi::vFloat accumulator_0 = sfpi::dst_reg[0 * vectors_per_tile + vector];
        sfpi::vFloat accumulator_1 = sfpi::dst_reg[1 * vectors_per_tile + vector];
        sfpi::vFloat accumulator_2 = sfpi::dst_reg[2 * vectors_per_tile + vector];
        accumulator_1 = accumulator_1 + sfpi::dst_reg[3 * vectors_per_tile + vector];
        accumulator_2 = accumulator_2 + sfpi::dst_reg[4 * vectors_per_tile + vector];
        accumulator_0 = accumulator_0 + sfpi::dst_reg[5 * vectors_per_tile + vector];
        accumulator_1 = accumulator_1 + sfpi::dst_reg[6 * vectors_per_tile + vector];
        accumulator_2 = accumulator_2 + sfpi::dst_reg[7 * vectors_per_tile + vector];
        sfpi::dst_reg[0 * vectors_per_tile + vector] = accumulator_0;
        sfpi::dst_reg[1 * vectors_per_tile + vector] = accumulator_1;
        sfpi::dst_reg[2 * vectors_per_tile + vector] = accumulator_2;
        sfpi::dst_reg[3 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[4 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[5 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[6 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[7 * vectors_per_tile + vector] = 0.0f;
    }
}

inline void accumulate_and_clear_batch_start2_face(
    const uint32_t,
    const uint32_t,
    const uint32_t) {
    constexpr uint32_t vectors_per_tile = 32;
    for (uint32_t vector = 0; vector < 8; ++vector) {
        sfpi::vFloat accumulator_0 = sfpi::dst_reg[0 * vectors_per_tile + vector];
        sfpi::vFloat accumulator_1 = sfpi::dst_reg[1 * vectors_per_tile + vector];
        sfpi::vFloat accumulator_2 = sfpi::dst_reg[2 * vectors_per_tile + vector];
        accumulator_2 = accumulator_2 + sfpi::dst_reg[3 * vectors_per_tile + vector];
        accumulator_0 = accumulator_0 + sfpi::dst_reg[4 * vectors_per_tile + vector];
        accumulator_1 = accumulator_1 + sfpi::dst_reg[5 * vectors_per_tile + vector];
        accumulator_2 = accumulator_2 + sfpi::dst_reg[6 * vectors_per_tile + vector];
        accumulator_0 = accumulator_0 + sfpi::dst_reg[7 * vectors_per_tile + vector];
        sfpi::dst_reg[0 * vectors_per_tile + vector] = accumulator_0;
        sfpi::dst_reg[1 * vectors_per_tile + vector] = accumulator_1;
        sfpi::dst_reg[2 * vectors_per_tile + vector] = accumulator_2;
        sfpi::dst_reg[3 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[4 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[5 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[6 * vectors_per_tile + vector] = 0.0f;
        sfpi::dst_reg[7 * vectors_per_tile + vector] = 0.0f;
    }
}
#endif

ALWI void accumulate_and_clear_batch(uint32_t start_component) {
    if (start_component == 0) {
        MATH((_llk_math_eltwise_binary_sfpu_params_(
            accumulate_and_clear_batch_start0_face, 0, 0, 0)));
    } else if (start_component == 1) {
        MATH((_llk_math_eltwise_binary_sfpu_params_(
            accumulate_and_clear_batch_start1_face, 0, 0, 0)));
    } else {
        MATH((_llk_math_eltwise_binary_sfpu_params_(
            accumulate_and_clear_batch_start2_face, 0, 0, 0)));
    }
}

ALWI void reconstruct_term(
    uint32_t cb_a,
    uint32_t cb_b,
    uint32_t abase,
    uint32_t idst) {
    UNPACK((llk_unpack_AB(cb_a, cb_b, abase + 0, 0)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, false)));

    UNPACK((llk_unpack_AB(cb_a, cb_b, abase + 0, 1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, abase + 1, 0)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, abase + 0, 2)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, abase + 1, 1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, false)));
    UNPACK((llk_unpack_AB(cb_a, cb_b, abase + 2, 0)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, false)));
}

void kernel_main() {
    const uint32_t n_groups = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t accumulate_terms = get_compile_time_arg_val(0);
    constexpr uint32_t scratch_batch_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t terms = 81;
    constexpr uint32_t output_components = 3;
    constexpr uint32_t scratch_first = 3;
    static_assert(scratch_batch_tiles >= 1 && scratch_batch_tiles <= 5);

    binary_op_init_common(cb_a, cb_b, cb_out);
    mul_tiles_init(cb_a, cb_b);

    for (uint32_t group = 0; group < n_groups; ++group) {
        tile_regs_acquire();
        // Once an SFPU operation has run on Wormhole, the FPU's per-call
        // destination-clear flag no longer clears the selected FP32 tile.
        // Initialize every destination explicitly and make the FPU path
        // accumulation-only.  Scratch is then reset explicitly after every
        // term instead of relying on the affected clear flag.
        fill_tile_init();
        for (uint32_t destination = 0;
             destination < scratch_first + scratch_batch_tiles;
             ++destination) {
            fill_tile(destination, 0.0f);
        }
        // The FPU multiply configuration survives SFPU calls.  Configure the
        // custom binary-SFPU address mode once per acquired destination set;
        // repeating both init sequences for every component/term dominated
        // the otherwise-correct kernel.
        add_binary_tile_init();
        uint32_t pending_scratch = 0;
        uint32_t pending_start_component = 0;
        for (uint32_t term = 0; term < terms; ++term) {
            cb_wait_front(cb_a, 9);
            cb_wait_front(cb_b, 3);
            if (term < accumulate_terms) {
                for (uint32_t output_component = 0;
                     output_component < output_components;
                     ++output_component) {
                    reconstruct_term(
                        cb_a,
                        cb_b,
                        output_component * 3,
                        scratch_first + pending_scratch);
                    ++pending_scratch;
                    if (pending_scratch == scratch_batch_tiles) {
                        accumulate_and_clear_batch(
                            pending_start_component);
                        pending_start_component =
                            (pending_start_component + pending_scratch) %
                            output_components;
                        pending_scratch = 0;
                    }
                }
            }
            cb_pop_front(cb_a, 9);
            cb_pop_front(cb_b, 3);
        }
        if (pending_scratch != 0) {
            accumulate_and_clear_batch(
                pending_start_component);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, output_components);
        tile_regs_wait();
        for (uint32_t output_component = 0;
             output_component < output_components;
             ++output_component) {
            pack_tile(
                output_component,
                cb_out,
                output_component);
        }
        cb_push_back(cb_out, output_components);
        tile_regs_release();
    }
}
