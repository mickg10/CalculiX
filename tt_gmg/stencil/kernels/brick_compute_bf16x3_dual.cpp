// Full six-product bf16x3 MAC with alternating-term B streams supplied by
// the two data-movement processors.
#include "api/compute/eltwise_binary.h"
#include <cstdint>

ALWI void mac_term(
    uint32_t cb_a,
    uint32_t cb_b,
    uint32_t ai,
    uint32_t bi,
    uint32_t idst,
    bool first) {
    UNPACK((llk_unpack_AB(cb_a, cb_b, ai, bi)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(
        cb_a, cb_b, idst, first)));
}

void kernel_main() {
    const uint32_t n_groups = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_beven = tt::CBIndex::c_1;
    constexpr auto cb_bodd = tt::CBIndex::c_3;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t accumulate_terms = get_compile_time_arg_val(0);
    constexpr uint32_t terms = 81;

    binary_op_init_common(cb_a, cb_beven, cb_out);
    mul_tiles_init(cb_a, cb_beven);
    for (uint32_t group = 0; group < n_groups; ++group) {
        tile_regs_acquire();
        for (uint32_t term = 0; term < terms; ++term) {
            const uint32_t cb_b =
                (term & 1u) == 0 ? cb_beven : cb_bodd;
            cb_wait_front(cb_a, 9);
            cb_wait_front(cb_b, 3);
            if (term < accumulate_terms) {
                const bool first = term == 0;
                for (uint32_t output_component = 0;
                     output_component < 3;
                     ++output_component) {
                    const uint32_t abase = output_component * 3;
                    mac_term(cb_a, cb_b, abase + 0, 0, output_component, first);
                    mac_term(cb_a, cb_b, abase + 0, 1, output_component, false);
                    mac_term(cb_a, cb_b, abase + 1, 0, output_component, false);
                    mac_term(cb_a, cb_b, abase + 0, 2, output_component, false);
                    mac_term(cb_a, cb_b, abase + 1, 1, output_component, false);
                    mac_term(cb_a, cb_b, abase + 2, 0, output_component, false);
                }
            }
            cb_pop_front(cb_a, 9);
            cb_pop_front(cb_b, 3);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 3);
        tile_regs_wait();
        pack_tile(0, cb_out, 0);
        pack_tile(1, cb_out, 1);
        pack_tile(2, cb_out, 2);
        cb_push_back(cb_out, 3);
        tile_regs_release();
    }
}
