// Six-product bf16x3 MAC, emitting three fp32 partial sums per output component.
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
        cb_a,
        cb_b,
        idst,
        first)));
}

void kernel_main() {
    const uint32_t n_groups = get_arg_val<uint32_t>(0);
    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t first_chunk_terms = get_compile_time_arg_val(0);
    constexpr uint32_t terms = 81;
    constexpr uint32_t chunks = 3;
    constexpr uint32_t terms_per_chunk = terms / chunks;

    binary_op_init_common(cb_a, cb_b, cb_out);
    mul_tiles_init(cb_a, cb_b);

    for (uint32_t group = 0; group < n_groups; ++group) {
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            tile_regs_acquire();
            for (uint32_t local_term = 0; local_term < terms_per_chunk; ++local_term) {
                cb_wait_front(cb_a, 9);
                cb_wait_front(cb_b, 3);
                if (chunk != 0 || local_term < first_chunk_terms) {
                    const bool first = local_term == 0;
                    for (uint32_t output_component = 0; output_component < 3; ++output_component) {
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
}
