// Full-fp32 SFPU reduction of three FPU partial sums per output component.
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include <cstdint>

void kernel_main() {
    const uint32_t n_groups = get_arg_val<uint32_t>(0);
    constexpr auto cb_partial = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t reduce_mode = get_compile_time_arg_val(0);
    constexpr uint32_t chunks = 3;
    constexpr uint32_t output_components = 3;

    init_sfpu(cb_partial, cb_out);
    pack_reconfig_data_format(cb_out);
    for (uint32_t group = 0; group < n_groups; ++group) {
        cb_wait_front(cb_partial, chunks * output_components);
        for (uint32_t output_component = 0;
             output_component < output_components;
             ++output_component) {
            tile_regs_acquire();
            copy_tile_init(cb_partial);
            copy_tile(cb_partial, output_component + 0 * output_components, 0);
            copy_tile(cb_partial, output_component + 1 * output_components, 1);
            copy_tile(cb_partial, output_component + 2 * output_components, 2);
            if constexpr (reduce_mode == 0) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                add_binary_tile(0, 2, 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_partial, chunks * output_components);
    }
}
