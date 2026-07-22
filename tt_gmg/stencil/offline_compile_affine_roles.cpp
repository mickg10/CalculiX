// Offline role compiler for default-off affine-fold, split-A, and lossless
// corner-constant brick candidates.
//
// This deliberately creates no TT device.  It reconstructs the Wormhole JIT
// build environment, selects the environment whose build key matches the
// proven QuietBox cache.  Affine mode compiles its reader plus all three
// compute roles.  Split-A mode additionally compiles the RISC-1 A-low/B-mid
// writer.  The compute descriptors are seeded from the measured selective-
// b-low kernel; split-A then enables its additional BF16 c_5 descriptor.

// This is a narrow 0.73-era validation utility, not a hardware benchmark.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "impl/kernels/kernel_source.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/genfiles.hpp"
#include "jit_build/jit_device_config.hpp"
#include "jit_build/jit_build_settings.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace fs = std::filesystem;

namespace tt::tt_metal {
namespace {

constexpr uint64_t kQuietBoxBuildKey = 16980933987073800404ULL;
constexpr std::string_view kAffineReaderName =
    "brick_reader_a_bhi_affine_fold_offline/";
constexpr std::string_view kAffineComputeName =
    "brick_compute_bf16x3_affine_corner_fold_offline/";
constexpr std::string_view kSplitReaderName =
    "brick_reader_ahi_amid_bhi_offline/";
constexpr std::string_view kSplitWriterName =
    "brick_reader_alow_bmid_writer_offline/";
constexpr std::string_view kSplitComputeName =
    "brick_compute_bf16x3_split_a_selective_blow_offline/";
constexpr std::string_view kCornerReaderName =
    "brick_reader_a_bhi_corner_constant_offline/";
constexpr std::string_view kCornerComputeName =
    "brick_compute_bf16x3_corner_constant_selective_blow_offline/";
constexpr std::string_view kCornerWriterName =
    "brick_reader_bmid_corner_constant_writer_offline/";
constexpr std::string_view kCornerReuseReaderName =
    "brick_reader_a_bhi_corner_reuse_offline/";
constexpr std::string_view kCornerReuseComputeName =
    "brick_compute_bf16x3_corner_reuse_selective_blow_offline/";
constexpr std::string_view kCornerReuseWriterName =
    "brick_reader_bmid_corner_reuse_writer_offline/";
constexpr std::string_view kPreexpandedReaderName =
    "brick_reader_ahi_amid_preexpanded_bhi_offline/";
constexpr std::string_view kPreexpandedWriterName =
    "brick_reader_alow_preexpanded_bmid_blow_writer_offline/";
constexpr std::string_view kPreexpandedComputeName =
    "brick_compute_bf16x3_preexpanded_b_offline/";
constexpr std::string_view kCanonicalBaseReaderName =
    "brick_reader_canonical_packed_base_bhi_offline/";
constexpr std::string_view kCanonicalBaseWriterName =
    "brick_reader_canonical_packed_base_bmid_blow_writer_offline/";
constexpr std::string_view kCanonicalBaseComputeName =
    "brick_compute_canonical_packed_base_offline/";
constexpr std::string_view kCanonicalFallbackReaderName =
    "brick_reader_canonical_packed_fallback_bhi_offline/";
constexpr std::string_view kCanonicalFallbackWriterName =
    "brick_reader_canonical_packed_fallback_bmid_blow_writer_offline/";
constexpr std::string_view kCanonicalFallbackComputeName =
    "brick_compute_canonical_packed_fallback_offline/";
constexpr std::string_view kCanonicalDynamicBaseReaderName =
    "brick_reader_canonical_dynamic_base_bhi_offline/";
constexpr std::string_view kCanonicalDynamicBaseWriterName =
    "brick_reader_canonical_dynamic_base_bmid_blow_writer_offline/";
constexpr std::string_view kCanonicalDynamicBaseComputeName =
    "brick_compute_canonical_dynamic_base_offline/";
constexpr std::string_view kCanonicalDynamicFallbackReaderName =
    "brick_reader_canonical_dynamic_fallback_bhi_offline/";
constexpr std::string_view kCanonicalDynamicFallbackWriterName =
    "brick_reader_canonical_dynamic_fallback_bmid_blow_writer_offline/";
constexpr std::string_view kCanonicalDynamicFallbackComputeName =
    "brick_compute_canonical_dynamic_fallback_offline/";
constexpr std::string_view kComputeSeed =
    "brick_compute_bf16x3_selective_blow/307844999196475530";

class OfflineSettings final : public JitBuildSettings {
public:
    OfflineSettings(
        std::string name,
        std::string opt,
        std::vector<uint32_t> args,
        std::map<std::string, std::string> defines) :
        name_(std::move(name)),
        opt_(std::move(opt)),
        args_(std::move(args)),
        defines_(std::move(defines)) {}

    const std::string& get_full_kernel_name() const override { return name_; }
    std::string_view get_compiler_opt_level() const override { return opt_; }
    std::string_view get_linker_opt_level() const override { return opt_; }
    void process_defines(
        std::function<void(const std::string&, const std::string&)> callback)
        const override {
        for (const auto& [name, value] : defines_) {
            callback(name, value);
        }
    }
    void process_compile_time_args(
        std::function<void(const std::vector<uint32_t>&)> callback) const override {
        callback(args_);
    }
    void process_named_compile_time_args(
        std::function<void(const std::unordered_map<std::string, uint32_t>&)>)
        const override {}

private:
    std::string name_;
    std::string opt_;
    std::vector<uint32_t> args_;
    std::map<std::string, std::string> defines_;
};

uint32_t dispatch_message_addr(const Hal& hal, DispatchCoreType dispatch_core_type) {
    const uint32_t stream_index =
        dispatch_core_type == DispatchCoreType::WORKER ? 48u : 16u;
    return hal.get_noc_overlay_start_addr() +
           hal.get_noc_stream_reg_space_size() * stream_index +
           hal.get_noc_stream_remote_dest_buf_space_available_update_reg_index() *
               sizeof(uint32_t);
}

void copy_compute_descriptors(
    const JitBuildEnv& env,
    const OfflineSettings& settings) {
    const fs::path seed = fs::path(env.get_out_kernel_root_path()) / kComputeSeed;
    const fs::path destination =
        fs::path(env.get_out_kernel_root_path()) / settings.get_full_kernel_name();
    if (!fs::is_directory(seed)) {
        throw std::runtime_error("compute descriptor seed is absent: " + seed.string());
    }
    fs::create_directories(destination);
    for (const auto& entry : fs::directory_iterator(seed)) {
        if (entry.is_regular_file()) {
            fs::copy_file(
                entry.path(),
                destination / entry.path().filename(),
                fs::copy_options::overwrite_existing);
        }
    }
}

void enable_c5_descriptor(
    const JitBuildEnv& env,
    const OfflineSettings& settings) {
    const fs::path descriptor =
        fs::path(env.get_out_kernel_root_path()) /
        settings.get_full_kernel_name() /
        "chlkc_descriptors.h";
    std::ifstream input(descriptor);
    if (!input) {
        throw std::runtime_error(
            "c_5 compute descriptor is absent: " + descriptor.string());
    }
    std::string contents(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    const auto replace_all = [&contents](
                                 const std::string& before,
                                 const std::string& after) {
        size_t cursor = 0;
        uint32_t replacements = 0;
        while ((cursor = contents.find(before, cursor)) != std::string::npos) {
            contents.replace(cursor, before.size(), after);
            cursor += after.size();
            ++replacements;
        }
        return replacements;
    };
    const uint32_t format_replacements = replace_all(
        "5,5,5,5,5,255",
        "5,5,5,5,5,5");
    const uint32_t size_replacements = replace_all(
        "2048,2048,2048,2048,2048,1088",
        "2048,2048,2048,2048,2048,2048");
    if (format_replacements == 0 || size_replacements == 0) {
        throw std::runtime_error(
            "c_5 descriptor seed did not contain the expected sentinel");
    }
    std::ofstream output(descriptor, std::ios::trunc);
    output << contents;
    if (!output) {
        throw std::runtime_error(
            "failed to update c_5 compute descriptor: " + descriptor.string());
    }
}

void compile_for_config(
    const JitDeviceConfig& config,
    const llrt::RunTimeOptions& rtoptions,
    const fs::path& reader_source,
    const fs::path& compute_source,
    const fs::path& writer_source,
    bool corner_constant,
    bool corner_reuse,
    bool preexpanded_b,
    bool canonical_packed_base,
    bool canonical_packed_fallback,
    bool canonical_dynamic_base,
    bool canonical_dynamic_fallback,
    bool& compiled) {
    BuildEnvManager manager(*config.hal);
    manager.add_build_env(0, config, rtoptions);
    const auto& device_env = manager.get_device_build_env(0);
    if (device_env.build_key() != kQuietBoxBuildKey) {
        return;
    }
    if (compiled) {
        // The three descriptor files contain repeated equivalent QuietBox
        // build configurations.  A build key identifies the effective JIT
        // environment, so compiling the first match is sufficient.
        return;
    }

    const uint32_t tensix =
        config.hal->get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const uint32_t dm_class =
        static_cast<uint32_t>(HalProcessorClassType::DM);
    const uint32_t compute_class =
        static_cast<uint32_t>(HalProcessorClassType::COMPUTE);
    const bool has_writer = !writer_source.empty();
    const bool split_a =
        has_writer && !corner_constant && !corner_reuse && !preexpanded_b &&
        !canonical_packed_base && !canonical_packed_fallback &&
        !canonical_dynamic_base && !canonical_dynamic_fallback;

    // CB indices followed by the reader's non-sharded DRAM accessor
    // records: {IsDram, aligned_page_size}.  Affine mode consumes all three
    // A splits; split-A RISC-0 consumes A-hi and A-mid only.
    OfflineSettings reader_settings(
        std::string(
            canonical_dynamic_base
                ? kCanonicalDynamicBaseReaderName
                : canonical_dynamic_fallback
                ? kCanonicalDynamicFallbackReaderName
                : canonical_packed_base
                ? kCanonicalBaseReaderName
                : canonical_packed_fallback
                ? kCanonicalFallbackReaderName
                : corner_constant
                ? kCornerReaderName
                : corner_reuse
                ? kCornerReuseReaderName
                : preexpanded_b
                ? kPreexpandedReaderName
                : split_a ? kSplitReaderName : kAffineReaderName),
        "O2",
        canonical_dynamic_base
            ? std::vector<uint32_t>{
                  0u, 1u, 2u, 8u, 9u,
                  2u, 2048u, 2u, 2048u, 2u, 2048u,
                  2u, 2048u, 2u, 64u, 2u, 2048u}
            : canonical_dynamic_fallback
            ? std::vector<uint32_t>{
                  0u, 1u, 2u, 8u, 9u,
                  2u, 2048u, 2u, 2048u, 2u, 2048u,
                  2u, 64u, 2u, 2048u}
            : canonical_packed_base
            ? std::vector<uint32_t>{
                  0u, 1u,
                  2u, 2048u, 2u, 2048u, 2u, 2048u, 2u, 2048u}
            : canonical_packed_fallback
            ? std::vector<uint32_t>{
                  0u, 1u,
                  2u, 2048u, 2u, 2048u, 2u, 2048u}
            : corner_constant
            ? std::vector<uint32_t>{
                  0u, 1u, 5u, 2u,
                  2u, 2048u, 2u, 2048u, 2u, 2048u, 2u, 2048u}
            : split_a
            ? std::vector<uint32_t>{
                  0u, 1u, 2u, 2u, 2048u, 2u, 2048u}
            : preexpanded_b
            ? std::vector<uint32_t>{
                  0u, 1u,
                  2u, 2048u, 2u, 2048u, 2u, 2048u}
            : std::vector<uint32_t>{
                  0u, 1u, 2u, 2u, 2048u, 2u, 2048u, 2u, 2048u},
        {{"NOC_INDEX", "0"}, {"NOC_MODE", "0"}});
    fs::create_directories(
        fs::path(device_env.build_env.get_out_kernel_root_path()) /
        reader_settings.get_full_kernel_name());
    const KernelSource reader_kernel(
        fs::absolute(reader_source).string(), KernelSource::FILE_PATH);
    jit_build_genfiles_kernel_include(
        device_env.build_env, reader_settings, reader_kernel);
    jit_build(
        manager.get_kernel_build_state(0, tensix, dm_class, 0),
        &reader_settings);

    if (has_writer) {
        // Split-A: c_5/c_3/c_2/c_16 followed by A-low, halo, and output.
        // Corner-constant: c_3/c_2/c_16 followed by halo and output.
        OfflineSettings writer_settings(
            std::string(
                canonical_dynamic_base
                    ? kCanonicalDynamicBaseWriterName
                    : canonical_dynamic_fallback
                    ? kCanonicalDynamicFallbackWriterName
                    : canonical_packed_base
                    ? kCanonicalBaseWriterName
                    : canonical_packed_fallback
                    ? kCanonicalFallbackWriterName
                    : corner_constant
                    ? kCornerWriterName
                    : corner_reuse
                    ? kCornerReuseWriterName
                    : preexpanded_b
                    ? kPreexpandedWriterName
                    : kSplitWriterName),
            "O2",
            canonical_dynamic_base
                ? std::vector<uint32_t>{
                      3u, 4u, 16u, 6u, 7u, 10u, 11u,
                      2u, 2048u, 2u, 2048u, 2u, 64u,
                      2u, 2048u, 2u, 4096u}
                : canonical_dynamic_fallback
                ? std::vector<uint32_t>{
                      5u, 3u, 4u, 16u, 6u, 7u, 10u, 11u,
                      2u, 2048u, 2u, 2048u, 2u, 2048u,
                      2u, 64u, 2u, 2048u, 2u, 4096u}
                : canonical_packed_base
                ? std::vector<uint32_t>{
                      3u, 4u, 16u,
                      2u, 2048u,
                      2u, 2048u,
                      2u, 4096u}
                : (canonical_packed_fallback || preexpanded_b)
                ? std::vector<uint32_t>{
                      5u, 3u, 4u, 16u,
                      2u, 2048u,
                      2u, 2048u,
                      2u, 2048u,
                      2u, 4096u}
            : (corner_constant || corner_reuse)
                ? std::vector<uint32_t>{
                      3u, 2u, 16u,
                      2u, 2048u,
                      2u, 4096u}
                : std::vector<uint32_t>{
                      5u, 3u, 2u, 16u,
                      2u, 2048u,
                      2u, 2048u,
                      2u, 4096u},
            {{"NOC_INDEX", "1"}, {"NOC_MODE", "0"}});
        fs::create_directories(
            fs::path(device_env.build_env.get_out_kernel_root_path()) /
            writer_settings.get_full_kernel_name());
        const KernelSource writer_kernel(
            fs::absolute(writer_source).string(), KernelSource::FILE_PATH);
        jit_build_genfiles_kernel_include(
            device_env.build_env, writer_settings, writer_kernel);
        jit_build(
            manager.get_kernel_build_state(0, tensix, dm_class, 1),
            &writer_settings);
    }

    // The benchmark compiles this kernel with term_count=81 and the legacy
    // second compile argument fixed at one.  The inherited descriptors encode
    // HiFi4, FP32 destination accumulation, and the same five BF16 CBs plus
    // Float32 output used by the affine candidate.  Split-A also enables the
    // sixth BF16 input CB in the copied descriptor.
    OfflineSettings compute_settings(
        std::string(
            canonical_dynamic_base
                ? kCanonicalDynamicBaseComputeName
                : canonical_dynamic_fallback
                ? kCanonicalDynamicFallbackComputeName
                : canonical_packed_base
                ? kCanonicalBaseComputeName
                : canonical_packed_fallback
                ? kCanonicalFallbackComputeName
                : corner_constant
                ? kCornerComputeName
                : corner_reuse
                ? kCornerReuseComputeName
                : preexpanded_b
                ? kPreexpandedComputeName
                : split_a ? kSplitComputeName : kAffineComputeName),
        "O3",
        {81u, 1u},
        {{"NOC_MODE", "0"}});
    copy_compute_descriptors(device_env.build_env, compute_settings);
    if (split_a || corner_constant || preexpanded_b ||
        canonical_packed_fallback || canonical_dynamic_fallback) {
        enable_c5_descriptor(device_env.build_env, compute_settings);
    }
    const KernelSource compute_kernel(
        fs::absolute(compute_source).string(), KernelSource::FILE_PATH);
    jit_build_genfiles_triscs_src(
        device_env.build_env, compute_settings, compute_kernel);
    const auto compute_states =
        manager.get_kernel_build_states(0, tensix, compute_class);
    jit_build_subset(compute_states, &compute_settings);

    std::cout << "OFFLINE_ROLE_COMPILE pass=1 build_key="
              << device_env.build_key()
              << " mode="
              << (canonical_dynamic_base
                      ? "canonical_dynamic_base"
                      : canonical_dynamic_fallback
                      ? "canonical_dynamic_fallback"
                      : canonical_packed_base
                      ? "canonical_packed_base"
                      : canonical_packed_fallback
                      ? "canonical_packed_fallback"
                      : corner_constant
                      ? "corner_constant"
                      : corner_reuse
                      ? "corner_reuse"
                      : preexpanded_b
                      ? "preexpanded_b"
                      : split_a ? "split_a" : "affine")
              << " reader=" << reader_settings.get_full_kernel_name()
              << " reader_source=" << fs::absolute(reader_source).string()
              << " writer="
              << (canonical_dynamic_base && has_writer
                      ? kCanonicalDynamicBaseWriterName
                      : canonical_dynamic_fallback && has_writer
                      ? kCanonicalDynamicFallbackWriterName
                      : canonical_packed_base && has_writer
                      ? kCanonicalBaseWriterName
                      : canonical_packed_fallback && has_writer
                      ? kCanonicalFallbackWriterName
                      : corner_constant && has_writer
                      ? kCornerWriterName
                      : corner_reuse && has_writer
                      ? kCornerReuseWriterName
                      : preexpanded_b && has_writer
                      ? kPreexpandedWriterName
                      : split_a ? kSplitWriterName : "none")
              << " compute=" << compute_settings.get_full_kernel_name()
              << " compute_source=" << fs::absolute(compute_source).string()
              << '\n';
    compiled = true;
}

void enumerate_matching_config(
    const llrt::RunTimeOptions& rtoptions,
    const fs::path& reader_source,
    const fs::path& compute_source,
    const fs::path& writer_source,
    bool corner_constant,
    bool corner_reuse,
    bool preexpanded_b,
    bool canonical_packed_base,
    bool canonical_packed_fallback,
    bool canonical_dynamic_base,
    bool canonical_dynamic_fallback,
    bool& compiled) {
    constexpr uint32_t profiler_dram_bytes = 0;
    constexpr bool enable_dram_backed_cq = false;
    constexpr bool enable_2_erisc_mode = false;
    constexpr uint32_t num_dram_banks = 12;
    constexpr CoreCoord pcie_core{0, 3};

    const fs::path descriptors =
        fs::path(rtoptions.get_root_dir()) / "tt_metal/core_descriptors";
    for (const char* descriptor_name : {
             "wormhole_b0_80_arch.yaml",
             "wormhole_b0_80_arch_eth_dispatch.yaml",
             "wormhole_b0_80_arch_fabric_mux.yaml"}) {
        const YAML::Node root = YAML::LoadFile((descriptors / descriptor_name).string());
        for (const auto& product : root) {
            std::vector<bool> routing_options = {true};
            const std::string product_name = product.first.as<std::string>();
            if (product_name == "galaxy" || product_name == "nebula_x1") {
                routing_options.push_back(false);
            }
            for (const auto& axis_node : product.second) {
                const auto axis = axis_node.first.as<std::string>() == "row"
                    ? DispatchCoreAxis::ROW
                    : DispatchCoreAxis::COL;
                for (const auto& cq_node : axis_node.second) {
                    const uint8_t num_hw_cqs = cq_node.first.as<uint8_t>();
                    const auto& values = cq_node.second;
                    const auto start =
                        values["compute_with_storage_grid_range"]["start"];
                    const auto end =
                        values["compute_with_storage_grid_range"]["end"];
                    const uint32_t num_l1_banks =
                        (end[0].as<uint32_t>() - start[0].as<uint32_t>() + 1) *
                        (end[1].as<uint32_t>() - start[1].as<uint32_t>() + 1);
                    const auto dispatch_type =
                        values["dispatch_core_type"].as<std::string>() == "tensix"
                        ? DispatchCoreType::WORKER
                        : DispatchCoreType::ETH;
                    for (const bool routing_enabled : routing_options) {
                        Hal hal(
                            ARCH::WORMHOLE_B0,
                            routing_enabled,
                            enable_2_erisc_mode,
                            profiler_dram_bytes,
                            enable_dram_backed_cq,
                            false,
                            true);
                        const JitDeviceConfig config{
                            .hal = &hal,
                            .arch = ARCH::WORMHOLE_B0,
                            .num_dram_banks = num_dram_banks,
                            .num_l1_banks = num_l1_banks,
                            .pcie_core = pcie_core,
                            .harvesting_mask = 0,
                            .dispatch_core_type = dispatch_type,
                            .dispatch_core_axis = axis,
                            .coordinate_virtualization_enabled = true,
                            .dispatch_message_addr =
                                dispatch_message_addr(hal, dispatch_type),
                            .max_cbs = hal.get_arch_num_circular_buffers(),
                            .num_hw_cqs = num_hw_cqs,
                            .routing_fw_enabled = routing_enabled,
                            .profiler_dram_bank_size_per_risc_bytes =
                                profiler_dram_bytes,
                        };
                        compile_for_config(
                            config,
                            rtoptions,
                            reader_source,
                            compute_source,
                            writer_source,
                            corner_constant,
                            corner_reuse,
                            preexpanded_b,
                            canonical_packed_base,
                            canonical_packed_fallback,
                            canonical_dynamic_base,
                            canonical_dynamic_fallback,
                            compiled);
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal

int main(int argc, char** argv) {
    try {
        const bool corner_constant =
            argc >= 2 && std::string(argv[1]) == "--corner-constant";
        const bool corner_reuse =
            argc >= 2 && std::string(argv[1]) == "--corner-reuse";
        const bool preexpanded_b =
            argc >= 2 && std::string(argv[1]) == "--preexpanded-b";
        const bool canonical_packed_base =
            argc >= 2 && std::string(argv[1]) == "--canonical-packed-base";
        const bool canonical_packed_fallback =
            argc >= 2 && std::string(argv[1]) == "--canonical-packed-fallback";
        const bool canonical_dynamic_base =
            argc >= 2 && std::string(argv[1]) == "--canonical-dynamic-base";
        const bool canonical_dynamic_fallback =
            argc >= 2 && std::string(argv[1]) == "--canonical-dynamic-fallback";
        const bool special_mode =
            corner_constant || corner_reuse || preexpanded_b ||
            canonical_packed_base || canonical_packed_fallback ||
            canonical_dynamic_base || canonical_dynamic_fallback;
        if ((!special_mode && argc != 3 && argc != 4) ||
            ((corner_constant || corner_reuse) && argc != 4 && argc != 5) ||
            ((preexpanded_b || canonical_packed_base ||
              canonical_packed_fallback || canonical_dynamic_base ||
              canonical_dynamic_fallback) && argc != 5)) {
            std::cerr
                << "usage: offline_compile_affine_roles READER.cpp COMPUTE.cpp [WRITER.cpp]\n"
                << "   or: offline_compile_affine_roles --corner-constant READER.cpp COMPUTE.cpp [WRITER.cpp]\n"
                << "   or: offline_compile_affine_roles --corner-reuse READER.cpp COMPUTE.cpp [WRITER.cpp]\n"
                << "   or: offline_compile_affine_roles --preexpanded-b READER.cpp COMPUTE.cpp WRITER.cpp\n"
                << "   or: offline_compile_affine_roles --canonical-packed-base READER.cpp COMPUTE.cpp WRITER.cpp\n"
                << "   or: offline_compile_affine_roles --canonical-packed-fallback READER.cpp COMPUTE.cpp WRITER.cpp\n"
                << "   or: offline_compile_affine_roles --canonical-dynamic-base READER.cpp COMPUTE.cpp WRITER.cpp\n"
                << "   or: offline_compile_affine_roles --canonical-dynamic-fallback READER.cpp COMPUTE.cpp WRITER.cpp\n";
            return 2;
        }
        tt::llrt::RunTimeOptions rtoptions;
        bool compiled = false;
        tt::tt_metal::enumerate_matching_config(
            rtoptions,
            special_mode ? fs::path(argv[2]) : fs::path(argv[1]),
            special_mode ? fs::path(argv[3]) : fs::path(argv[2]),
            special_mode
                ? (argc == 5 ? fs::path(argv[4]) : fs::path())
                : (argc == 4 ? fs::path(argv[3]) : fs::path()),
            corner_constant,
            corner_reuse,
            preexpanded_b,
            canonical_packed_base,
            canonical_packed_fallback,
            canonical_dynamic_base,
            canonical_dynamic_fallback,
            compiled);
        if (!compiled) {
            throw std::runtime_error("QuietBox build key was not reconstructed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OFFLINE_ROLE_COMPILE pass=0 error=" << error.what() << '\n';
        return 1;
    }
}
