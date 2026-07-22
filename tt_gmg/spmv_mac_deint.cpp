// spmv_mac_deint.cpp — DE-INTERLEAVED fused-SpMV test harness (G3 throughput; Test 1 for gather_reader_deint).
// Mirrors spmv_mac.cpp's proven SPMV_GATHER path 1:1, but for the de-interleaved layout proven bit-exact in
// make_deint_op.py (layout + kernel-logic sims both rel=0.000e+00):
//   * a is OUTPUT-TILE-major a_deint[(gtile*3+r)*81 + oc][1024]  == spmv_mac [n_out][K] shape (K=81) -> shards
//     by output tile exactly like the proven harness; apage = otile*81 + oc (the kernel + sim agree on this).
//   * x is DE-INTERLEAVED: 3 node-major planes X_c[node] (each bf16x3), so the gather is 1 value/node (no div,
//     no r-broadcast); reader = gather_reader_deint (81 (o,c) gathers cached per node-block, reused across 3 r).
//   * output tiles = 3*nnode_tiles (3 r-planes); compute/writer UNCHANGED (n_out=3*ntile, K=81).
// STATUS: single-chip (SPMV_NCHIP=1, SPMV_DEINT_NTILES subset) FIRST (Test 1: rel_err == pre-stored path); 8-chip
// needs the per-chip global_off program (GATHER_DESIGN 134-140), and the 81*3-tile gather cache CB (c_12) L1
// budget (486KB) may force small npc / more cores -- resolve by device feedback. PENDING device build/run.
// Build: place under tt-metal-073/tt_metal/programming_examples/spmv_mac/, target metal_example_spmv_mac_deint.
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

static std::shared_ptr<distributed::MeshBuffer> MakeBuf(
    const std::shared_ptr<distributed::MeshDevice>& dev, uint32_t gt, uint32_t ns, uint32_t eb = sizeof(bfloat16)) {
    constexpr uint32_t H = tt::constants::TILE_HEIGHT, W = tt::constants::TILE_WIDTH; uint32_t ts = eb*H*W, st = gt/ns;
    distributed::DeviceLocalBufferConfig lc{.page_size = ts, .buffer_type = BufferType::DRAM};
    distributed::ShardedBufferConfig bc{.global_size=(uint64_t)ts*gt,.global_buffer_shape={H,gt*W},.shard_shape={H,st*W},.shard_orientation=ShardOrientation::ROW_MAJOR};
    return distributed::MeshBuffer::create(bc, lc, dev.get());
}
static std::shared_ptr<distributed::MeshBuffer> MakeRepl(
    const std::shared_ptr<distributed::MeshDevice>& dev, uint32_t gt, uint32_t eb = sizeof(bfloat16)) {
    constexpr uint32_t H = tt::constants::TILE_HEIGHT, W = tt::constants::TILE_WIDTH; uint32_t ts = eb*H*W;
    distributed::DeviceLocalBufferConfig lc{.page_size = ts, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size=(uint64_t)ts*gt}, lc, dev.get());
}
static void MakeCB(Program& p, const CoreRangeSet& cr, tt::CBIndex cb, uint32_t nt, tt::DataFormat f = tt::DataFormat::Float16_b) {
    uint32_t eb = (f==tt::DataFormat::Float32)?4u:2u, ts = eb*tt::constants::TILE_WIDTH*tt::constants::TILE_HEIGHT;
    CreateCircularBuffer(p, cr, CircularBufferConfig(nt*ts, {{cb, f}}).set_page_size(cb, ts));
}
static inline void split3(float v, bfloat16& h, bfloat16& m, bfloat16& l){ h=bfloat16(v); float r=v-(float)h; m=bfloat16(r); l=bfloat16(r-(float)m); }

int main() {
    const uint32_t TE = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    // ---- load row236_deint_op.bin: <qqq>(nb,K3=243,NBpad); X[3,NBpad]f32; nbr[27,NBpad]i32; A[243,nb]f32; REF[3,nb]f32 ----
    FILE* fp = fopen("/tmp/row236_deint_op.bin", "rb"); int64_t hd[3]; (void)!fread(hd,8,3,fp);
    const uint32_t nb=(uint32_t)hd[0], K3=(uint32_t)hd[1], NBpad=(uint32_t)hd[2];
    const uint32_t nnode_tiles = NBpad/1024;
    std::vector<float> Xf((size_t)3*NBpad); (void)!fread(Xf.data(),4,(size_t)3*NBpad,fp);
    std::vector<int32_t> nbrv((size_t)27*NBpad); (void)!fread(nbrv.data(),4,(size_t)27*NBpad,fp);
    std::vector<float> Af((size_t)243*nb); (void)!fread(Af.data(),4,(size_t)243*nb,fp);
    std::vector<float> REF((size_t)3*nb); (void)!fread(REF.data(),4,(size_t)3*nb,fp); fclose(fp);
    const uint32_t NCHIP = getenv("SPMV_NCHIP") ? (uint32_t)atoi(getenv("SPMV_NCHIP")) : 1;   // single-chip first
    const uint32_t nnt_pad = ((nnode_tiles + NCHIP - 1)/NCHIP)*NCHIP;
    const uint32_t n_out = 3*nnt_pad;                 // 3 r-planes; output tiles
    const float VSCALE = 1e6f;
    // a_deint[(gt*3+r)*81+oc][i] = Af[(o*3+r)*3+c, gt*1024+i]  (K=81=o*3+c)  -- output-tile-major, shardable
    const size_t nA = (size_t)n_out*81*TE; std::vector<bfloat16> ah(nA),am(nA),al(nA);
    for (uint32_t gt=0; gt<nnode_tiles; ++gt) for (uint32_t r=0;r<3;++r) for (uint32_t oc=0;oc<81;++oc) {
        uint32_t o=oc/3,c=oc%3, plane=(o*3+r)*3+c, ot=gt*3+r;
        for (uint32_t i=0;i<1024;++i){ uint32_t node=gt*1024+i; float v=(node<nb)?Af[(size_t)plane*nb+node]:0.f;
            size_t d=((size_t)ot*81+oc)*TE+i; split3(v,ah[d],am[d],al[d]); }
    }
    // 9 de-interleaved x planes (node-major, bf16x3): xh_c/xm_c/xl_c, length NBpad, scaled
    std::vector<bfloat16> xh[3],xm[3],xl[3];
    for (uint32_t c=0;c<3;++c){ xh[c].resize(NBpad); xm[c].resize(NBpad); xl[c].resize(NBpad);
        for (uint32_t nd=0;nd<NBpad;++nd){ float v=(nd<nb?Xf[(size_t)c*NBpad+nd]:0.f)*VSCALE; split3(v,xh[c][nd],xm[c][nd],xl[c][nd]); } }
    // ref de-interleaved, scaled: REFs[r*NBpad+node]
    std::vector<float> REFs((size_t)3*NBpad,0.f);
    for (uint32_t r=0;r<3;++r) for (uint32_t nd=0;nd<nb;++nd) REFs[(size_t)r*NBpad+nd]=REF[(size_t)r*nb+nd]*VSCALE;

    auto dev = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(1,NCHIP)));
    Program program = CreateProgram(); auto& cq = dev->mesh_command_queue();
    const char* nte = getenv("SPMV_DEINT_NTILES");    // node-tile subset for L1-bounded single-chip Test 1
    const uint32_t nl_nodes = (NCHIP==1 && nte)? std::min((uint32_t)atoi(nte),nnode_tiles) : nnt_pad/NCHIP;  // node-tiles/chip
    auto ahB=MakeBuf(dev,n_out*81,NCHIP), amB=MakeBuf(dev,n_out*81,NCHIP), alB=MakeBuf(dev,n_out*81,NCHIP);
    auto cB =MakeBuf(dev,n_out,NCHIP,4);
    std::shared_ptr<distributed::MeshBuffer> xhB[3],xmB[3],xlB[3];
    for (uint32_t c=0;c<3;++c){ xhB[c]=MakeRepl(dev,nnt_pad); xmB[c]=MakeRepl(dev,nnt_pad); xlB[c]=MakeRepl(dev,nnt_pad); }
    uint32_t nbr_tiles=((uint32_t)NBpad*27+1023)/1024; auto nbrB=MakeRepl(dev,nbr_tiles,4);

    distributed::EnqueueWriteMeshBuffer(cq,ahB,ah,true); distributed::EnqueueWriteMeshBuffer(cq,amB,am,true); distributed::EnqueueWriteMeshBuffer(cq,alB,al,true);
    for (uint32_t c=0;c<3;++c){ distributed::EnqueueWriteMeshBuffer(cq,xhB[c],xh[c],true); distributed::EnqueueWriteMeshBuffer(cq,xmB[c],xm[c],true); distributed::EnqueueWriteMeshBuffer(cq,xlB[c],xl[c],true); }
    std::vector<int32_t> nbr2((size_t)nbr_tiles*1024,-1);
    for (uint32_t nd=0;nd<NBpad;++nd) for (uint32_t o=0;o<27;++o) nbr2[(size_t)nd*27+o]=nbrv[(size_t)o*NBpad+nd];
    distributed::EnqueueWriteMeshBuffer(cq,nbrB,nbr2,true);
    std::vector<float> zc((size_t)n_out*TE,0.f); distributed::EnqueueWriteMeshBuffer(cq,cB,zc,true);

    auto grid = dev->compute_with_storage_grid_size(); CoreRangeSet all(CoreRange({0,0},{grid.x-1,grid.y-1}));
    MakeCB(program,all,tt::CBIndex::c_0,3); MakeCB(program,all,tt::CBIndex::c_1,3); MakeCB(program,all,tt::CBIndex::c_16,8,tt::DataFormat::Float32);
    // split NODE-tiles across cores (reader owns node-tiles; emits 3 output tiles each)
    auto [ncores,cores,gA,gB,nA1,nB1] = tt::tt_metal::split_work_to_cores(grid, nl_nodes, true);
    std::vector<uint32_t> pc_xlo,pc_xnt,pc_nlo,pc_nn; uint32_t s=0,max_xnt=1,max_npg=1;
    for (auto [grp,npc] : {std::make_pair(gA,nA1),std::make_pair(gB,nB1)})
        for (const auto& cr:grp.ranges()) for (const auto& cc:cr){ (void)cc;
            uint32_t nlo=s*1024, nhi=(s+npc)*1024-1; if (nhi>=NBpad) nhi=NBpad-1;
            int32_t nmin=INT32_MAX,nmax=-1;                      // NODE-unit window (de-interleaved x is node-indexed)
            for (uint32_t nd=nlo; nd<=nhi && nd<nb; ++nd) for (uint32_t o=0;o<27;++o){ int32_t v=nbrv[(size_t)o*NBpad+nd]; if(v>=0){ if(v<nmin)nmin=v; if(v>nmax)nmax=v; } }
            if (nmax<0){nmin=0;nmax=0;}
            uint32_t tlo=(uint32_t)nmin/1024, tnt=(uint32_t)nmax/1024-tlo+1;
            pc_xlo.push_back(tlo); pc_xnt.push_back(tnt); pc_nlo.push_back(nlo); pc_nn.push_back(nhi-nlo+1);
            if(tnt>max_xnt)max_xnt=tnt;
            uint32_t sil=nlo*27, plo=sil/1024, npg=(sil+(nhi-nlo+1)*27+1023)/1024-plo; if(npg>max_npg)max_npg=npg;
            s+=npc;
        }
    // CB c_2..c_10 = 9 x-windows; c_11 = nbr; c_12 = 81*3 gather cache (L1 budget: 486KB -- shrink nl_nodes if tight)
    for (int i=0;i<9;++i) MakeCB(program,all,(tt::CBIndex)((int)tt::CBIndex::c_2+i),max_xnt);
    MakeCB(program,all,tt::CBIndex::c_11,max_npg,tt::DataFormat::Float32);
    MakeCB(program,all,tt::CBIndex::c_12,81*3);
    std::vector<uint32_t> rct; for (int i=0;i<13;++i) rct.push_back((uint32_t)tt::CBIndex::c_0+ (i<11?i:(i==11?11:12)) );
    // (rct compile CB order must be: cb_a,cb_b,cb_xh0..cb_xl2 (9),cb_nbr,cb_bc == c_0,c_1,c_2..c_10,c_11,c_12)
    rct.clear(); rct={ (uint32_t)tt::CBIndex::c_0,(uint32_t)tt::CBIndex::c_1 };
    for (int i=0;i<9;++i) rct.push_back((uint32_t)tt::CBIndex::c_2+i);
    rct.push_back((uint32_t)tt::CBIndex::c_11); rct.push_back((uint32_t)tt::CBIndex::c_12);
    TensorAccessorArgs(*ahB).append_to(rct); TensorAccessorArgs(*amB).append_to(rct); TensorAccessorArgs(*alB).append_to(rct);
    for (uint32_t c=0;c<3;++c){ TensorAccessorArgs(*xhB[c]).append_to(rct); TensorAccessorArgs(*xmB[c]).append_to(rct); TensorAccessorArgs(*xlB[c]).append_to(rct); }
    TensorAccessorArgs(*nbrB).append_to(rct);
    std::vector<uint32_t> wct={(uint32_t)tt::CBIndex::c_16}; TensorAccessorArgs(*cB).append_to(wct);
    auto reader = CreateKernel(program,"tt_metal/programming_examples/spmv_mac/kernels/gather_reader_deint.cpp",cores,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default,.compile_args=rct});
    auto writer = CreateKernel(program,"tt_metal/programming_examples/spmv_mac/kernels/mac_writer.cpp",cores,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=wct});
    auto compute = CreateKernel(program,"tt_metal/programming_examples/spmv_mac/kernels/mac_compute.cpp",cores,
        ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.math_approx_mode=false,.compile_args={}});
    uint32_t start=0,ci=0;
    for (auto [grp,npc] : {std::make_pair(gA,nA1),std::make_pair(gB,nB1)})
        for (const auto& cr:grp.ranges()) for (const auto& cc:cr){
            SetRuntimeArgs(program,reader,cc,{(uint32_t)ahB->address(),(uint32_t)amB->address(),(uint32_t)alB->address(),
                (uint32_t)xhB[0]->address(),(uint32_t)xmB[0]->address(),(uint32_t)xlB[0]->address(),
                (uint32_t)xhB[1]->address(),(uint32_t)xmB[1]->address(),(uint32_t)xlB[1]->address(),
                (uint32_t)xhB[2]->address(),(uint32_t)xmB[2]->address(),(uint32_t)xlB[2]->address(),
                (uint32_t)nbrB->address(), npc, start, nnt_pad, pc_xlo[ci], pc_xnt[ci], pc_nlo[ci], pc_nn[ci]});
            SetRuntimeArgs(program,compute,cc,{3*npc,81u});                 // 3 output tiles / node-tile, K=81
            SetRuntimeArgs(program,writer,cc,{(uint32_t)cB->address(),3*npc,3*start});
            start+=npc; ci++;
        }
    distributed::MeshWorkload wl; wl.add_program(distributed::MeshCoordinateRange(dev->shape()),std::move(program));
    distributed::EnqueueMeshWorkload(cq,wl,true);
    auto t0=std::chrono::high_resolution_clock::now(); const int reps=20;
    for (int i=0;i<reps;++i) distributed::EnqueueMeshWorkload(cq,wl,false); Finish(cq);
    double ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/reps;
    std::vector<float> cd; distributed::EnqueueReadMeshBuffer(cq,cd,cB,true);
    // validate: cd[(gt*3+r)*1024+i] == REFs[r*NBpad + gt*1024+i]
    double maxerr=0,scale=0; size_t chk=(size_t)nl_nodes; if(NCHIP>1) chk=(size_t)nnt_pad/NCHIP;
    for (uint32_t gt=0; gt<chk && gt<nnode_tiles; ++gt) for (uint32_t r=0;r<3;++r) for (uint32_t i=0;i<1024;++i){
        size_t ci2=((size_t)(gt*3+r))*1024+i; if(ci2>=cd.size())continue; float v=cd[ci2]; float ref=REFs[(size_t)r*NBpad+gt*1024+i];
        scale=std::max(scale,(double)std::abs(ref)); if(std::isfinite(v)) maxerr=std::max(maxerr,std::abs((double)v-ref));
    }
    double gb=(double)n_out*81*6*TE*sizeof(bfloat16)/1e9;
    printf("SpMV-MAC-DEINT: %.3f ms/apply  %.0f GB/s  rel_err=%.3e  (nb=%u out_tiles=%u nl_nodes=%u)\n",
           ms, gb/(ms/1000.0), maxerr/(scale+1e-30), nb, n_out, nl_nodes);
    return 0;
}
