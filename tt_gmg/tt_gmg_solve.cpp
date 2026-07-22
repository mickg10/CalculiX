// tt_gmg_solve.cpp — the TT-GMG driver: persistent 8-chip bf16x3 fine-SpMV wired into the CCX GMG-PCG.
//   setup(): mesh + resident coeff (a_hi/mid/lo) + nbr gather table + program (built once, G1/G2).
//   tt_fine_spmv(x,y): gather b=x[nbr] (DIA), split bf16x3, upload b, run kernel, download fp32 y  (called ~228x).
// gmg_solve.cpp's g_tt_fine_spmv hook routes the fine (lv==0) smoother here; outer PCG stays exact fp64 on host.
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
using namespace tt;
using namespace tt::tt_metal;
namespace dist = tt::tt_metal::distributed;

extern "C" {
    int ccx_gmg_solve_from_dump(const char*, int, double, int, double*, double*);
    extern int  (*g_tt_fine_spmv)(const double*, double*);   // hook symbols owned by gmg_solve.cpp
    extern long g_tt_fine_n;
}

// ---- persistent TT state ----
static std::shared_ptr<dist::MeshDevice> g_dev;
static std::shared_ptr<dist::MeshBuffer> g_ah,g_am,g_al,g_bh,g_bm,g_bl,g_c;
static Program g_prog;
static std::unique_ptr<dist::MeshWorkload> g_wl;
static uint32_t g_n_out=0, g_K=81, g_NCHIP=8, g_TE=1024, g_n=0;
static std::vector<int32_t> g_nbr;   // [27*nb]
static long g_nb=0;
static long g_napply=0; static double g_apply_s=0;

static std::shared_ptr<dist::MeshBuffer> Mk(uint32_t gt,uint32_t eb){
    uint32_t H=constants::TILE_HEIGHT,W=constants::TILE_WIDTH, ts=eb*H*W, sh=gt/g_NCHIP;
    dist::DeviceLocalBufferConfig lc{.page_size=ts,.buffer_type=BufferType::DRAM};
    dist::ShardedBufferConfig bc{.global_size=(uint64_t)ts*gt,.global_buffer_shape={H,gt*W},
        .shard_shape={H,sh*W},.shard_orientation=ShardOrientation::ROW_MAJOR};
    return dist::MeshBuffer::create(bc,lc,g_dev.get());
}
static void MkCB(const CoreRangeSet&cs,CBIndex cb,uint32_t nt,DataFormat fmt=DataFormat::Float16_b){
    uint32_t eb=(fmt==DataFormat::Float32)?4u:2u, ts=eb*constants::TILE_WIDTH*constants::TILE_HEIGHT;
    CreateCircularBuffer(g_prog,cs,CircularBufferConfig(nt*ts,{{cb,fmt}}).set_page_size(cb,ts));
}
static inline void split3(float v,bfloat16&h,bfloat16&m,bfloat16&l){ h=bfloat16(v); float r=v-(float)h; m=bfloat16(r); l=bfloat16(r-(float)m); }

// gather b = x[nbr] in DIA order and tile-lay + bf16x3 split into bh/bm/bl (host).
static void gather_split(const double* x, std::vector<bfloat16>&bh,std::vector<bfloat16>&bm,std::vector<bfloat16>&bl){
    const uint32_t K=g_K,TE=g_TE; const long nb=g_nb,n=g_n;
    #pragma omp parallel for schedule(static)
    for(long t=0;t<g_n_out;t++) for(uint32_t k=0;k<K;k++){ int o=k/3,c=k%3; const int32_t* nr=&g_nbr[(size_t)o*nb];
        size_t base=((size_t)t*K+k)*TE;
        for(uint32_t e=0;e<TE;e++){ long i=(long)t*TE+e; float bv=0.f;
            if(i<n){ long node=i/3; int32_t nn=nr[node]; if(nn>=0) bv=(float)x[3*(long)nn+c]; }
            split3(bv,bh[base+e],bm[base+e],bl[base+e]); } }
}

// ---- the fine-SpMV apply called by the GMG smoother ----
static int tt_fine_spmv(const double* x, double* y){
    auto t0=std::chrono::high_resolution_clock::now();
    const size_t nA=(size_t)g_n_out*g_K*g_TE;
    static std::vector<bfloat16> bh,bm,bl; if(bh.empty()){bh.resize(nA);bm.resize(nA);bl.resize(nA);}
    gather_split(x,bh,bm,bl);
    auto&cq=g_dev->mesh_command_queue();
    dist::EnqueueWriteMeshBuffer(cq,g_bh,bh,true); dist::EnqueueWriteMeshBuffer(cq,g_bm,bm,true);
    dist::EnqueueWriteMeshBuffer(cq,g_bl,bl,true);
    dist::EnqueueMeshWorkload(cq,*g_wl,true);            // BLOCKING: PCG needs a deterministic fixed operator
    std::vector<float> cd; dist::EnqueueReadMeshBuffer(cq,cd,g_c,true);
    for(long i=0;i<g_n;i++) y[i]=cd[i];
    g_napply++; g_apply_s += std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    return 0;
}

static void setup(const char* coeff_bin,const char* nbr_bin){
    // nbr
    FILE* fn=fopen(nbr_bin,"rb"); int64_t h2[2]; (void)!fread(h2,8,2,fn); g_nb=h2[1];
    g_nbr.resize((size_t)27*g_nb); (void)!fread(g_nbr.data(),4,(size_t)27*g_nb,fn); fclose(fn);
    // coeff (a) tile-laid, x-independent -> resident bf16x3
    FILE* fc=fopen(coeff_bin,"rb"); int64_t h[2]; (void)!fread(h,8,2,fc);
    g_n_out=(uint32_t)h[0]; g_K=(uint32_t)h[1]; g_n=(uint32_t)(3*g_nb);
    size_t nA=(size_t)g_n_out*g_K*g_TE;
    std::vector<float> adf(nA); (void)!fread(adf.data(),4,nA,fc); fclose(fc);   // only coeff needed
    std::vector<bfloat16> ah(nA),am(nA),al(nA); for(size_t i=0;i<nA;i++) split3(adf[i],ah[i],am[i],al[i]);
    g_dev=dist::MeshDevice::create(dist::MeshDeviceConfig(dist::MeshShape(1,g_NCHIP)));
    g_ah=Mk(g_n_out*g_K,2);g_am=Mk(g_n_out*g_K,2);g_al=Mk(g_n_out*g_K,2);
    g_bh=Mk(g_n_out*g_K,2);g_bm=Mk(g_n_out*g_K,2);g_bl=Mk(g_n_out*g_K,2); g_c=Mk(g_n_out,4);
    auto&cq=g_dev->mesh_command_queue();
    dist::EnqueueWriteMeshBuffer(cq,g_ah,ah,false);dist::EnqueueWriteMeshBuffer(cq,g_am,am,false);dist::EnqueueWriteMeshBuffer(cq,g_al,al,true);
    g_prog=CreateProgram(); uint32_t n_local=g_n_out/g_NCHIP;
    auto grid=g_dev->compute_with_storage_grid_size(); CoreRange all({0,0},{grid.x-1,grid.y-1}); CoreRangeSet cs(all);
    MkCB(cs,CBIndex::c_0,6);MkCB(cs,CBIndex::c_1,6);MkCB(cs,CBIndex::c_16,8,DataFormat::Float32);
    auto[nc,cores,g1,g2,n1,n2]=tt_metal::split_work_to_cores(grid,n_local,true);
    std::vector<uint32_t> rct={(uint32_t)CBIndex::c_0,(uint32_t)CBIndex::c_1};
    TensorAccessorArgs(*g_ah).append_to(rct);TensorAccessorArgs(*g_am).append_to(rct);TensorAccessorArgs(*g_al).append_to(rct);
    TensorAccessorArgs(*g_bh).append_to(rct);TensorAccessorArgs(*g_bm).append_to(rct);TensorAccessorArgs(*g_bl).append_to(rct);
    std::vector<uint32_t> wct={(uint32_t)CBIndex::c_16}; TensorAccessorArgs(*g_c).append_to(wct);
    auto rd=CreateKernel(g_prog,"tt_metal/programming_examples/tt_gmg_solve/kernels/mac_reader.cpp",cores,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default,.compile_args=rct});
    auto wr=CreateKernel(g_prog,"tt_metal/programming_examples/tt_gmg_solve/kernels/mac_writer.cpp",cores,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=wct});
    auto cp=CreateKernel(g_prog,"tt_metal/programming_examples/tt_gmg_solve/kernels/mac_compute.cpp",cores,
        ComputeConfig{.fp32_dest_acc_en=true,.math_approx_mode=false,.compile_args={}});
    uint32_t start=0;
    for(auto[grp,npc]:{std::make_pair(g1,n1),std::make_pair(g2,n2)}) for(const auto&cr:grp.ranges()) for(const auto&cc:cr){
        SetRuntimeArgs(g_prog,rd,cc,{(uint32_t)g_ah->address(),(uint32_t)g_am->address(),(uint32_t)g_al->address(),
            (uint32_t)g_bh->address(),(uint32_t)g_bm->address(),(uint32_t)g_bl->address(),npc,g_K,start});
        SetRuntimeArgs(g_prog,cp,cc,{npc,g_K}); SetRuntimeArgs(g_prog,wr,cc,{(uint32_t)g_c->address(),npc,start}); start+=npc; }
    g_wl=std::make_unique<dist::MeshWorkload>(); g_wl->add_program(dist::MeshCoordinateRange(g_dev->shape()),std::move(g_prog));
    g_tt_fine_n=g_n; g_tt_fine_spmv=tt_fine_spmv;
}

int main(int argc,char**argv){
    const char* dump = argc>1?argv[1]:"/tmp/row236_fine.bin";
    printf("[tt-gmg] setup (mesh + resident coeff + nbr)...\n"); fflush(stdout);
    auto ts=std::chrono::high_resolution_clock::now();
    setup("/tmp/row236_real_op.bin","/tmp/row236_nbr.bin");
    printf("[tt-gmg] setup done %.2fs\n",
           std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-ts).count()); fflush(stdout);
    // --- self-check: tt_fine_spmv(v) vs CPU B*v on a DENSE RANDOM v (a single-vector check is insufficient:
    //     tt could equal B+E with E*bp~0 but E*v!=0). Load B (BCSR) from the dump and compare. ---
    { FILE* f=fopen(dump,"rb"); int64_t hh[5]; (void)!fread(hh,8,5,f); long nb=hh[0],nblk=hh[1]; int n=3*nb;
      std::vector<long> ptr(nb+1); std::vector<int> col(nblk); std::vector<double> val((size_t)nblk*9);
      (void)!fread(ptr.data(),8,nb+1,f); (void)!fread(col.data(),4,nblk,f); (void)!fread(val.data(),8,(size_t)nblk*9,f); fclose(f);
      std::vector<double> v(n),yc(n),yt(n); uint64_t s=99; for(int i=0;i<n;i++){ s^=s<<13;s^=s>>7;s^=s<<17; v[i]=(double)((s>>11)&1023)/1024.0-0.5; }
      for(long I=0;I<nb;I++){ double y0=0,y1=0,y2=0; for(long b=ptr[I];b<ptr[I+1];b++){ const double*m=&val[(size_t)b*9]; const double*x=&v[3*col[b]];
          y0+=m[0]*x[0]+m[1]*x[1]+m[2]*x[2]; y1+=m[3]*x[0]+m[4]*x[1]+m[5]*x[2]; y2+=m[6]*x[0]+m[7]*x[1]+m[8]*x[2]; } yc[3*I]=y0;yc[3*I+1]=y1;yc[3*I+2]=y2; }
      tt_fine_spmv(v.data(),yt.data());
      double mx=0,sc=0; long bi=0; for(int i=0;i<n;i++){ double e=std::fabs(yt[i]-yc[i]); if(e>mx){mx=e;bi=i;} sc=std::max(sc,std::fabs(yc[i])); }
      printf("[tt-gmg] SELF-CHECK tt_fine_spmv(rand) vs CPU B*rand: rel=%.3e  worst i=%ld tt=%.4e cpu=%.4e\n",mx/(sc+1e-30),bi,yt[bi],yc[bi]); fflush(stdout); }
    printf("[tt-gmg] solving row236 with TT fine-SpMV...\n"); fflush(stdout);
    int maxit = argc>2 ? atoi(argv[2]) : 500;
    double maxu=0; int rc=ccx_gmg_solve_from_dump(dump,maxit,1e-6,1,nullptr,&maxu);
    printf("[tt-gmg] rc=%d maxU=%.7f  TT applies=%ld total=%.2fs avg=%.1fms/apply\n",
           rc,maxu,g_napply,g_apply_s,g_napply?1000*g_apply_s/g_napply:0);
    return rc;
}
