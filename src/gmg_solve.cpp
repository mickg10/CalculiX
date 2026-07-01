// gmg_solve.cpp — geometric multigrid solver for CCX, callable from accel_solver.c.
// The target problem is a homogeneous regular Cartesian voxel-grid C3D8 model (3 DOF/node).
// Method: support-closed trilinear prolongation P + Galerkin coarse operator A_c = P^T A P + block-Jacobi
// 4th-kind Chebyshev smoother + PCG (V/W-cycle preconditioner). The result is gated on the true residual.
// Entry: ccx_gmg_solve_mem(n, cs, ri, va, b, co, na, nk, mt, ...) -- in-memory coords, the path CalculiX uses;
// ccx_gmg_solve(..., coordmap_path, ...) is a file-based compatibility wrapper. Returns 0 on success (-> b).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif
// GMG solve thread policy: on this gather-latency-bound kernel, using all logical cores is
// slower than the performance-core count, because efficiency cores cause OpenMP barrier load-imbalance. Cap the
// SOLVE to the P-core count (auto-detected on Apple Silicon; ambient elsewhere) while leaving CCX assembly/output
// at ambient threads. Override with GMG_THREADS. The cap affects speed only, never the result.
namespace {
int gmg_perf_core_default(int ambient){
#if defined(__APPLE__) && defined(__aarch64__)
    int p=0; size_t sz=sizeof p;
    if(sysctlbyname("hw.perflevel0.physicalcpu",&p,&sz,nullptr,0)==0 && p>0 && p<ambient) return p;  // P-cores
#endif
    return ambient;
}
struct OmpThreadCap {       // RAII: cap on entry, restore ambient on every exit (returns + catch)
#ifdef _OPENMP
    int saved;
    explicit OmpThreadCap(int cap): saved(omp_get_max_threads()){ if(cap>0 && cap!=saved) omp_set_num_threads(cap); }
    ~OmpThreadCap(){ omp_set_num_threads(saved); }
#else
    explicit OmpThreadCap(int){}
#endif
};
} // namespace
// Coarse dense solve uses LAPACK (dpotrf/dpotrs). Apple Accelerate is the default on macOS (and is where AMX
// lives); the rest of the solver is portable OpenMP C++. Compile-time control (-D...):
//   default on macOS         -> CCX_GMG_ACCEL=1  (Apple Accelerate LAPACK)
//   -DCCX_GMG_NOACCEL        -> portable reference: standard LAPACK Fortran ABI (link OpenBLAS/MKL/Netlib)
//   -DCCX_GMG_ACCEL=1 / =0   -> force either path explicitly
#if defined(CCX_GMG_ACCEL)
  /* caller forced it explicitly */
#elif defined(__APPLE__) && !defined(CCX_GMG_NOACCEL)
  #define CCX_GMG_ACCEL 1
#else
  #define CCX_GMG_ACCEL 0
#endif
#if CCX_GMG_ACCEL
  #include <Accelerate/Accelerate.h>   // Accelerate LAPACK dpotrf_/dpotrs_ (AMX-backed)
#else
  extern "C" {                          // portable LAPACK Fortran ABI (OpenBLAS / MKL / Netlib)
    void dpotrf_(const char*, const int*, double*, const int*, int*);
    void dpotrs_(const char*, const int*, const int*, const double*, const int*, double*, const int*, int*);
  }
#endif

namespace {
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b){ return std::chrono::duration<double>(b-a).count(); }

// --- TT-precision emulation (research probe, off by default) ---------------------------------------
// GMG_EMU_BF16=1 rounds the SMOOTHER's SpMV and block-Jacobi results to bf16 (8-bit mantissa, round-to-
// nearest-even) to emulate running the V-cycle smoother on a Tenstorrent Wormhole (bf16 matrix engine,
// fp32 accumulate). The fp64 OUTER PCG, the exact double outer operator bspmv(A0), and the coarse solve
// are UNCHANGED -- this measures only the convergence cost of a low-precision preconditioner on row236.
static int g_emu_bf16 = 0;
static int g_emu_bf16_maxlv = 1000;   // bf16 only on levels <= this (env GMG_EMU_BF16_MAXLV; 0 = finest only)
static int g_emu_mode = 1;            // compensated-SpMV precision (GMG_EMU_MODE): 1=bf16x1, 2=bf16x2(~fp16+), 3=bf16x3(~fp32), 32=fp32
// TT-GMG hook: when set by the Tenstorrent driver, the FINE-level (lv==0) smoother SpMV y=A0*x runs on the
// 8-chip mesh (bf16x3, ~fp32). Coarser levels, the vcycle residual, and the outer PCG operator stay exact fp64
// on the host -> the true-residual acceptance gate is unchanged (a wrong TT result can never be accepted).
extern "C" { int (*g_tt_fine_spmv)(const double* x, double* y) = nullptr;   // returns 0 on success (set by TT driver)
             long g_tt_fine_n = 0; }  // n (=3*nb) the hook expects; guards accidental level/size mismatch
static int g_emu_mbits = 0;   // >0: round fine-SpMV output to this many mantissa bits (probe matmul-diagonal precision)
static inline double round_mbits(double x, int mb){
    if(x==0.0 || !std::isfinite(x)) return x;
    int e; double m = std::frexp(x, &e); double s = std::ldexp(1.0, mb);
    return std::ldexp(std::round(m*s)/s, e);
}
static double g_emu_abserr = 0.0;  // >0: add per-row noise ~ abserr*sum|m*x| to fine SpMV (emulates the bf16-PRODUCT
                                   // absolute error that swamps cancellation — the real TT matmul-diagonal failure)
static int g_defl_rbm = 0;         // >0: deflate 6 rigid-body modes from the fine SpMV input+output (smoother side)
static int g_defl_corr = 0;        // >0: add the exact fp64 RBM coarse-correction to the PCG preconditioner
static double g_defl_reg = 0.0;    // E-regularization (relative): DE[a,a] += reg*maxdiag before Cholesky
static int g_defl_deg = 1;         // polynomial deflation degree (1=RBMs; higher enlarges the near-null subspace)
                                      // multi-pass = Ozaki/error-free-transform: split each value into hi/lo bf16 terms,
                                      // do extra bf16 products, accumulate fp32 -> recover precision from fast bf16 passes.
static inline double bf16round(double d){
    float f=(float)d; uint32_t u; std::memcpy(&u,&f,4);
    uint32_t rb=(u>>16)&1u; u += 0x7FFFu + rb;            // round-to-nearest-even into the top 16 bits
    u &= 0xFFFF0000u; std::memcpy(&f,&u,4); return (double)f;
}
static inline void bf16vec(double*v,int n){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;i++) v[i]=bf16round(v[i]);
}
static inline float bf16f(float f){                       // round a float to bf16 (RNE)
    uint32_t u; std::memcpy(&u,&f,4); uint32_t rb=(u>>16)&1u; u += 0x7FFFu + rb;
    u &= 0xFFFF0000u; std::memcpy(&f,&u,4); return f;
}

struct CSR { int nr=0,nc=0; std::vector<long> ptr; std::vector<int> col; std::vector<double> val;
             long nnz() const { return ptr.empty()?0:ptr.back(); } };

static void spmv(const CSR&A,const double*x,double*y){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<A.nr;i++){ double s=0; for(long p=A.ptr[i];p<A.ptr[i+1];p++) s+=A.val[p]*x[A.col[p]]; y[i]=s; }
}
// block-3x3 CSR. Profiling showed the SpMV is matrix-STREAM-bound (val+col), x is SLC-resident.
// BCSR collapses col indices 9x (297M->33M, col 1.19GB->0.13GB) -> ~1.4x fine SpMV, exact. nb block-rows,
// each block = 9 doubles row-major (3x3); block (I,J)[r][c] = A[3I+r, 3J+c].
struct BCSR { int nb=0; std::vector<long> ptr; std::vector<int> col; std::vector<double> val; };
static BCSR to_bcsr(const CSR&A){
    int nb=A.nr/3; BCSR B; B.nb=nb; B.ptr.assign(nb+1,0);
    #pragma omp parallel
    { std::vector<int> mark(nb,-1);
      #pragma omp for schedule(dynamic,1024)
      for(int I=0;I<nb;I++){ long cnt=0;
        for(int r=0;r<3;r++){ int row=3*I+r; for(long p=A.ptr[row];p<A.ptr[row+1];p++){ int J=A.col[p]/3;
          if(mark[J]!=I){mark[J]=I;cnt++;} } }
        B.ptr[I+1]=cnt; } }
    for(int I=0;I<nb;I++) B.ptr[I+1]+=B.ptr[I];
    B.col.resize(B.ptr.back()); B.val.assign((size_t)B.ptr.back()*9,0.0);
    #pragma omp parallel
    { std::vector<int> last(nb,-1); std::vector<long> slot(nb,0);   // I-VALUE marker: order-independent (dynamic-sched safe)
      #pragma omp for schedule(dynamic,1024)
      for(int I=0;I<nb;I++){ long b=B.ptr[I];
        for(int r=0;r<3;r++){ int row=3*I+r; for(long p=A.ptr[row];p<A.ptr[row+1];p++){ int J=A.col[p]/3,c=A.col[p]%3;
          if(last[J]!=I){ last[J]=I; slot[J]=b; B.col[b]=J; b++; }
          B.val[(size_t)slot[J]*9 + r*3 + c]+=A.val[p]; } } }  // += handles CCX unsummed duplicate (r,c)
    }
    #pragma omp parallel for schedule(dynamic,1024)
    for(int I=0;I<B.nb;I++){ long s=B.ptr[I],e=B.ptr[I+1];   // insertion-sort each block-row by col (rows are short ~27)
        for(long a=s+1;a<e;a++){ int jc=B.col[a]; double tmp[9]; for(int k=0;k<9;k++) tmp[k]=B.val[(size_t)a*9+k];
            long b=a-1; while(b>=s && B.col[b]>jc){ B.col[b+1]=B.col[b]; for(int k=0;k<9;k++) B.val[(size_t)(b+1)*9+k]=B.val[(size_t)b*9+k]; b--; }
            B.col[b+1]=jc; for(int k=0;k<9;k++) B.val[(size_t)(b+1)*9+k]=tmp[k]; } }
    return B;
}
static void bspmv(const BCSR&B,const double*x,double*y){
    #pragma omp parallel for schedule(static)
    for(int I=0;I<B.nb;I++){ double y0=0,y1=0,y2=0;
        for(long b=B.ptr[I];b<B.ptr[I+1];b++){ const double*m=&B.val[(size_t)b*9]; const double*xx=&x[3*B.col[b]];
            double x0=xx[0],x1=xx[1],x2=xx[2];
            y0+=m[0]*x0+m[1]*x1+m[2]*x2; y1+=m[3]*x0+m[4]*x1+m[5]*x2; y2+=m[6]*x0+m[7]*x1+m[8]*x2; }
        y[3*I]=y0; y[3*I+1]=y1; y[3*I+2]=y2; }
}
// Tenstorrent-Wormhole SpMV emulation with a COMPENSATED (Ozaki/error-free-transform) precision ladder.
// Each fp value is split into bf16 terms hi + lo (+ lo2); the product uses 1/3/6 bf16 sub-products so the
// result recovers ~bf16 / ~fp16+ / ~fp32 accuracy from fast bf16 passes, accumulated in fp32. This models
// what a real TT kernel would do: N bf16 matmul passes (N*throughput) to reach the accuracy PCG needs.
static inline float prod_emu(double mval,double xval){
    if(g_emu_mode==32) return (float)mval*(float)xval;          // fp32 reference
    float mf=(float)mval, xf=(float)xval, mh=bf16f(mf), xh=bf16f(xf);
    float p=mh*xh;                                              // pass 1: bf16 x bf16
    if(g_emu_mode>=2){ float ml=bf16f(mf-mh), xl=bf16f(xf-xh); p+=mh*xl+ml*xh;             // +2 cross terms (~16-bit)
        if(g_emu_mode>=3){ float ml2=bf16f(mf-mh-ml), xl2=bf16f(xf-xh-xl); p+=ml*xl+mh*xl2+ml2*xh; } } // ~24-bit ≈ fp32
    return p;
}
static void bspmv_emu(const BCSR&B,const double*x,double*y){
    #pragma omp parallel for schedule(static)
    for(int I=0;I<B.nb;I++){ float y0=0,y1=0,y2=0;
        for(long b=B.ptr[I];b<B.ptr[I+1];b++){ const double*m=&B.val[(size_t)b*9]; const double*xx=&x[3*B.col[b]];
            y0+=prod_emu(m[0],xx[0])+prod_emu(m[1],xx[1])+prod_emu(m[2],xx[2]);
            y1+=prod_emu(m[3],xx[0])+prod_emu(m[4],xx[1])+prod_emu(m[5],xx[2]);
            y2+=prod_emu(m[6],xx[0])+prod_emu(m[7],xx[1])+prod_emu(m[8],xx[2]); }
        if(g_emu_mode==1){ y[3*I]=bf16round((double)y0); y[3*I+1]=bf16round((double)y1); y[3*I+2]=bf16round((double)y2); }
        else { y[3*I]=(double)y0; y[3*I+1]=(double)y1; y[3*I+2]=(double)y2; } }   // multi-pass keeps fp32 result
}
// 3x3 block-diagonal inverse straight from BCSR
static void block_diag_inv_b(const BCSR&B, std::vector<double>&Dinv){
    int nb=B.nb; Dinv.assign((size_t)9*nb,0.0);
    #pragma omp parallel for schedule(static)
    for(int I=0;I<nb;I++){ const double*M=nullptr;
        for(long b=B.ptr[I];b<B.ptr[I+1];b++) if(B.col[b]==I){ M=&B.val[(size_t)b*9]; break; }
        double a,bb,c,d,e,f,g,h,i;
        if(M){a=M[0];bb=M[1];c=M[2];d=M[3];e=M[4];f=M[5];g=M[6];h=M[7];i=M[8];} else {a=e=i=1;bb=c=d=f=g=h=0;}
        double A00=e*i-f*h,A01=-(bb*i-c*h),A02=bb*f-c*e,A10=-(d*i-f*g),A11=a*i-c*g,A12=-(a*f-c*d),A20=d*h-e*g,A21=-(a*h-bb*g),A22=a*e-bb*d;
        double det=a*A00+bb*A10+c*A20; double id=(det!=0)?1.0/det:0.0; double*o=&Dinv[9*I];
        o[0]=A00*id;o[1]=A01*id;o[2]=A02*id;o[3]=A10*id;o[4]=A11*id;o[5]=A12*id;o[6]=A20*id;o[7]=A21*id;o[8]=A22*id; }
}
static CSR spmm(const CSR&A,const CSR&B){
    CSR C; C.nr=A.nr; C.nc=B.nc; C.ptr.assign(A.nr+1,0); int n=B.nc;
    #pragma omp parallel
    { std::vector<int> mark(n,-1);
      #pragma omp for schedule(dynamic,2048)
      for(int i=0;i<A.nr;i++){ long cnt=0;
        for(long pa=A.ptr[i];pa<A.ptr[i+1];pa++){ int a=A.col[pa];
          for(long pb=B.ptr[a];pb<B.ptr[a+1];pb++){ int bc=B.col[pb]; if(mark[bc]!=i){mark[bc]=i;cnt++;} } }
        C.ptr[i+1]=cnt; } }
    for(int i=0;i<A.nr;i++) C.ptr[i+1]+=C.ptr[i];
    C.col.resize(C.ptr.back()); C.val.resize(C.ptr.back());
    #pragma omp parallel
    { std::vector<int> nxt(n,-1); std::vector<double> acc(n,0.0);
      #pragma omp for schedule(dynamic,2048)
      for(int i=0;i<A.nr;i++){ int head=-2; long len=0;
        for(long pa=A.ptr[i];pa<A.ptr[i+1];pa++){ int a=A.col[pa]; double va=A.val[pa];
          for(long pb=B.ptr[a];pb<B.ptr[a+1];pb++){ int bc=B.col[pb]; acc[bc]+=va*B.val[pb];
            if(nxt[bc]==-1){ nxt[bc]=head; head=bc; len++; } } }
        long p=C.ptr[i];
        for(long t=0;t<len;t++){ int c=head; head=nxt[c]; C.col[p]=c; C.val[p]=acc[c]; p++; acc[c]=0; nxt[c]=-1; } } }
    #pragma omp parallel for schedule(dynamic,2048)
    for(int i=0;i<C.nr;i++){ long s=C.ptr[i],e=C.ptr[i+1]; std::vector<std::pair<int,double>> t(e-s);
        for(long p=s;p<e;p++) t[p-s]={C.col[p],C.val[p]}; std::sort(t.begin(),t.end());
        for(long p=s;p<e;p++){ C.col[p]=t[p-s].first; C.val[p]=t[p-s].second; } }
    return C;
}
static CSR transpose(const CSR&A){
    CSR T; T.nr=A.nc; T.nc=A.nr; T.ptr.assign(A.nc+1,0);
    for(long p=0;p<A.nnz();p++) T.ptr[A.col[p]+1]++;
    for(int i=0;i<A.nc;i++) T.ptr[i+1]+=T.ptr[i];
    T.col.resize(A.nnz()); T.val.resize(A.nnz());
    std::vector<long> cur(T.ptr.begin(),T.ptr.end()-1);
    for(int i=0;i<A.nr;i++) for(long p=A.ptr[i];p<A.ptr[i+1];p++){ int c=A.col[p]; long d=cur[c]++; T.col[d]=i; T.val[d]=A.val[p]; }
    return T;
}
static CSR build_P(const std::vector<long>&ijk,int nbf,std::vector<long>&cijk_out,int&nbc_out){
    static const int D[8][3]={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,0,1},{0,1,1},{1,1,1}};
    std::vector<int> rrow; std::vector<int> rk; std::vector<double> rw;
    rrow.reserve((size_t)nbf*4); rk.reserve((size_t)nbf*4); rw.reserve((size_t)nbf*4);
    std::unordered_map<long,int> cid; cid.reserve(nbf); std::vector<long> cijk;
    for(int i=0;i<nbf;i++){ long ix=ijk[3*i],iy=ijk[3*i+1],iz=ijk[3*i+2];
        long bx=(ix>>1)<<1,by=(iy>>1)<<1,bz=(iz>>1)<<1; int fx=(int)(ix-bx),fy=(int)(iy-by),fz=(int)(iz-bz);
        for(int d=0;d<8;d++){ double wx=fx?0.5:(D[d][0]?0.0:1.0), wy=fy?0.5:(D[d][1]?0.0:1.0), wz=fz?0.5:(D[d][2]?0.0:1.0);
            double w=wx*wy*wz; if(w==0) continue;
            long px=bx+2*D[d][0],py=by+2*D[d][1],pz=bz+2*D[d][2]; long key=(px*100000+py)*100000+pz;
            auto it=cid.find(key); int cc;
            if(it==cid.end()){ cc=(int)cijk.size()/3; cid.emplace(key,cc); cijk.push_back(px>>1);cijk.push_back(py>>1);cijk.push_back(pz>>1);} else cc=it->second;
            rrow.push_back(i); rk.push_back(cc); rw.push_back(w); } }
    int nbc=(int)cijk.size()/3; nbc_out=nbc; cijk_out=cijk;
    CSR P; P.nr=3*nbf; P.nc=3*nbc; P.ptr.assign(3*nbf+1,0);
    std::vector<int> deg(nbf,0); for(size_t t=0;t<rrow.size();t++) deg[rrow[t]]++;
    for(int i=0;i<nbf;i++){ P.ptr[3*i+1]=deg[i];P.ptr[3*i+2]=deg[i];P.ptr[3*i+3]=deg[i]; }
    for(int r=0;r<3*nbf;r++) P.ptr[r+1]+=P.ptr[r];
    P.col.resize(P.ptr.back()); P.val.resize(P.ptr.back());
    std::vector<long> cur(3*nbf); for(int r=0;r<3*nbf;r++) cur[r]=P.ptr[r];
    for(size_t t=0;t<rrow.size();t++){ int i=rrow[t]; int cc=rk[t]; double w=rw[t];
        for(int c=0;c<3;c++){ long d=cur[3*i+c]++; P.col[d]=3*cc+c; P.val[d]=w; } }
    return P;
}
static void bj_apply(const std::vector<double>&Dinv,const double*r,double*z,int n){
    int nb=n/3;
    #pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++){ const double*M=&Dinv[9*b]; const double*rr=&r[3*b]; double*zz=&z[3*b];
        zz[0]=M[0]*rr[0]+M[1]*rr[1]+M[2]*rr[2]; zz[1]=M[3]*rr[0]+M[4]*rr[1]+M[5]*rr[2]; zz[2]=M[6]*rr[0]+M[7]*rr[1]+M[8]*rr[2]; }
}
// Determinism: the OpenMP reduction combines per-thread partial sums in an unspecified order, so this
// dot product is NOT bit-identical run-to-run even at a fixed thread count. That perturbs the PCG path at the
// ~1e-16 level; the iteration still converges to the same true residual, so maxU is reproducible to solver
// tolerance -- far inside the ±1% acceptance bound. Bit-exact reproducibility would need an ordered tree reduction.
static double ddot(const double*a,const double*b,int n){ double s=0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for(int i=0;i<n;i++) s+=a[i]*b[i]; return s; }
static double lmax(const BCSR&B,const std::vector<double>&Dinv){
    int n=B.nb*3; std::vector<double> v(n),w(n),t(n); uint64_t s=88172645463325252ull;
    for(int i=0;i<n;i++){ s^=s<<13; s^=s>>7; s^=s<<17; v[i]=(double)((s>>11)&0xFFFFF)/524288.0-1.0; }
    double nv=std::sqrt(ddot(v.data(),v.data(),n)); for(int i=0;i<n;i++) v[i]/=nv; double lam=1;
    for(int it=0;it<30;it++){ bspmv(B,v.data(),w.data()); bj_apply(Dinv,w.data(),t.data(),n);
        double nw=std::sqrt(ddot(t.data(),t.data(),n)); if(nw<=0) break; lam=nw; for(int i=0;i<n;i++) v[i]=t[i]/nw; }
    return lam;
}
struct Level { CSR A,P,Pt; BCSR B; std::vector<double> Dinv; double rho; std::vector<long> ijk; };
struct WS { std::vector<double> r,z,d,Ad,res,rc,ec; };

// Polynomial deflation basis: per-component monomials of the (normalized) lattice coords up to total degree `deg`,
// Gram-Schmidt orthonormalized (near-dependent dropped). deg>=1 already spans the 6 rigid-body modes; higher deg
// approximates the broader near-null eigenspace that drives the bf16-product cancellation divergence.
static std::vector<std::vector<double>> g_Z;
static void build_defl(const Level&L,int deg){
    int nb=L.B.nb, n=3*nb; if((int)L.ijk.size()<(size_t)3*nb || deg<1){ g_Z.clear(); return; }
    double mn[3]={1e300,1e300,1e300},mx[3]={-1e300,-1e300,-1e300};
    for(int i=0;i<nb;i++)for(int d=0;d<3;d++){double v=(double)L.ijk[3*i+d]; if(v<mn[d])mn[d]=v; if(v>mx[d])mx[d]=v;}
    double sc[3]; for(int d=0;d<3;d++) sc[d]=(mx[d]>mn[d])?2.0/(mx[d]-mn[d]):0.0;
    std::vector<int> ea,eb,ec;
    for(int td=0;td<=deg;td++)for(int a=0;a<=td;a++)for(int b=0;b<=td-a;b++){ea.push_back(a);eb.push_back(b);ec.push_back(td-a-b);}
    int nm=(int)ea.size(); g_Z.clear();
    for(int m=0;m<nm;m++) for(int comp=0;comp<3;comp++){
        std::vector<double> v(n,0.0);
        for(int i=0;i<nb;i++){ double cx=((double)L.ijk[3*i]-mn[0])*sc[0]-1,cy=((double)L.ijk[3*i+1]-mn[1])*sc[1]-1,cz=((double)L.ijk[3*i+2]-mn[2])*sc[2]-1;
            double val=1; for(int e=0;e<ea[m];e++)val*=cx; for(int e=0;e<eb[m];e++)val*=cy; for(int e=0;e<ec[m];e++)val*=cz;
            v[3*i+comp]=val; }
        for(auto&q:g_Z){ double d=0; for(int i=0;i<n;i++)d+=v[i]*q[i]; for(int i=0;i<n;i++)v[i]-=d*q[i]; }
        double nr=0; for(int i=0;i<n;i++)nr+=v[i]*v[i]; nr=std::sqrt(nr);
        if(nr>1e-8){ for(int i=0;i<n;i++)v[i]/=nr; g_Z.push_back(std::move(v)); } }
}
static void deflate_rbm(int n,double*v){ for(auto&z:g_Z){ double d=0; for(int i=0;i<n;i++) d+=z[i]*v[i];
                                                          for(int i=0;i<n;i++) v[i]-=d*z[i]; } }
// Fine-smoother SpMV dispatch: TT 8-chip bf16x3 (lv==0, if hooked & size matches) > bf16 emulation > exact double.
// The hook falling back on any nonzero return keeps the solve correct even if a TT apply fails mid-run.
static inline void smoo_spmv(const Level&L,const double*x,double*y,int lv,bool emu){
    if(g_tt_fine_spmv && lv==0 && (long)L.B.nb*3==g_tt_fine_n){ if(g_tt_fine_spmv(x,y)==0) return; }
    if(emu) bspmv_emu(L.B,x,y); else bspmv(L.B,x,y);
    if(g_emu_mbits>0 && lv==0){ int n=L.B.nb*3; for(int i=0;i<n;i++) y[i]=round_mbits(y[i],g_emu_mbits); }  // probe fine-SpMV precision
    if(g_emu_abserr>0.0 && lv==0){   // emulate bf16-PRODUCT absolute error: DETERMINISTIC (fixed per-DOF sign so the
        const BCSR&B=L.B;            // preconditioner is a fixed function of x -> PCG-consistent, faithful to real TT)
        auto sgn=[](uint64_t i){ i=(i^0x9E3779B97F4A7C15ULL)*0xBF58476D1CE4E5B9ULL; i^=i>>27; return (i&1)?1.0:-1.0; };
        for(int I=0;I<B.nb;I++){ double t0=0,t1=0,t2=0;
            for(long b=B.ptr[I];b<B.ptr[I+1];b++){ const double*m=&B.val[(size_t)b*9]; const double*xx=&x[3*B.col[b]];
                double a0=std::fabs(xx[0]),a1=std::fabs(xx[1]),a2=std::fabs(xx[2]);
                t0+=std::fabs(m[0])*a0+std::fabs(m[1])*a1+std::fabs(m[2])*a2;
                t1+=std::fabs(m[3])*a0+std::fabs(m[4])*a1+std::fabs(m[5])*a2;
                t2+=std::fabs(m[6])*a0+std::fabs(m[7])*a1+std::fabs(m[8])*a2; }
            y[3*I]+=g_emu_abserr*t0*sgn(3*I); y[3*I+1]+=g_emu_abserr*t1*sgn(3*I+1); y[3*I+2]+=g_emu_abserr*t2*sgn(3*I+2); }
    }
}

/* 4th-kind Chebyshev block-Jacobi smoother. NOTE: float (mixed) SpMV was tried and REJECTED — it degrades
   convergence 15->28 iters on this near-singular operator (soft modes need double even in the preconditioner),
   a net loss vs double. Keep double. */
static void cheb4(const Level&L,int deg,const double*rhs,double*x,WS&w,int lv){
    int n=L.B.nb*3; bool emu = g_emu_bf16 && lv<=g_emu_bf16_maxlv;   // bf16 only on the requested (fine) levels
    smoo_spmv(L,x,w.Ad.data(),lv,emu);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;i++){ w.r[i]=rhs[i]-w.Ad[i]; w.d[i]=0; }
    for(int it=1;it<=deg;it++){ double c=(double)(2*it-3)/(double)(2*it+1); double a=(double)(4*(2*it-1))/((double)(2*it+1)*L.rho);
        bj_apply(L.Dinv,w.r.data(),w.z.data(),n);
        if(emu && g_emu_mode==1) bf16vec(w.z.data(),n);  // bf16 block-Jacobi (single-pass mode only)
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++){ w.d[i]=c*w.d[i]+a*w.z[i]; x[i]+=w.d[i]; }
        if(it<deg){ smoo_spmv(L,w.d.data(),w.Ad.data(),lv,emu);
            #pragma omp parallel for schedule(static)
            for(int i=0;i<n;i++) w.r[i]-=w.Ad[i]; } }
}

// Solver state lives in a GmgCtx, one per ccx_gmg_solve_mem() call -> the solver is REENTRANT; nothing persists
// across calls and the ctx frees itself on return. Cycle defaults are tuned for the bandwidth-bound regular-voxel
// solve: the fine SpMV re-streams the multi-GB matrix from DRAM every apply, so minimizing SpMV COUNT wins.
// DEG=2 V-cycle (GAMMA=1) needs more PCG iters than DEG=4 W-cycle but each iter is far cheaper -> ~20% faster e2e.
// Env-overridable (GMG_DEG/NPRE/NPOST/GAMMA); the acceptance gate falls back to a direct solve if it won't converge.
struct GmgCtx {
    std::vector<Level> LV;
    std::vector<WS> WSP;
    std::vector<double> Cfac;            // dense Cholesky factor of the coarsest level (lower)
    int Cn = 0;                          // coarsest size
    int DEG = 2, NPRE = 1, NPOST = 1, GAMMA = 1;
    int coarse_fail = 0;                 // set if the coarse dpotrs reports info != 0 -> reject the solve
};

static void vcycle(GmgCtx&g, int lv, const double*rhs, double*x){
    if(lv==(int)g.LV.size()-1){ for(int i=0;i<g.Cn;i++) x[i]=rhs[i]; int nrhs=1,info=0; char uplo='L';
        dpotrs_(&uplo,&g.Cn,&nrhs,g.Cfac.data(),&g.Cn,x,&g.Cn,&info); if(info!=0) g.coarse_fail=1; return; }
    Level&L=g.LV[lv]; WS&w=g.WSP[lv]; int n=L.A.nr;
    for(int s=0;s<g.NPRE;s++) cheb4(L,g.DEG,rhs,x,w,lv);
    bspmv(L.B,x,w.Ad.data()); for(int i=0;i<n;i++) w.res[i]=rhs[i]-w.Ad[i];
    int ncoarse=L.P.nc; spmv(L.Pt,w.res.data(),w.rc.data());
    for(int i=0;i<ncoarse;i++) w.ec[i]=0;
    for(int gi=0;gi<g.GAMMA;gi++) vcycle(g,lv+1,w.rc.data(),w.ec.data());
    spmv(L.P,w.ec.data(),w.Ad.data()); for(int i=0;i<n;i++) x[i]+=w.Ad[i];
    for(int s=0;s<g.NPOST;s++) cheb4(L,g.DEG,rhs,x,w,lv);
}
// Fine-level operator dump for the TT-GMG port (env GMG_DUMP_FINE=path). Binary layout (all little-endian):
//   int64 nb, int64 nblk, int64 nx, int64 ny, int64 nz
//   int64 ptr[nb+1]; int32 col[nblk]; double val[nblk*9]   (BCSR: block (I,J) = val[9*slot + 3r + c])
//   int64 ijk[nb*3]                                          (lattice coord of each block-row, stencil structure)
//   double xref[3*nb]; double yref[3*nb]                     (reference SpMV y = A*x for bit-correctness on TT)
static void dump_fine(const char*path,const BCSR&B,const std::vector<long>&ijk,const double*bp,int n,long nx,long ny,long nz){
    FILE*f=fopen(path,"wb"); if(!f){ fprintf(stderr,"[gmg] GMG_DUMP_FINE: cannot open %s\n",path); return; }
    int64_t nb=B.nb, nblk=B.ptr.back(), h[5]={nb,nblk,nx,ny,nz};
    fwrite(h,8,5,f);
    fwrite(B.ptr.data(),8,(size_t)nb+1,f);
    fwrite(B.col.data(),4,(size_t)nblk,f);
    fwrite(B.val.data(),8,(size_t)nblk*9,f);
    fwrite(ijk.data(),8,(size_t)nb*3,f);
    std::vector<double> x(n),y(n); for(int i=0;i<n;i++) x[i]=bp[i]; bspmv(B,x.data(),y.data());
    fwrite(x.data(),8,(size_t)n,f); fwrite(y.data(),8,(size_t)n,f);
    fclose(f);
    fprintf(stderr,"[gmg] GMG_DUMP_FINE: wrote %s (nb=%lld nblk=%lld lattice=%lldx%lldx%lld)\n",
            path,(long long)nb,(long long)nblk,(long long)nx,(long long)ny,(long long)nz);
}
} // namespace

// IN-MEMORY core: caller supplies co[3*nk] (node coords) + na[mt*nk] (CCX nactdof, 1-based eqn|0). No file.
// This is built directly from CalculiX's in-memory arrays at solve time -> no coordmap file needed.
extern "C" int ccx_gmg_solve_mem(int n, const long* cs, const int* ri, const double* va, double* b,
                                 const double* co, const int* na, long nk, int mt,
                                 int maxit, double tol, int verbose)
{
    auto T0=clk::now();
    GmgCtx ctx;   /* solve-local state (reentrant; frees on return). Same-named refs keep the body unchanged. */
    auto&LV=ctx.LV; auto&WSP=ctx.WSP; auto&Cfac=ctx.Cfac; int&Cn=ctx.Cn;
    int&DEG=ctx.DEG; int&NPRE=ctx.NPRE; int&NPOST=ctx.NPOST; int&GAMMA=ctx.GAMMA;
    /* env knobs, clamped to sane upper bounds: a typo'd/huge value (or atoi UB) must not explode runtime --
       e.g. a large W-cycle GAMMA is exponential per V-cycle -- and must not break the round-trip budget. */
    const char*ev; long lv;
    if((ev=getenv("GMG_DEG"))   && (lv=strtol(ev,NULL,10))>0) DEG  =(int)(lv<8   ?lv:8);
    if((ev=getenv("GMG_NPRE"))  && (lv=strtol(ev,NULL,10))>0) NPRE =(int)(lv<8   ?lv:8);
    if((ev=getenv("GMG_NPOST")) && (lv=strtol(ev,NULL,10))>0) NPOST=(int)(lv<8   ?lv:8);
    if((ev=getenv("GMG_GAMMA")) && (lv=strtol(ev,NULL,10))>0) GAMMA=(int)(lv<4   ?lv:4);
    if((ev=getenv("GMG_MAXIT")) && (lv=strtol(ev,NULL,10))>0) maxit=(int)(lv<100000?lv:100000);
    if((ev=getenv("GMG_TOL"))   && atof(ev)>0) tol=atof(ev);
    g_emu_bf16 = (getenv("GMG_EMU_BF16") && atoi(getenv("GMG_EMU_BF16"))>0) ? 1 : 0;   // TT-precision probe
    g_emu_bf16_maxlv = 1000;
    if((ev=getenv("GMG_EMU_BF16_MAXLV")) && (lv=strtol(ev,NULL,10))>=0) g_emu_bf16_maxlv=(int)(lv<1000?lv:1000);
    g_emu_mode = 1;
    if((ev=getenv("GMG_EMU_MODE")) && (lv=strtol(ev,NULL,10))>0) g_emu_mode=(int)lv;
    if(g_emu_bf16 && verbose) fprintf(stderr,"[gmg] TT-PRECISION EMULATION ON: mode=%d (1=bf16x1 3=bf16x3 32=fp32) smoother on levels<=%d (outer PCG + coarser stay fp64)\n",g_emu_mode,g_emu_bf16_maxlv);
    if(mt<3 || mt>64 || nk<=0 || maxit<=0 || !(tol>0)) return 1;   // invalid inputs -> direct fallback
    int gmg_amb =
#ifdef _OPENMP
        omp_get_max_threads();
#else
        1;
#endif
    int gmg_cap = gmg_perf_core_default(gmg_amb);
    if((ev=getenv("GMG_THREADS")) && (lv=strtol(ev,NULL,10))>0) gmg_cap=(int)(lv<gmg_amb?lv:gmg_amb); /* cap at ambient -> no oversubscribe */
    OmpThreadCap _gmg_omp_cap(gmg_cap);
    if(verbose) fprintf(stderr,"[gmg] solve threads cap=%d (ambient=%d)\n", gmg_cap, gmg_amb);
  try {
    // disp columns -> eqnode/eqcomp
    std::vector<long> cnt(mt,0);
    for(long nd=0;nd<nk;nd++) for(int j=0;j<mt;j++){int e=na[mt*nd+j]; if(e>=1&&e<=n) cnt[j]++;}
    std::vector<int> ordr(mt); for(int j=0;j<mt;j++) ordr[j]=j;
    std::sort(ordr.begin(),ordr.end(),[&](int a,int c){return cnt[a]>cnt[c];});
    std::vector<int> disp(ordr.begin(),ordr.begin()+3); std::sort(disp.begin(),disp.end());
    std::vector<long> eqnode(n,-1); std::vector<int> eqcomp(n,-1);
    for(long nd=0;nd<nk;nd++) for(int ci=0;ci<3;ci++){int j=disp[ci]; int e=na[mt*nd+j]; if(e>=1&&e<=n){eqnode[e-1]=nd;eqcomp[e-1]=ci;}}
    std::vector<int> dofcount(nk,0); for(int e=0;e<n;e++){long nd=eqnode[e]; if(nd>=0) dofcount[nd]++;}
    std::vector<int> cnode(nk,-1); int nb=0; for(long nd=0;nd<nk;nd++) if(dofcount[nd]==3) cnode[nd]=nb++;
    if((long)nb*3!=n){ if(verbose)fprintf(stderr,"[gmg] not clean 3-dof blocks: 3nb=%d n=%d -> fallback\n",3*nb,n); return 2; }
    std::vector<int> perm(n); std::vector<long> blk_node(nb);
    for(int e=0;e<n;e++){long nd=eqnode[e];int c=eqcomp[e]; perm[e]=3*cnode[nd]+c; blk_node[cnode[nd]]=nd;}

    // ---- AUTO grid detection: per-axis pitch + regularity gate (no hardcoded pitch) ----
    // Reject non-finite coordinates up front: a NaN/Inf (e.g. from a degenerate input) would break std::sort's
    // ordering and make llround() implementation-defined, and NaN compares false so the regularity gate could
    // pass on garbage. Fall back to the direct solve instead.
    double mn[3]={1e300,1e300,1e300}, mx[3]={-1e300,-1e300,-1e300};
    for(int bb=0;bb<nb;bb++){ long nd=blk_node[bb]; for(int a=0;a<3;a++){ double q=co[3*nd+a];
        if(!std::isfinite(q)){ if(verbose) fprintf(stderr,"[gmg] non-finite node coordinate -> direct fallback\n"); return 3; }
        if(q<mn[a])mn[a]=q; if(q>mx[a])mx[a]=q; } }
    /* Coords are finite, but the EXTENT mx-mn can still overflow to Inf (e.g. +/-1e308), which would feed Inf
       into (co-mn)/h and make the subsequent llround() implementation-defined. Reject up front. */
    for(int a=0;a<3;a++) if(!std::isfinite(mx[a]-mn[a])){ if(verbose) fprintf(stderr,"[gmg] non-finite coordinate extent -> direct fallback\n"); return 3; }
    double h[3];
    for(int a=0;a<3;a++){
        std::vector<double> v(nb); for(int bb=0;bb<nb;bb++) v[bb]=co[3*blk_node[bb]+a];
        std::sort(v.begin(),v.end());
        double hmin=1e300; for(int i=1;i<nb;i++){ double d=v[i]-v[i-1]; if(d>1e-7 && d<hmin) hmin=d; }
        h[a]=(hmin<1e300)?hmin:1.0;   // degenerate single layer -> pitch 1
    }
    /* Guard the cell-count span before ANY llround: even with a finite extent a tiny pitch h can make
       (co-mn)/h overflow to Inf (-> llround implementation-defined). Reject if non-finite/huge. (The strict
       <1e5/axis key bound is still enforced below; 1e8 here only keeps llround well-defined.) */
    for(int a=0;a<3;a++){ double span=(mx[a]-mn[a])/h[a];
        if(!std::isfinite(span) || span<0.0 || span>1e8){ if(verbose) fprintf(stderr,"[gmg] lattice span non-finite/too large -> direct fallback\n"); return 3; } }
    long offlat=0; std::vector<long> ijk0(3*nb);
    for(int bb=0;bb<nb;bb++){ long nd=blk_node[bb]; int bad=0;
        for(int a=0;a<3;a++){ double r=(co[3*nd+a]-mn[a])/h[a]; long ri=llround(r); ijk0[3*bb+a]=ri;
            if(std::fabs(r-(double)ri)>0.05) bad=1; }
        offlat+=bad; }
    double regfrac=1.0-(double)offlat/(double)nb;
    long nx=llround((mx[0]-mn[0])/h[0])+1, ny=llround((mx[1]-mn[1])/h[1])+1, nz=llround((mx[2]-mn[2])/h[2])+1;
    if(verbose) fprintf(stderr,"[gmg] AUTO grid: pitch=(%.4g,%.4g,%.4g) lattice=%ldx%ldx%ld occ=%.2f%% regular=%.3f%%\n",
                        h[0],h[1],h[2],nx,ny,nz,100.0*nb/((double)nx*(double)ny*(double)nz),100.0*regfrac);
    if(regfrac<0.999){ if(verbose) fprintf(stderr,"[gmg] NOT a regular grid (regular %.3f%% < 99.9%%) -> fallback\n",100.0*regfrac); return 3; }
    /* build_P keys coarse lattice nodes as (px*1e5+py)*1e5+pz; px can reach (nx-1 rounded even)+2 == nx+1, so
       each axis index must stay < 1e5 to be collision-free -> require nx,ny,nz <= 99998 (< 10 m at 0.1 mm
       pitch). Beyond that, fall back rather than risk a hash clash. (A real mesh is orders of magnitude smaller.) */
    if(nx>=99999||ny>=99999||nz>=99999){ if(verbose) fprintf(stderr,"[gmg] lattice %ldx%ldx%ld exceeds build_P key bound (<1e5/axis) -> fallback\n",nx,ny,nz); return 3; }
    /* injective node->voxel check: a real voxel grid maps distinct nodes to distinct lattice cells. If two nodes
       round to the SAME (i,j,k) -- e.g. a sub-pitch lattice whose gaps fell below the detector and collapsed to
       h=1 (all nodes -> one cell) -- the grid is degenerate; fall back rather than build a nonsense hierarchy.
       (Coords are bounded <1e5 by the guard above, so the packed key cannot collide spuriously.) */
    { std::vector<long> keys((size_t)nb);
      for(int bb=0;bb<nb;bb++) keys[bb]=(ijk0[3*bb]*100000L+ijk0[3*bb+1])*100000L+ijk0[3*bb+2];
      std::sort(keys.begin(),keys.end());
      for(int bb=1;bb<nb;bb++) if(keys[bb]==keys[bb-1]){
          if(verbose) fprintf(stderr,"[gmg] duplicate lattice node (sub-pitch/degenerate grid) -> direct fallback\n"); return 3; } }

    // ---- build fine scalar full CRS (node-block order) from lower CSC ----
    std::vector<int> deg(n,0);
    for(int c=0;c<n;c++) for(long p=cs[c];p<cs[c+1];p++){int r=ri[p];int pc=perm[c],pr=perm[r]; deg[pc]++; if(r!=c) deg[pr]++;}
    CSR A; A.nr=A.nc=n; A.ptr.assign(n+1,0); for(int i=0;i<n;i++) A.ptr[i+1]=A.ptr[i]+deg[i];
    A.col.resize(A.ptr.back()); A.val.resize(A.ptr.back());
    std::vector<long> cur(A.ptr.begin(),A.ptr.end()-1);
    for(int c=0;c<n;c++) for(long p=cs[c];p<cs[c+1];p++){int r=ri[p];double v=va[p];int pc=perm[c],pr=perm[r];
        A.col[cur[pc]]=pr;A.val[cur[pc]]=v;cur[pc]++; if(r!=c){A.col[cur[pr]]=pc;A.val[cur[pr]]=v;cur[pr]++;} }
    #pragma omp parallel for schedule(dynamic,2048)
    for(int i=0;i<n;i++){ long s=A.ptr[i],e=A.ptr[i+1]; std::vector<std::pair<int,double>> t(e-s);
        for(long p=s;p<e;p++) t[p-s]={A.col[p],A.val[p]}; std::sort(t.begin(),t.end());
        for(long p=s;p<e;p++){A.col[p]=t[p-s].first;A.val[p]=t[p-s].second;} }
    std::vector<double> bp(n); for(int e=0;e<n;e++) bp[perm[e]]=b[e];
    if(verbose) fprintf(stderr,"[gmg] setup: nb=%d fineCRS nnz=%ld (%.2fs)\n",nb,A.nnz(),secs(T0,clk::now()));

    // ---- hierarchy (Galerkin) ---- (ctx is fresh: LV/WSP/Cfac already empty)
    { Level L; L.A=std::move(A); L.ijk=ijk0; LV.push_back(std::move(L)); }
    while((int)LV[LV.size()-1].A.nr>6000){
        size_t fi=LV.size()-1; int nbf=LV[fi].A.nr/3; std::vector<long> cijk; int nbc;
        CSR P=build_P(LV[fi].ijk,nbf,cijk,nbc); CSR Pt=transpose(P);
        CSR AP=spmm(LV[fi].A,P); CSR Ac=spmm(Pt,AP);
        LV[fi].P=std::move(P); LV[fi].Pt=std::move(Pt);
        Level C; C.A=std::move(Ac); C.ijk=std::move(cijk); LV.push_back(std::move(C));
    }
    for(int lv=0;lv<(int)LV.size()-1;lv++){ LV[lv].B=to_bcsr(LV[lv].A);   // block-3x3 (matrix-stream win)
        if(verbose){ int nn=LV[lv].A.nr; std::vector<double> tv(nn),ya(nn),yb(nn); uint64_t s=12345+lv;
            for(int i=0;i<nn;i++){ s^=s<<13;s^=s>>7;s^=s<<17; tv[i]=(double)((s>>11)&1023)/1024.0-0.5; }
            spmv(LV[lv].A,tv.data(),ya.data()); bspmv(LV[lv].B,tv.data(),yb.data());
            double mx=0,nrm=0; for(int i=0;i<nn;i++){ mx=std::max(mx,std::fabs(ya[i]-yb[i])); nrm=std::max(nrm,std::fabs(ya[i])); }
            fprintf(stderr,"[gmg] BCSR equiv lv%d: blocks=%ld maxdiff=%.2e (rel %.2e)\n",lv,LV[lv].B.ptr.back(),mx,mx/(nrm+1e-300)); }
        block_diag_inv_b(LV[lv].B,LV[lv].Dinv); LV[lv].rho=1.1*lmax(LV[lv].B,LV[lv].Dinv);
        LV[lv].A.col.clear(); LV[lv].A.col.shrink_to_fit(); LV[lv].A.val.clear(); LV[lv].A.val.shrink_to_fit(); } // free scalar (BCSR used in solve)
    Cn=LV.back().A.nr; Cfac.assign((size_t)Cn*Cn,0.0);
    { const CSR&Ac=LV.back().A; for(int i=0;i<Cn;i++) for(long p=Ac.ptr[i];p<Ac.ptr[i+1];p++) Cfac[(size_t)Ac.col[p]*Cn+i]=Ac.val[p]; }
    { char uplo='L'; int info; dpotrf_(&uplo,&Cn,Cfac.data(),&Cn,&info);
      if(info){ fprintf(stderr,"[gmg] coarse dpotrf failed info=%d (coarse not SPD) -> direct fallback\n",info);
                return 5; } }   // b untouched (ctx frees on return)
    WSP.resize(LV.size());
    for(int lv=0;lv<(int)LV.size();lv++){ int nn=LV[lv].A.nr; WS&w=WSP[lv];
        w.r.resize(nn);w.z.resize(nn);w.d.resize(nn);w.Ad.resize(nn);w.res.resize(nn);
        int ncoarse = (lv<(int)LV.size()-1)? LV[lv].P.nc : nn; w.rc.resize(ncoarse); w.ec.resize(ncoarse); }
    if(verbose) fprintf(stderr,"[gmg] hierarchy %zu levels, coarsest n=%d, total setup %.2fs (deg=%d npre=%d npost=%d gamma=%d)\n",
                        LV.size(),Cn,secs(T0,clk::now()),DEG,NPRE,NPOST,GAMMA);

    if((ev=getenv("GMG_DUMP_FINE")) && ev[0]) dump_fine(ev,LV[0].B,LV[0].ijk,bp.data(),n,nx,ny,nz);   // TT-GMG port target

    // ---- PCG with V/W-cycle preconditioner ----
    const BCSR&A0=LV[0].B; int N=n;
    std::vector<double> x(N,0.0),r(N),z(N),p(N),Ap(N);
    for(int i=0;i<N;i++) r[i]=bp[i];
    double res0=std::sqrt(ddot(r.data(),r.data(),N)); if(res0==0) res0=1;
    if(!std::isfinite(res0)){ if(verbose) fprintf(stderr,"[gmg] non-finite ||b|| (overflow) -> direct fallback\n"); return 4; }
    for(int i=0;i<N;i++) z[i]=0; vcycle(ctx,0,r.data(),z.data());
    for(int i=0;i<N;i++) p[i]=z[i]; double rz=ddot(r.data(),z.data(),N);
    auto ts=clk::now(); int iters=maxit; double rel=1;
    for(int it=0;it<maxit;it++){
        bspmv(A0,p.data(),Ap.data());                       // EXACT double outer operator (block-3x3)
        double pAp=ddot(p.data(),Ap.data(),N);
        if(!(pAp>0.0)){ if(verbose) fprintf(stderr,"[gmg] PCG breakdown p*Ap=%.3e\n",pAp); break; } // SPD-indefinite guard
        double al=rz/pAp;
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;i++){ x[i]+=al*p[i]; r[i]-=al*Ap[i]; }
        double rr=std::sqrt(ddot(r.data(),r.data(),N)); rel=rr/res0;
        if(rel<tol){ iters=it+1; break; }
        for(int i=0;i<N;i++) z[i]=0; vcycle(ctx,0,r.data(),z.data());
        double rzn=ddot(r.data(),z.data(),N);
        if(!(rz!=0.0) || !std::isfinite(rzn)){ if(verbose) fprintf(stderr,"[gmg] PCG breakdown rz=%.3e rzn=%.3e\n",rz,rzn); break; }
        double bet=rzn/rz;
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;i++) p[i]=z[i]+bet*p[i];
        rz=rzn;
    }
    // ACCEPTANCE GATE: recompute the TRUE residual ||b-Ax|| (recursive r drifts on this near-singular op;
    // also catches non-convergence at maxit, PCG breakdown, and NaN since !(NaN<tol*2)==true). On failure,
    // leave b UNTOUCHED and return nonzero -> accel_solver falls back to the exact direct solve with the
    // original RHS. (A fast WRONG answer must never be returned as success.)
    bspmv(A0,x.data(),Ap.data());
    double tr=0; for(int i=0;i<N;i++){ double e=bp[i]-Ap[i]; tr+=e*e; }
    double true_rel=std::sqrt(tr)/res0;
    if(verbose) fprintf(stderr,"[gmg] PCG iters=%d rel=%.2e true_rel=%.2e solve=%.2fs total=%.2fs\n",
                        iters,rel,true_rel,secs(ts,clk::now()),secs(T0,clk::now()));
    // SAFETY ceiling decoupled from the perf tol: accept only if the TRUE residual is below BOTH a small
    // multiple of the requested convergence tol AND a hard physical ceiling (1e-2, the acceptable-accuracy floor).
    // Loosening GMG_TOL for speed can therefore never silently lower the correctness bar -- a looser perf
    // tol only triggers the exact-solve fallback sooner; it never lets a worse-than-1% solution pass.
    // (NaN-safe: !(NaN < accept) == true -> reject -> fallback.) Not env-overridable on purpose.
    const double GMG_ACCEPT_MAX = 1e-2;
    double accept = tol*2.0; if(accept > GMG_ACCEPT_MAX) accept = GMG_ACCEPT_MAX;
    if(ctx.coarse_fail || !(true_rel < accept)){
        fprintf(stderr,"[gmg] NOT ACCEPTED (true_rel=%.3e accept=%.1e tol=%.1e iters=%d coarse_fail=%d) -> direct fallback\n",
                true_rel,accept,tol,iters,ctx.coarse_fail);
        return 4;   // b untouched (ctx frees on return)
    }
    for(int ie=0;ie<n;ie++) b[ie]=x[perm[ie]];   // converged -> un-permute solution into b (equation order)
    return 0;
  } catch(...) {   // bad_alloc/etc must not cross the extern "C" boundary (UB) -> clean direct fallback
    fprintf(stderr,"[gmg] exception (likely OOM) -> direct fallback\n");
    return 6;   // b untouched (ctx frees on return)
  }
}

// TT-GMG driver entry: load a GMG_DUMP_FINE dump (fine BCSR + ijk + rhs), rebuild the scalar CSR, run the
// full GMG-PCG. The fine (lv==0) smoother SpMV runs on the 8-chip TT mesh when the caller has set
// g_tt_fine_spmv; coarser levels + the vcycle residual + the outer PCG stay exact fp64. Returns 0 on accept
// (true_rel < 2*tol), writes the solution to out_x (n doubles) and maxU to out_maxu.
extern "C" int ccx_gmg_solve_from_dump(const char* path, int maxit, double tol, int verbose,
                                       double* out_x, double* out_maxu) {
  try {
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"[tt-gmg] cannot open %s\n",path); return 1; }
    int64_t h[5]; if(fread(h,8,5,f)!=5){ fclose(f); return 1; }
    long nb=(long)h[0], nblk=(long)h[1]; int n=(int)(3*nb);
    BCSR B; B.nb=(int)nb; B.ptr.resize(nb+1); B.col.resize(nblk); B.val.resize((size_t)nblk*9);
    std::vector<long> ijk((size_t)nb*3); std::vector<double> bp(n), yref(n);
    if(fread(B.ptr.data(),8,(size_t)nb+1,f)!=(size_t)nb+1 || fread(B.col.data(),4,(size_t)nblk,f)!=(size_t)nblk ||
       fread(B.val.data(),8,(size_t)nblk*9,f)!=(size_t)nblk*9 || fread(ijk.data(),8,(size_t)nb*3,f)!=(size_t)nb*3 ||
       fread(bp.data(),8,(size_t)n,f)!=(size_t)n){ fclose(f); return 1; }
    (void)!fread(yref.data(),8,(size_t)n,f); fclose(f);
    auto T0=clk::now();
    // rebuild scalar CSR A from the block-3x3 BCSR (rows already sorted by block-col in the dump)
    CSR A; A.nr=A.nc=n; A.ptr.assign((size_t)n+1,0);
    for(long I=0;I<nb;I++){ long bn=B.ptr[I+1]-B.ptr[I]; for(int r=0;r<3;r++) A.ptr[3*I+r+1]=3*bn; }
    for(int i=0;i<n;i++) A.ptr[i+1]+=A.ptr[i];
    A.col.resize(A.ptr[n]); A.val.resize(A.ptr[n]);
    #pragma omp parallel for schedule(dynamic,1024)
    for(long I=0;I<nb;I++) for(int r=0;r<3;r++){ long o=A.ptr[3*I+r];
        for(long b=B.ptr[I];b<B.ptr[I+1];b++){ int J=B.col[b]; for(int c=0;c<3;c++){ A.col[o]=3*J+c; A.val[o]=B.val[(size_t)b*9+r*3+c]; o++; } } }
    g_tt_fine_n = n;   // arm the fine-SpMV hook size guard
    GmgCtx ctx; auto&LV=ctx.LV; auto&WSP=ctx.WSP; auto&Cfac=ctx.Cfac; int&Cn=ctx.Cn;
    const char*ev; long lvv;
    if((ev=getenv("GMG_DEG"))  &&(lvv=strtol(ev,NULL,10))>0) ctx.DEG  =(int)(lvv<8?lvv:8);
    if((ev=getenv("GMG_NPRE")) &&(lvv=strtol(ev,NULL,10))>0) ctx.NPRE =(int)(lvv<8?lvv:8);
    if((ev=getenv("GMG_NPOST"))&&(lvv=strtol(ev,NULL,10))>0) ctx.NPOST=(int)(lvv<8?lvv:8);
    if((ev=getenv("GMG_GAMMA"))&&(lvv=strtol(ev,NULL,10))>0) ctx.GAMMA=(int)(lvv<4?lvv:4);
    // emu bracket: exercise the compensated-precision smoother on CPU to find the convergence threshold
    g_emu_bf16 = (getenv("GMG_EMU_BF16") && atoi(getenv("GMG_EMU_BF16"))>0) ? 1 : 0;
    g_emu_bf16_maxlv = 1000; g_emu_mode = 1;
    if((ev=getenv("GMG_EMU_MODE")) && (lvv=strtol(ev,NULL,10))>0) g_emu_mode=(int)lvv;
    g_emu_mbits = (getenv("GMG_EMU_MBITS") && (lvv=strtol(getenv("GMG_EMU_MBITS"),NULL,10))>0) ? (int)lvv : 0;
    g_emu_abserr = getenv("GMG_EMU_ABSERR") ? atof(getenv("GMG_EMU_ABSERR")) : 0.0;
    g_defl_rbm = (getenv("GMG_DEFL_RBM") && atoi(getenv("GMG_DEFL_RBM"))>0) ? 1 : 0;
    g_defl_corr = (getenv("GMG_DEFL_CORR") && atoi(getenv("GMG_DEFL_CORR"))>0) ? 1 : 0;
    g_defl_reg = getenv("GMG_DEFL_REG") ? atof(getenv("GMG_DEFL_REG")) : 0.0;
    { int dd; g_defl_deg = (getenv("GMG_DEFL_DEG") && (dd=atoi(getenv("GMG_DEFL_DEG")))>=1) ? dd : 1; }
    if((g_emu_abserr>0.0||g_defl_rbm) && verbose) fprintf(stderr,"[tt-gmg] abserr=%.3e defl_rbm=%d (TT failure-mode probe)\n",g_emu_abserr,g_defl_rbm);
    if(g_emu_bf16 && verbose) fprintf(stderr,"[tt-gmg] CPU EMU smoother ON mode=%d (2=bf16x2 3=bf16x3 32=fp32)\n",g_emu_mode);
    if(g_emu_mbits && verbose) fprintf(stderr,"[tt-gmg] fine-SpMV output rounded to %d mantissa bits (2^-%d ~ %.1e rel)\n",g_emu_mbits,g_emu_mbits,std::ldexp(1.0,-g_emu_mbits));
    { Level L; L.A=std::move(A); L.ijk=ijk; LV.push_back(std::move(L)); }
    while((int)LV[LV.size()-1].A.nr>6000){
        size_t fi=LV.size()-1; int nbf=LV[fi].A.nr/3; std::vector<long> cijk; int nbc;
        CSR P=build_P(LV[fi].ijk,nbf,cijk,nbc); CSR Pt=transpose(P);
        CSR AP=spmm(LV[fi].A,P); CSR Ac=spmm(Pt,AP);
        LV[fi].P=std::move(P); LV[fi].Pt=std::move(Pt);
        Level C; C.A=std::move(Ac); C.ijk=std::move(cijk); LV.push_back(std::move(C)); }
    for(int lv=0;lv<(int)LV.size()-1;lv++){ LV[lv].B=to_bcsr(LV[lv].A);
        block_diag_inv_b(LV[lv].B,LV[lv].Dinv); LV[lv].rho=1.1*lmax(LV[lv].B,LV[lv].Dinv);
        LV[lv].A.col.clear();LV[lv].A.col.shrink_to_fit();LV[lv].A.val.clear();LV[lv].A.val.shrink_to_fit(); }
    Cn=LV.back().A.nr; Cfac.assign((size_t)Cn*Cn,0.0);
    { const CSR&Ac=LV.back().A; for(int i=0;i<Cn;i++) for(long p=Ac.ptr[i];p<Ac.ptr[i+1];p++) Cfac[(size_t)Ac.col[p]*Cn+i]=Ac.val[p]; }
    { char uplo='L'; int info; dpotrf_(&uplo,&Cn,Cfac.data(),&Cn,&info); if(info){ fprintf(stderr,"[tt-gmg] coarse dpotrf info=%d\n",info); return 5; } }
    WSP.resize(LV.size());
    for(int lv=0;lv<(int)LV.size();lv++){ int nn=LV[lv].A.nr; WS&w=WSP[lv];
        w.r.resize(nn);w.z.resize(nn);w.d.resize(nn);w.Ad.resize(nn);w.res.resize(nn);
        int ncoarse=(lv<(int)LV.size()-1)?LV[lv].P.nc:nn; w.rc.resize(ncoarse); w.ec.resize(ncoarse); }
    if(verbose) fprintf(stderr,"[tt-gmg] setup %.2fs, %zu levels, coarsest=%d, TT_fine=%s\n",
                        secs(T0,clk::now()),LV.size(),Cn,g_tt_fine_spmv?"ON":"off");
    const BCSR&A0=LV[0].B; int N=n;
    std::vector<double> x(N,0.0),r(N),z(N),p(N),Ap(N);
    for(int i=0;i<N;i++) r[i]=bp[i];
    double res0=std::sqrt(ddot(r.data(),r.data(),N)); if(res0==0) res0=1;
    // ---- Deflated PCG (Saad DCG). The near-null space (6 RBMs) is solved EXACTLY in fp64; the CG runs in the
    // A-orthogonal complement via the projected operator PA = A - AZ E^-1 (AZ)^T, so the (bf16/TT) smoother only ever
    // sees complement residuals -> no extreme cancellation -> bf16x3 is accurate there. E = Z^T A Z (6x6, Cholesky).
    std::vector<std::vector<double>> DZ, DAZ; std::vector<double> DE; bool defl=(g_defl_corr!=0); int K6=0;
    if(defl){ build_defl(LV[0],g_defl_deg); DZ=g_Z; K6=(int)DZ.size();
        if(K6<1){ defl=false; }
        else { DAZ.assign(K6,std::vector<double>(N)); DE.assign((size_t)K6*K6,0.0);
        for(int a=0;a<K6;a++) bspmv(A0,DZ[a].data(),DAZ[a].data());
        for(int a=0;a<K6;a++) for(int b=0;b<K6;b++) DE[(size_t)a*K6+b]=ddot(DZ[a].data(),DAZ[b].data(),N);
        double md=0; for(int a=0;a<K6;a++) md=std::max(md,std::fabs(DE[(size_t)a*K6+a]));
        if(g_defl_reg>0.0) for(int a=0;a<K6;a++) DE[(size_t)a*K6+a]+=g_defl_reg*md;
        char u='L'; int kk=K6,info; dpotrf_(&u,&kk,DE.data(),&kk,&info);
        if(info){ if(verbose) fprintf(stderr,"[tt-gmg] E dpotrf info=%d -> defl off\n",info); defl=false; }
        else if(verbose) fprintf(stderr,"[tt-gmg] deflated PCG on: k=%d (deg=%d) E maxdiag=%.3e reg=%.3e\n",K6,g_defl_deg,md,g_defl_reg); } }
    auto Esolve=[&](double*c){ char u='L'; int kk=K6,one=1,info; dpotrs_(&u,&kk,&one,DE.data(),&kk,c,&kk,&info); (void)info; };
    auto Papply=[&](double*v){ if(!defl) return; std::vector<double> c(K6);   // v <- (I - AZ E^-1 Z^T) v
        for(int a=0;a<K6;a++) c[a]=ddot(DZ[a].data(),v,N); Esolve(c.data());
        for(int a=0;a<K6;a++){ double ca=c[a]; const double*az=DAZ[a].data(); for(int i=0;i<N;i++) v[i]-=ca*az[i]; } };
    auto addZ=[&](const double*rr,double*xx){ std::vector<double> c(K6);      // xx += Z E^-1 Z^T rr  (near-null solve)
        for(int a=0;a<K6;a++) c[a]=ddot(DZ[a].data(),rr,N); Esolve(c.data());
        for(int a=0;a<K6;a++){ double ca=c[a]; const double*za=DZ[a].data(); for(int i=0;i<N;i++) xx[i]+=ca*za[i]; } };
    if(defl){ addZ(bp.data(),x.data()); bspmv(A0,x.data(),Ap.data());         // x0 = Z E^-1 Z^T b  -> r0 deflated
        for(int i=0;i<N;i++) r[i]=bp[i]-Ap[i]; Papply(r.data()); }
    for(int i=0;i<N;i++) z[i]=0; vcycle(ctx,0,r.data(),z.data());
    for(int i=0;i<N;i++) p[i]=z[i]; double rz=ddot(r.data(),z.data(),N);
    auto ts=clk::now(); int iters=maxit; double rel=1;
    for(int it=0;it<maxit;it++){
        bspmv(A0,p.data(),Ap.data()); Papply(Ap.data());     // DEFLATED operator P*A*p (keeps CG in the complement)
        double pAp=ddot(p.data(),Ap.data(),N); if(!(pAp>0.0)) break;
        double al=rz/pAp;
        for(int i=0;i<N;i++){ x[i]+=al*p[i]; r[i]-=al*Ap[i]; }
        rel=std::sqrt(ddot(r.data(),r.data(),N))/res0; if(rel<tol){ iters=it+1; break; }
        if(verbose && (it<6||it%5==0)) fprintf(stderr,"[tt-gmg] it=%d rel=%.3e (%.2fs elapsed)\n",it,rel,secs(ts,clk::now()));
        for(int i=0;i<N;i++) z[i]=0; vcycle(ctx,0,r.data(),z.data());
        double rzn=ddot(r.data(),z.data(),N); if(!(rz!=0.0)||!std::isfinite(rzn)) break;
        double bet=rzn/rz; for(int i=0;i<N;i++) p[i]=z[i]+bet*p[i]; rz=rzn; }
    if(defl){ bspmv(A0,x.data(),Ap.data()); std::vector<double> rr(N);        // final exact near-null correction
        for(int i=0;i<N;i++) rr[i]=bp[i]-Ap[i]; addZ(rr.data(),x.data()); }
    bspmv(A0,x.data(),Ap.data());
    double tr=0; for(int i=0;i<N;i++){ double e=bp[i]-Ap[i]; tr+=e*e; }
    double true_rel=std::sqrt(tr)/res0, mx=0; for(int i=0;i<N;i++) mx=std::max(mx,std::fabs(x[i]));
    if(out_maxu) *out_maxu=mx; if(out_x) for(int i=0;i<N;i++) out_x[i]=x[i];
    if(verbose) fprintf(stderr,"[tt-gmg] PCG iters=%d rel=%.2e true_rel=%.2e maxU=%.7f solve=%.2fs total=%.2fs\n",
                        iters,rel,true_rel,mx,secs(ts,clk::now()),secs(T0,clk::now()));
    return (ctx.coarse_fail || !(true_rel < 2.0*tol)) ? 4 : 0;
  } catch(...) { fprintf(stderr,"[tt-gmg] exception -> abort\n"); return 6; }
}

// FILE wrapper (backward compat): read the coord/DOF-map file, then call the in-memory core.
extern "C" int ccx_gmg_solve(int n, const long* cs, const int* ri, const double* va,
                             double* b, const char* coordmap, int maxit, double tol, int verbose)
{
    FILE*g=fopen(coordmap,"rb"); if(!g){ if(verbose)fprintf(stderr,"[gmg] cannot open coordmap %s\n",coordmap); return 1; }
    int64_t nk64,mt64; if(fread(&nk64,8,1,g)!=1||fread(&mt64,8,1,g)!=1){fclose(g);return 1;}
    long nk=(long)nk64; int mt=(int)mt64;
    if(mt<3||mt>64||nk<=0||nk>1000000000L){fclose(g);return 1;}   // validate+bound (untrusted file) before alloc, so (size_t)3*nk/mt*nk can't wrap
    int rc=1;
    try {                                            // bad_alloc from the co/na vectors must not cross extern "C"
        std::vector<double> co((size_t)3*nk); std::vector<int> na((size_t)mt*nk);
        if(fread(co.data(),8,(size_t)3*nk,g)==(size_t)3*nk && fread(na.data(),4,(size_t)mt*nk,g)==(size_t)mt*nk){
            fclose(g); g=NULL;
            return ccx_gmg_solve_mem(n,cs,ri,va,b,co.data(),na.data(),nk,mt,maxit,tol,verbose);
        }
    } catch(...) { rc=6; }
    if(g) fclose(g);
    return rc;
}
