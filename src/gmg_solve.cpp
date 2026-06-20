// gmg_solve.cpp — geometric multigrid solver for CCX, callable from accel_solver.c. Tooling / APHYSICAL.
// The row236-class problem is a homogeneous regular 0.1mm Cartesian voxel grid C3D8 (3 DOF/node).
// Validated in Python (scripts/gmg_proto.py) and C (_amg/gmg_test.cpp): support-closed trilinear P +
// Galerkin A_c=P^T A P + block-Jacobi 4th-kind Chebyshev + PCG (W-cycle) => 15 iters, maxU exact.
// Entry: ccx_gmg_solve(n, cs, ri, va, b, coordmap_path, ...). Returns 0 on success (solution into b).
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
// GMG solve thread policy (goal §20.8/§20.0): on this gather-latency-bound kernel, using all logical cores is
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

struct CSR { int nr=0,nc=0; std::vector<long> ptr; std::vector<int> col; std::vector<double> val;
             long nnz() const { return ptr.empty()?0:ptr.back(); } };

static void spmv(const CSR&A,const double*x,double*y){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<A.nr;i++){ double s=0; for(long p=A.ptr[i];p<A.ptr[i+1];p++) s+=A.val[p]*x[A.col[p]]; y[i]=s; }
}
// block-3x3 CSR. Phase-0 (iter16) showed the SpMV is matrix-STREAM-bound (val+col), x is SLC-resident.
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
// Determinism (M7): the OpenMP reduction combines per-thread partial sums in an unspecified order, so this
// dot product is NOT bit-identical run-to-run even at a fixed thread count. That perturbs the PCG path at the
// ~1e-16 level; the iteration still converges to the same true residual, so maxU is reproducible to solver
// tolerance -- far inside the §18 ±1% bound. Bit-exact reproducibility would need an ordered tree reduction.
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

/* 4th-kind Chebyshev block-Jacobi smoother. NOTE: float (mixed) SpMV was tried and REJECTED — it degrades
   convergence 15->28 iters on this near-singular operator (soft modes need double even in the preconditioner),
   a net loss vs double. Keep double. */
static void cheb4(const Level&L,int deg,const double*rhs,double*x,WS&w){
    int n=L.B.nb*3; bspmv(L.B,x,w.Ad.data());
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;i++){ w.r[i]=rhs[i]-w.Ad[i]; w.d[i]=0; }
    for(int it=1;it<=deg;it++){ double c=(double)(2*it-3)/(double)(2*it+1); double a=(double)(4*(2*it-1))/((double)(2*it+1)*L.rho);
        bj_apply(L.Dinv,w.r.data(),w.z.data(),n);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++){ w.d[i]=c*w.d[i]+a*w.z[i]; x[i]+=w.d[i]; }
        if(it<deg){ bspmv(L.B,w.d.data(),w.Ad.data());
            #pragma omp parallel for schedule(static)
            for(int i=0;i<n;i++) w.r[i]-=w.Ad[i]; } }
}

// Solver state lives in a GmgCtx, one per ccx_gmg_solve_mem() call -> the solver is REENTRANT; nothing persists
// across calls and the ctx frees itself on return. Cycle defaults are tuned for the bandwidth-bound regular-voxel
// solve (row236): the fine SpMV re-streams the 2.4GB matrix from DRAM every apply, so minimizing SpMV COUNT wins.
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
    for(int s=0;s<g.NPRE;s++) cheb4(L,g.DEG,rhs,x,w);
    bspmv(L.B,x,w.Ad.data()); for(int i=0;i<n;i++) w.res[i]=rhs[i]-w.Ad[i];
    int ncoarse=L.P.nc; spmv(L.Pt,w.res.data(),w.rc.data());
    for(int i=0;i<ncoarse;i++) w.ec[i]=0;
    for(int gi=0;gi<g.GAMMA;gi++) vcycle(g,lv+1,w.rc.data(),w.ec.data());
    spmv(L.P,w.ec.data(),w.Ad.data()); for(int i=0;i<n;i++) x[i]+=w.Ad[i];
    for(int s=0;s<g.NPOST;s++) cheb4(L,g.DEG,rhs,x,w);
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
    const char*ev;
    if((ev=getenv("GMG_DEG"))   && atoi(ev)>0) DEG=atoi(ev);
    if((ev=getenv("GMG_NPRE"))  && atoi(ev)>0) NPRE=atoi(ev);
    if((ev=getenv("GMG_NPOST")) && atoi(ev)>0) NPOST=atoi(ev);
    if((ev=getenv("GMG_GAMMA")) && atoi(ev)>0) GAMMA=atoi(ev);
    if((ev=getenv("GMG_MAXIT")) && atoi(ev)>0) maxit=atoi(ev);
    if((ev=getenv("GMG_TOL"))   && atof(ev)>0) tol=atof(ev);
    if(mt<3 || mt>64 || nk<=0 || maxit<=0 || !(tol>0)) return 1;   // invalid inputs -> direct fallback
    int gmg_amb =
#ifdef _OPENMP
        omp_get_max_threads();
#else
        1;
#endif
    int gmg_cap = gmg_perf_core_default(gmg_amb);
    if((ev=getenv("GMG_THREADS")) && atoi(ev)>0) gmg_cap=atoi(ev);
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
    double mn[3]={1e300,1e300,1e300}, mx[3]={-1e300,-1e300,-1e300};
    for(int bb=0;bb<nb;bb++){ long nd=blk_node[bb]; for(int a=0;a<3;a++){ double q=co[3*nd+a]; if(q<mn[a])mn[a]=q; if(q>mx[a])mx[a]=q; } }
    double h[3];
    for(int a=0;a<3;a++){
        std::vector<double> v(nb); for(int bb=0;bb<nb;bb++) v[bb]=co[3*blk_node[bb]+a];
        std::sort(v.begin(),v.end());
        double hmin=1e300; for(int i=1;i<nb;i++){ double d=v[i]-v[i-1]; if(d>1e-7 && d<hmin) hmin=d; }
        h[a]=(hmin<1e300)?hmin:1.0;   // degenerate single layer -> pitch 1
    }
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
    // build_P keys coarse lattice nodes as (px*100000+py)*100000+pz; that is collision-free only while every
    // axis index < 100000 (i.e. < 10 m at 0.1 mm pitch). Beyond that, fall back rather than risk a hash clash.
    if(nx>=100000||ny>=100000||nz>=100000){ if(verbose) fprintf(stderr,"[gmg] lattice %ldx%ldx%ld exceeds build_P hash bound (1e5/axis) -> fallback\n",nx,ny,nz); return 3; }

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
    for(int lv=0;lv<(int)LV.size()-1;lv++){ LV[lv].B=to_bcsr(LV[lv].A);   // block-3x3 (matrix-stream win, iter16)
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

    // ---- PCG with V/W-cycle preconditioner ----
    const BCSR&A0=LV[0].B; int N=n;
    std::vector<double> x(N,0.0),r(N),z(N),p(N),Ap(N);
    for(int i=0;i<N;i++) r[i]=bp[i];
    double res0=std::sqrt(ddot(r.data(),r.data(),N)); if(res0==0) res0=1;
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
    // multiple of the requested convergence tol AND a hard physical ceiling (1e-2, the §18 accept floor).
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

// FILE wrapper (backward compat): read the coord/DOF-map file, then call the in-memory core.
extern "C" int ccx_gmg_solve(int n, const long* cs, const int* ri, const double* va,
                             double* b, const char* coordmap, int maxit, double tol, int verbose)
{
    FILE*g=fopen(coordmap,"rb"); if(!g){ if(verbose)fprintf(stderr,"[gmg] cannot open coordmap %s\n",coordmap); return 1; }
    int64_t nk64,mt64; if(fread(&nk64,8,1,g)!=1||fread(&mt64,8,1,g)!=1){fclose(g);return 1;}
    long nk=(long)nk64; int mt=(int)mt64; if(mt<3||mt>64||nk<=0){fclose(g);return 1;}   // validate before alloc
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
