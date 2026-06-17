// gmg_driver.cpp — drive the PRODUCTION gmg_solve (ccx_gmg_solve_mem) on a dumped matrix+coordmap, time the
// full solve, print maxU + L2 checksum + timing as a parseable SUMMARY. Built BOTH ways by gmg_bench.sh
// (Apple Accelerate vs portable OpenBLAS) to compare correctness + speed. Tooling / APHYSICAL.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

extern "C" int ccx_gmg_solve_mem(int n, const long* cs, const int* ri, const double* va, double* b,
                                 const double* co, const int* na, long nk, int mt,
                                 int maxit, double tol, int verbose);

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s m2.mat coordmap.bin [tol]\n", argv[0]); return 1; }
    double tol = (argc > 3) ? atof(argv[3]) : 1e-4;
    const char* variant =
#if defined(__APPLE__) && !defined(CCX_GMG_NOACCEL)
        "ACCEL(Apple-Accelerate-LAPACK)";
#else
        "REFERENCE(portable-LAPACK)";
#endif
    // ---- load lower-CSC matrix + RHS ----
    FILE* f = fopen(argv[1], "rb"); if (!f) { perror("mat"); return 1; }
    int64_t n64, nnz64; if (fread(&n64,8,1,f)!=1 || fread(&nnz64,8,1,f)!=1) return 1;
    int n = (int)n64; long nnz = (long)nnz64;
    std::vector<long> cs(n+1); std::vector<int> ri(nnz); std::vector<double> va(nnz), b(n);
    if (fread(cs.data(),8,n+1,f)!=(size_t)(n+1) || fread(ri.data(),4,nnz,f)!=(size_t)nnz
        || fread(va.data(),8,nnz,f)!=(size_t)nnz || fread(b.data(),8,n,f)!=(size_t)n) return 1;
    fclose(f);
    // ---- coordmap ----
    FILE* g = fopen(argv[2], "rb"); if (!g) { perror("coordmap"); return 1; }
    int64_t nk64, mt64; if (fread(&nk64,8,1,g)!=1 || fread(&mt64,8,1,g)!=1) return 1;
    long nk = (long)nk64; int mt = (int)mt64;
    std::vector<double> co(3*nk); std::vector<int> na((size_t)mt*nk);
    if (fread(co.data(),8,3*nk,g)!=(size_t)(3*nk) || fread(na.data(),4,(size_t)mt*nk,g)!=(size_t)mt*nk) return 1;
    fclose(g);

    // ---- solve (production gmg_solve), timed ----
    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = ccx_gmg_solve_mem(n, cs.data(), ri.data(), va.data(), b.data(), co.data(), na.data(),
                               nk, mt, 80, tol, /*verbose=*/0);
    double solve_s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();

    // ---- maxU per node (b is now the solution in equation order) ----
    std::vector<long> cnt(mt,0);
    for (long nd=0; nd<nk; nd++) for (int j=0;j<mt;j++){ int e=na[mt*nd+j]; if(e>=1&&e<=n) cnt[j]++; }
    std::vector<int> ordr(mt); for(int j=0;j<mt;j++) ordr[j]=j;
    std::sort(ordr.begin(),ordr.end(),[&](int a,int c){return cnt[a]>cnt[c];});
    std::vector<int> disp(ordr.begin(),ordr.begin()+3); std::sort(disp.begin(),disp.end());
    double maxU=0, l2=0;
    for (long nd=0; nd<nk; nd++){ double u[3]={0,0,0}; bool any=false;
        for(int ci=0;ci<3;ci++){ int e=na[mt*nd+disp[ci]]; if(e>=1&&e<=n){ u[ci]=b[e-1]; any=true; } }
        if(any){ double m=std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]); if(m>maxU)maxU=m; } }
    for (int i=0;i<n;i++) l2 += b[i]*b[i]; l2=std::sqrt(l2);

    printf("SUMMARY variant=%s rc=%d solve_s=%.3f maxU=%.10f l2=%.10e tol=%.0e\n",
           variant, rc, solve_s, maxU, l2, tol);
    return rc;
}
