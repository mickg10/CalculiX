// CPU validation of the dump-loading GMG entry (no TT hook set -> fine smoother uses exact fp64 bspmv).
// If this converges to maxU = 107.0569734213 on row236, the dump loader + CSR rebuild + hierarchy + PCG are correct.
#include <cstdio>
#include <cstdlib>
#include <cmath>
extern "C" int ccx_gmg_solve_from_dump(const char*, int, double, int, double*, double*);
int main(int argc, char** argv) {
    const char* path = argc>1 ? argv[1] : "/tmp/row236_fine.bin";
    double maxu = 0;
    int rc = ccx_gmg_solve_from_dump(path, 500, 1e-6, 1, nullptr, &maxu);
    const double golden = 107.0569734213;
    printf("rc=%d maxU=%.7f golden=%.7f rel=%.3e %s\n", rc, maxu, golden,
           std::fabs(maxu-golden)/golden, (rc==0 && std::fabs(maxu-golden)/golden < 1e-2) ? "PASS" : "CHECK");
    return rc;
}
