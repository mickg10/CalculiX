/* accel.h — prototypes for the optional Apple Accelerate / geometric-multigrid backend.  Tooling / APHYSICAL.
 *
 * This backend is compiled in only when CalculiX is built with -DCCX_ACCEL (see Makefile_accel). Every hook
 * in the upstream sources (spooles.c, linstatic.c, frd.c, printout.f, printoutint.f) is wrapped in
 * #ifdef CCX_ACCEL / #ifdef CCX_ACCEL_ARROW, so a default build (Makefile / Makefile_MT) is byte-for-byte
 * unaffected. At run time every feature is additionally env-gated and defaults to off, falling back to the
 * stock CalculiX path. Include this AFTER CalculiX.h (it relies on ITG).
 */
#ifndef CCX_ACCEL_H
#define CCX_ACCEL_H

/* Symmetric real direct / GMG solve behind spooles() for the inputformat==0 (lower-CSC) contract.
   Returns 0 on success (solution written into b); nonzero -> caller falls back to SPOOLES. */
int  accel_spooles(double *ad, double *au, double *adb, double *sigma,
                   double *b, ITG *icol, ITG *irow, ITG *neq, ITG *nzs);

/* Stash CalculiX's in-memory coord/DOF map (co, nactdof, nk, mi) right before the static solve so the
   GMG path can reconstruct the voxel grid without a file. Also honours CCX_ACCEL_DUMP2=<path> (dump+continue)
   for offline analysis. Fortran-style trailing-underscore symbol (called from C, but ABI-stable). */
void accel_set_coordmap_(double *co, ITG *nactdof, ITG *nk, ITG *mi);

/* Write nodal fields (coords, displacement, optional stress) as an Arrow IPC stream. Requires the vendored
   nanoarrow objects; only referenced under CCX_ACCEL_ARROW. Returns 0 on success. */
int  ccx_arrow_write(const char *path, long nk, int mt,
                     const double *co, const double *v, const double *stn);

/* Transparent streaming reader for plain / gzip (.gz) / zstd (.zst) input decks (readinput.c). ccx_zopen
   also tries <path>.gz then <path>.zst when <path> is absent; ccx_zgets matches fgets semantics. */
typedef struct ccx_zfile ccx_zfile;
ccx_zfile *ccx_zopen(const char *path);
char      *ccx_zgets(char *buf, int n, ccx_zfile *z);
void       ccx_zclose(ccx_zfile *z);

#endif /* CCX_ACCEL_H */
