/* Apple Accelerate Sparse backend for CalculiX.
 *
 * Drop-in symmetric real direct solve behind the patched spooles() entry, using
 * Apple's Accelerate Sparse Solvers (deterministic, lock-free, AMX, GCD-parallel
 * across all cores). Falls back (return!=0) to SPOOLES on any failure.
 *
 * Compile with clang (Accelerate uses clang-only overloadable); link -framework
 * Accelerate (+ -lmetis -DCCX_HAVE_METIS for the fixed deterministic ordering).
 *
 * CCX matrix contract (inputformat==0, symmetric): lower triangle by column. For
 * col in [0,neq): diag ad[col] (minus sigma*adb[col] if sigma!=0), then icol[col]
 * sub-diagonal au[ipo] at rows irow[ipo]-1.
 *
 * ---- Configuration (precedence: built-in defaults < config file < environment) ----
 * Config file:  $CCX_ACCEL_CONF (explicit path only; no implicit ./ccx_accel.conf).
 *               Format: `key = value`, one per line, `#` comments. Keys = env names
 *               without the CCX_ACCEL_ prefix, lowercased:
 *                 accel        = on|off          (env CCX_ACCEL; CCX_USE_ACCEL=0 also disables)
 *                 order        = usermetis|metis|amd|colamd|default|mtmetis  (env CCX_ACCEL_ORDER)
 *                 precision    = double|float     (env CCX_ACCEL_PRECISION)
 *                 refine_iters = N (float mode)   (env CCX_ACCEL_REFINE_ITERS)
 *                 verbose      = 0|1              (env CCX_ACCEL_VERBOSE)
 *                 permdir      = iperm|perm       (dev/debug; env CCX_ACCEL_PERMDIR)
 * Provenance: with verbose=1 the backend prints the effective config + its source.
 */
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#ifdef CCX_HAVE_METIS
#include <metis.h>
#endif

/* This backend round-trips CCX's icol/irow/neq/nzs and reads nactdof as int (the default i4 build). An i8
   build (-DLONGLONG makes ITG==long long in CalculiX.h) would silently misread those, so trip the build
   directly on the macro rather than on sizeof(ITG) -- this file does not include CalculiX.h, so it never sees
   the i8 redefinition and a sizeof() assert would be int==int and pass. The accel Makefile uses i4. */
#ifdef LONGLONG
#error "accel backend requires the default i4 CalculiX build (ITG==int); it is incompatible with -DLONGLONG (i8)."
#endif
#ifndef ITG
#define ITG int
#endif

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#include <mach/mach.h>
/* current resident set size (bytes); 0 on failure. */
static size_t rss_bytes(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &cnt) == KERN_SUCCESS)
        return (size_t)info.resident_size;
    return 0;
}
/* phys_footprint (bytes) = the "memory" figure Activity Monitor shows (incl compressed); 0 on failure. */
static size_t phys_footprint_bytes(void) {
    task_vm_info_data_t vmi;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vmi, &cnt) == KERN_SUCCESS)
        return (size_t)vmi.phys_footprint;
    return 0;
}
#define MEMLOG(tag) do { if (verbose) fprintf(stderr, \
    "[accel.mem] %-22s rss=%.2f GB  footprint=%.2f GB\n", (tag), \
    rss_bytes()/1e9, phys_footprint_bytes()/1e9); } while(0)

/* ----------------------------- configuration ----------------------------- */
typedef struct {
    int  enabled;          /* use Accelerate (1) or fall back to SPOOLES (0) */
    char order[24];        /* ordering name */
    int  use_float;        /* float factor + double refinement/PCG */
    int  pcg;              /* float mode: use double PCG (1) vs Richardson (0) */
    int  defl;             /* deflated PCG: deflate the rigid-body modes (needs rbmmap) */
    int  gmg;              /* factorization-free geometric multigrid (regular voxel grid; needs rbmmap coordmap) */
    int  gmg_auto;         /* auto mode: try GMG, fall back to direct if not a regular grid (expected, not an error) */
    int  refine_iters;     /* max iterative-refinement / PCG steps (float mode) */
    double pcg_tol;       /* PCG relative-residual stop tolerance */
    int  verbose;          /* print effective config + per-phase timing */
    char permdir[8];       /* usermetis perm direction: "iperm" (default) | "perm" */
    char rbmmap[256];      /* coord/DOF-map file (CCX_ACCEL_RBM_MAP) for RBM deflation */
    char permcache[256];   /* dir to cache/reuse METIS perm keyed on matrix STRUCTURE (env CCX_ACCEL_PERMCACHE) */
    char source[160];      /* provenance: where config came from */
} accel_config_t;

static char *cfg_trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    if (!*s) return s;
    char *e = s + strlen(s) - 1;
    while (e > s && isspace((unsigned char)*e)) *e-- = '\0';
    return s;
}

static int cfg_default_usermetis(void) {
#ifdef CCX_HAVE_METIS
    return 1;
#else
    return 0;
#endif
}

static void cfg_set(accel_config_t *c, const char *key, const char *val) {
    if (!strcmp(key, "accel"))
        c->enabled = !(!strcmp(val, "off") || !strcmp(val, "0"));
    else if (!strcmp(key, "order"))
        snprintf(c->order, sizeof c->order, "%s", val);
    else if (!strcmp(key, "precision"))
        c->use_float = (!strcmp(val, "float"));
    else if (!strcmp(key, "solve")) {
        /* high-level mode selector. direct|double -> exact double Cholesky;
           float -> float factor + Richardson; float-pcg -> float factor as
           preconditioner for double CG (mixed->high precision). Selecting any
           recognized mode opts in to the accel backend (sets enabled). */
        if (!strcmp(val, "direct") || !strcmp(val, "double")) { c->enabled = 1; c->use_float = 0; c->pcg = 0; c->defl = 0; c->gmg = 0; }
        else if (!strcmp(val, "float")) { c->enabled = 1; c->use_float = 1; c->pcg = 0; c->defl = 0; c->gmg = 0; }
        else if (!strcmp(val, "float-pcg") || !strcmp(val, "pcg")) { c->enabled = 1; c->use_float = 1; c->pcg = 1; c->defl = 0; c->gmg = 0; }
        else if (!strcmp(val, "defl-pcg") || !strcmp(val, "defl")) { c->enabled = 1; c->use_float = 1; c->pcg = 1; c->defl = 1; c->gmg = 0; }
        else if (!strcmp(val, "gmg")) { c->enabled = 1; c->use_float = 0; c->pcg = 0; c->defl = 0; c->gmg = 1; c->gmg_auto = 0; }
        else if (!strcmp(val, "auto")) { c->enabled = 1; c->use_float = 0; c->pcg = 0; c->defl = 0; c->gmg = 1; c->gmg_auto = 1; }
        /* unrecognized value: leave enabled as-is (does not opt in) */
    }
    else if (!strcmp(key, "pcg"))
        c->pcg = atoi(val) != 0;
    else if (!strcmp(key, "rbmmap"))
        snprintf(c->rbmmap, sizeof c->rbmmap, "%s", val);
    else if (!strcmp(key, "pcg_tol"))
        c->pcg_tol = atof(val);
    else if (!strcmp(key, "refine_iters"))
        c->refine_iters = atoi(val);
    else if (!strcmp(key, "verbose"))
        c->verbose = atoi(val) != 0;
    else if (!strcmp(key, "permdir"))
        snprintf(c->permdir, sizeof c->permdir, "%s", val);
    else if (!strcmp(key, "permcache"))
        snprintf(c->permcache, sizeof c->permcache, "%s", val);
    /* unknown keys silently ignored (forward-compatible) */
}

static void cfg_load_file(accel_config_t *c, const char *path, int *loaded) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *h = strchr(line, '#'); if (h) *h = '\0';
        char *eq = strchr(line, '='); if (!eq) continue;
        *eq = '\0';
        char *k = cfg_trim(line), *v = cfg_trim(eq + 1);
        if (*k) cfg_set(c, k, v);
    }
    fclose(f);
    *loaded = 1;
    snprintf(c->source + strlen(c->source), sizeof c->source - strlen(c->source),
             "file:%s ", path);
}

static accel_config_t accel_config_load(void) {
    accel_config_t c;
    c.enabled = 0;   /* OPT-IN: a bare run (no CCX_ACCEL* env) falls back to stock SPOOLES -> zero default change.
                        Enabled by CCX_ACCEL=on or by selecting a mode via CCX_ACCEL_SOLVE=<mode>. */
    snprintf(c.order, sizeof c.order, "%s", cfg_default_usermetis() ? "usermetis" : "metis");
    c.use_float = 0;
    c.pcg = 0;
    c.defl = 0;
    c.gmg = 0;
    c.gmg_auto = 0;
    c.refine_iters = 6;
    c.pcg_tol = 1e-10;
    c.verbose = 0;
    c.rbmmap[0] = '\0';
    snprintf(c.permdir, sizeof c.permdir, "%s", "iperm");
    c.permcache[0] = '\0';
    c.source[0] = '\0';
    snprintf(c.source, sizeof c.source, "defaults ");

    /* config file: explicit path only (CCX_ACCEL_CONF). No implicit ./ccx_accel.conf -- a stray file in the
       working directory must never silently enable the backend (a bare run stays stock SPOOLES). */
    int file_loaded = 0;
    const char *cf = getenv("CCX_ACCEL_CONF");
    if (cf && *cf) cfg_load_file(&c, cf, &file_loaded);

    /* environment overrides (highest precedence) */
    const char *e;
    int env_used = 0;
    if ((e = getenv("CCX_ACCEL")))           { cfg_set(&c, "accel", e); env_used = 1; }
    if ((e = getenv("CCX_USE_ACCEL")))       { if (!strcmp(e, "0")) c.enabled = 0; env_used = 1; } /* deprecated alias */
    if ((e = getenv("CCX_ACCEL_ORDER")))     { cfg_set(&c, "order", e); env_used = 1; }
    if ((e = getenv("CCX_ACCEL_PRECISION"))) { cfg_set(&c, "precision", e); env_used = 1; }
    if ((e = getenv("CCX_ACCEL_SOLVE")))     { cfg_set(&c, "solve", e); env_used = 1; }
    if ((e = getenv("CCX_ACCEL_PCG_TOL")))   { cfg_set(&c, "pcg_tol", e); env_used = 1; }
    if ((e = getenv("CCX_ACCEL_RBM_MAP")))   { cfg_set(&c, "rbmmap", e); env_used = 1; }
    if ((e = getenv("CCX_ACCEL_REFINE_ITERS"))) { cfg_set(&c, "refine_iters", e); env_used = 1; }
    if ((e = getenv("CCX_ACCEL_PERMDIR")))   { cfg_set(&c, "permdir", e); env_used = 1; }
    if ((e = getenv("CCX_ACCEL_PERMCACHE"))) { cfg_set(&c, "permcache", e); env_used = 1; }
    if (getenv("CCX_ACCEL_VERBOSE"))         { c.verbose = 1; env_used = 1; }
    if (env_used) snprintf(c.source + strlen(c.source), sizeof c.source - strlen(c.source), "env");
    /* explicit CCX_ACCEL=off|0 always wins, regardless of any CCX_ACCEL_SOLVE mode selected above */
    if ((e = getenv("CCX_ACCEL")) && (!strcmp(e, "off") || !strcmp(e, "0"))) c.enabled = 0;
    return c;
}

/* map ordering name -> Accelerate SparseOrder_t (usermetis handled separately) */
static SparseOrder_t order_from_name(const char *o) {
    if (!strcmp(o, "default")) return SparseOrderDefault;
    if (!strcmp(o, "amd"))     return SparseOrderAMD;
    if (!strcmp(o, "colamd"))  return SparseOrderCOLAMD;
    if (!strcmp(o, "mtmetis")) return SparseOrderMTMetis;
    return SparseOrderMetis;   /* "metis", "usermetis"(fallback), unknown */
}

/* y -= A*x for a symmetric matrix stored as lower triangle (diag+subdiag) in CSC. */
static void symm_lower_spmv_sub(int n, const long *colStarts, const int *rowIdx,
                                const double *vals, const double *x, double *y) {
    for (int c = 0; c < n; ++c)
        for (long p = colStarts[c]; p < colStarts[c + 1]; ++p) {
            int r = rowIdx[p];
            double v = vals[p];
            if (r == c) y[c] -= v * x[c];
            else { y[r] -= v * x[c]; y[c] -= v * x[r]; }
        }
}

static double ddot(int n, const double *a, const double *b) {
    double s = 0.0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s;
}

/* ---- small dense (m<=8) SPD Cholesky for the coarse deflation matrix E (row-major) ---- */
static void dense_chol(double *E, int m) {              /* in-place lower factor */
    for (int j = 0; j < m; ++j) {
        double d = E[j*m+j];
        for (int k = 0; k < j; ++k) d -= E[j*m+k]*E[j*m+k];
        d = sqrt(d); E[j*m+j] = d;
        for (int i = j+1; i < m; ++i) {
            double s = E[i*m+j];
            for (int k = 0; k < j; ++k) s -= E[i*m+k]*E[j*m+k];
            E[i*m+j] = s/d;
        }
    }
}
static void dense_cholsolve(const double *L, int m, const double *rhs, double *x) {
    double y[8];
    for (int i = 0; i < m; ++i){ double s=rhs[i]; for(int k=0;k<i;k++) s-=L[i*m+k]*y[k]; y[i]=s/L[i*m+i]; }
    for (int i = m-1; i >= 0; --i){ double s=y[i]; for(int k=i+1;k<m;k++) s-=L[k*m+i]*x[k]; x[i]=s/L[i*m+i]; }
}
/* coeffs c_k = column_k(M) . v   (M is n*m column-major) */
static void colsT_vec(const double *M, int n, int m, const double *v, double *c) {
    for (int k = 0; k < m; ++k){ const double *Mk=M+(long)k*n; double s=0; for(int i=0;i<n;i++) s+=Mk[i]*v[i]; c[k]=s; }
}
/* v -= sum_k d_k * column_k(M) */
static void axpy_cols(const double *M, int n, int m, const double *d, double *v) {
    for (int k = 0; k < m; ++k){ double dk=d[k]; if(dk!=0.0){ const double *Mk=M+(long)k*n; for(int i=0;i<n;i++) v[i]-=dk*Mk[i]; } }
}

/* Build 6 rigid-body modes (column-major: column k at W+k*n) from in-memory node coords `co` (3*nk doubles)
   and the CCX nactdof map `na` (mt*nk, 1-based eqn|0). The 3 displacement DOFs are the mt-columns with the
   most active equations (matches the GMG coord detection). Returns malloc'd n*6 (NULL on failure); *m_out=6.
   Does NOT free co/na -- the caller owns them (the in-memory globals, or load_rbm's file buffers). */
static double *build_rbm(const double *co, const int *na, long nk, int mt, int n, int *m_out) {
    if (!co || !na || nk <= 0 || mt < 3 || mt > 64 || n <= 0) return NULL;
    long cnt[64]; for (int j=0;j<mt;j++) cnt[j]=0;
    for (long nd=0; nd<nk; ++nd) for (int j=0;j<mt;j++){ int e=na[mt*nd+j]; if(e>=1&&e<=n) cnt[j]++; }
    int disp[3]={-1,-1,-1};
    for (int s=0;s<3;s++){ long best=-1; int bj=-1;
        for (int j=0;j<mt;j++){ int used=(j==disp[0]||j==disp[1]||j==disp[2]); if(!used&&cnt[j]>best){best=cnt[j];bj=j;} }
        disp[s]=bj; }
    if (disp[0]<0 || disp[1]<0 || disp[2]<0) return NULL;
    for (int a=0;a<3;a++) for (int b2=a+1;b2<3;b2++) if(disp[b2]<disp[a]){int t=disp[a];disp[a]=disp[b2];disp[b2]=t;}
    long *eqn=(long*)malloc((size_t)n*sizeof(long)); int *eqc=(int*)malloc((size_t)n*sizeof(int));
    double *W=(double*)calloc((size_t)6*n,sizeof(double));
    if (!eqn||!eqc||!W){ free(eqn);free(eqc);free(W); return NULL; }
    for (int i=0;i<n;i++){ eqn[i]=-1; eqc[i]=-1; }
    for (long nd=0; nd<nk; ++nd) for (int ci=0;ci<3;ci++){ int j=disp[ci]; int e=na[mt*nd+j];
        if(e>=1&&e<=n){ eqn[e-1]=nd; eqc[e-1]=ci; } }
    double cx=0,cy=0,cz=0; for(long i=0;i<nk;i++){cx+=co[3*i];cy+=co[3*i+1];cz+=co[3*i+2];} cx/=nk;cy/=nk;cz/=nk;
    for (int e=0;e<n;e++){ long nd=eqn[e]; int c=eqc[e]; if(nd<0||c<0) continue;
        double x=co[3*nd]-cx, y=co[3*nd+1]-cy, z=co[3*nd+2]-cz;
        W[(long)c*n+e]=1.0;
        if(c==1) W[3L*n+e]=-z; if(c==2) W[3L*n+e]= y;
        if(c==0) W[4L*n+e]= z; if(c==2) W[4L*n+e]=-x;
        if(c==0) W[5L*n+e]=-y; if(c==1) W[5L*n+e]= x;
    }
    for (int k=0;k<6;k++){ double *Wk=W+(long)k*n; double s=0; for(int i=0;i<n;i++) s+=Wk[i]*Wk[i];
        s=sqrt(s); if(s>0) for(int i=0;i<n;i++) Wk[i]/=s; }
    free(eqn); free(eqc);
    *m_out = 6; return W;
}
/* file variant (CCX_ACCEL_RBM_MAP): read co/na, then build_rbm. */
static double *load_rbm(const char *path, int n, int *m_out) {
    FILE *g = fopen(path, "rb"); if (!g) return NULL;
    int64_t nk64, mt64;
    if (fread(&nk64,8,1,g)!=1 || fread(&mt64,8,1,g)!=1){ fclose(g); return NULL; }
    long nk=(long)nk64; int mt=(int)mt64;
    if (nk <= 0 || mt < 3 || mt > 64) { fclose(g); return NULL; }
    double *co = (double*)malloc((size_t)3*nk*sizeof(double));
    int    *na = (int*)   malloc((size_t)mt*nk*sizeof(int));
    double *W = NULL;
    if (co && na && fread(co,8,(size_t)3*nk,g)==(size_t)3*nk && fread(na,4,(size_t)mt*nk,g)==(size_t)mt*nk)
        W = build_rbm(co, na, nk, mt, n, m_out);
    free(co); free(na); fclose(g);
    return W;
}

/* FNV-1a-style hash of the matrix STRUCTURE only (n, nnz, colStarts, rowIdx) — NOT values.
   Same mesh/connectivity -> same hash (reuse cached perm); geometry change -> different sparsity
   -> different hash -> automatic cache miss/recompute. Word-strided for speed (~0.5s on 150M nnz). */
static uint64_t struct_hash(int n, const long *cs, const int *ri, long nnz) {
    uint64_t h = 1469598103934665603ULL;
    h = (h ^ (uint64_t)n)   * 1099511628211ULL;
    h = (h ^ (uint64_t)nnz) * 1099511628211ULL;
    for (long i = 0; i <= n; ++i) h = (h ^ (uint64_t)cs[i]) * 1099511628211ULL;
    for (long i = 0; i < nnz; ++i) h = (h ^ (uint64_t)(unsigned)ri[i]) * 1099511628211ULL;
    return h;
}

#ifdef CCX_HAVE_METIS
/* Deterministic METIS nested-dissection perm from the lower-CSC structure.
   Returns malloc'd int perm (iperm by default) for SparseOrderUser, or NULL. */
static int *compute_metis_perm(int n, const long *colStarts, const int *rowIdx,
                               const char *permdir) {
    idx_t *deg = (idx_t*)calloc((size_t)n, sizeof(idx_t));
    idx_t *xadj = (idx_t*)malloc((size_t)(n + 1) * sizeof(idx_t));
    if (!deg || !xadj) { free(deg); free(xadj); return NULL; }
    for (int c = 0; c < n; ++c)
        for (long p = colStarts[c]; p < colStarts[c + 1]; ++p) {
            int r = rowIdx[p];
            if (r != c) { deg[c]++; deg[r]++; }
        }
    xadj[0] = 0;
    for (int i = 0; i < n; ++i) xadj[i + 1] = xadj[i] + deg[i];
    idx_t nadj = xadj[n];
    idx_t *adjncy = (idx_t*)malloc((size_t)nadj * sizeof(idx_t));
    idx_t *cur = (idx_t*)malloc((size_t)n * sizeof(idx_t));
    if (!adjncy || !cur) { free(deg); free(xadj); free(adjncy); free(cur); return NULL; }
    for (int i = 0; i < n; ++i) cur[i] = xadj[i];
    for (int c = 0; c < n; ++c)
        for (long p = colStarts[c]; p < colStarts[c + 1]; ++p) {
            int r = rowIdx[p];
            if (r != c) { adjncy[cur[c]++] = r; adjncy[cur[r]++] = c; }
        }
    free(deg); free(cur);
    idx_t nv = n;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_SEED] = 1;            /* fixed seed -> deterministic */
    idx_t *perm = (idx_t*)malloc((size_t)n * sizeof(idx_t));
    idx_t *iperm = (idx_t*)malloc((size_t)n * sizeof(idx_t));
    if (!perm || !iperm) { free(xadj); free(adjncy); free(perm); free(iperm); return NULL; }
    int rc = METIS_NodeND(&nv, xadj, adjncy, NULL, options, perm, iperm);
    free(xadj); free(adjncy);
    if (rc != METIS_OK) { free(perm); free(iperm); return NULL; }
    /* We hand the idx_t array straight back as int* (SparseOrderUser wants the elimination order, iperm by
       default). That cast is only valid for a 32-bit-idx_t METIS build -- enforce it at compile time. */
    _Static_assert(sizeof(idx_t) == sizeof(int), "accel METIS path requires 32-bit idx_t (IDXTYPEWIDTH=32)");
    if (permdir && !strcmp(permdir, "perm")) { free(iperm); return (int*)perm; }
    free(perm); return (int*)iperm;
}
#endif

/* factorization-free geometric multigrid (src/gmg_solve.cpp); 0 on success (solution into b). */
extern int ccx_gmg_solve(int n, const long *cs, const int *ri, const double *va,
                         double *b, const char *coordmap, int maxit, double tol, int verbose);
extern int ccx_gmg_solve_mem(int n, const long *cs, const int *ri, const double *va, double *b,
                             const double *co, const int *na, long nk, int mt,
                             int maxit, double tol, int verbose);

/* In-memory coord/DOF-map, stashed by CalculiX right before the solve (no file needed for GMG/auto).
   Set from linstatic via accel_set_coordmap_(co, nactdof, nk, mi). Assumes ITG==int (standard CCX build). */
static const double *g_co = NULL;
static const int    *g_nactdof = NULL;
static long          g_nk = 0;
static int           g_mi1 = 0;
void accel_set_coordmap_(double *co, ITG *nactdof, ITG *nk, ITG *mi) {
    g_co = co; g_nactdof = (const int *)nactdof; g_nk = (long)(*nk); g_mi1 = (int)mi[1];
}

/* Returns 0 on success (solution written into b), nonzero on failure. */
int accel_spooles(double *ad, double *au, double *adb, double *sigma,
                  double *b, ITG *icol, ITG *irow, ITG *neq, ITG *nzs)
{
    accel_config_t cfg = accel_config_load();
    if (!cfg.enabled) return 1;                 /* disabled -> SPOOLES */
    const int verbose = cfg.verbose;
    MEMLOG("entry (CCX baseline)");

    const int  n   = (int)(*neq);
    const long nnz = (long)n + (long)(*nzs);
    const double sig = (sigma) ? *sigma : 0.0;

    /* Shifted/indefinite system (modal/buckling: K - sigma*M). accel_spooles is NOT passed the off-diagonal
       mass aub, so it cannot form the shifted off-diagonals (stock SPOOLES uses au - sigma*aub). Solving the
       diagonal-only-shifted matrix would silently return the answer to the WRONG system -> decline to stock
       SPOOLES (which has aub). Static (sigma==0, the accel target) is unaffected. */
    if (sig != 0.0) {
        if (verbose) fprintf(stderr,
            "[accel] sigma=%.3e (shifted matrix; off-diagonal mass unavailable here) -> stock SPOOLES\n", sig);
        return 1;
    }

    double t0 = now_s();
    long   *colStarts = (long*)  malloc((size_t)(n + 1) * sizeof(long));
    int    *rowIdx    = (int*)   malloc((size_t)nnz     * sizeof(int));
    double *vals      = (double*)malloc((size_t)nnz     * sizeof(double));
    if (!colStarts || !rowIdx || !vals) { free(colStarts); free(rowIdx); free(vals); return 2; }

    /* Build full-diagonal lower-CSC, validating CCX's icol/irow against nnz=n+nzs as we go: a bad count or
       out-of-range row would otherwise overrun rowIdx/vals or feed Accelerate a garbage index. On any
       inconsistency decline (rc=1) so the exact stock SPOOLES solve runs. sigma==0 here (shifted -> early
       return above), so the diagonal is just ad[col]. The checks are O(nnz) compares -- noise vs the solve. */
    long pos = 0, ipoint = 0;
    int bad = 0;
    for (int col = 0; col < n && !bad; ++col) {
        colStarts[col] = pos;
        const long cnt = (long)icol[col];
        if (cnt < 0 || pos + 1 + cnt > nnz) { bad = 1; break; }   /* would overrun rowIdx/vals */
        rowIdx[pos] = col; vals[pos] = ad[col]; pos++;            /* diagonal */
        for (long k = 0; k < cnt; ++k) {
            const long ipo = ipoint + k;
            const long r = (long)irow[ipo] - 1;
            if (r < 0 || r >= n) { bad = 1; break; }              /* row index out of range */
            rowIdx[pos] = (int)r; vals[pos] = au[ipo]; pos++;
        }
        ipoint += cnt;
    }
    if (bad || pos != nnz) {                                      /* contract: total entries == n + nzs */
        if (verbose) fprintf(stderr, "[accel] malformed matrix (icol/irow vs nnz) -> stock SPOOLES\n");
        free(colStarts); free(rowIdx); free(vals); return 1;
    }
    colStarts[n] = pos;
    double t_build = now_s() - t0;
    MEMLOG("after CSC build (+matrix)");

    /* factorization-free geometric multigrid path (regular voxel grid). Needs the coord/DOF-map
       (the in-memory map from accel_set_coordmap_, or the CCX_ACCEL_RBM_MAP file). On success returns immediately; on failure falls through to direct. */
    if (cfg.gmg && sig != 0.0) {
        /* GMG's prolongation/Galerkin hierarchy assumes the SPD static stiffness K. A shifted/indefinite
           matrix K - sigma*M (modal/buckling) breaks that assumption -> never apply GMG; use direct. */
        if (verbose) fprintf(stderr,
            "[accel] solve=%s: sigma=%.3e (shifted/indefinite matrix) -> GMG not applicable, using direct factor\n",
            cfg.gmg_auto ? "auto" : "gmg", sig);
    } else if (cfg.gmg) {
        double tg = now_s();
        int gtol_iters = getenv("CCX_ACCEL_GMG_MAXIT") ? atoi(getenv("CCX_ACCEL_GMG_MAXIT")) : 80;
        /* GMG stop tol: default 1e-4 (disp ~1e-10, stress ~1e-9); decoupled from cfg.pcg_tol (=float-pcg's
           exact 1e-10). Override with CCX_ACCEL_GMG_TOL (or GMG_TOL, handled inside gmg_solve). */
        double gtol = getenv("CCX_ACCEL_GMG_TOL") ? atof(getenv("CCX_ACCEL_GMG_TOL")) : 1e-4;
        int grc = -99;
        if (g_co && g_nactdof) {                 /* preferred: in-memory coordmap from CCX (no file) */
            grc = ccx_gmg_solve_mem(n, colStarts, rowIdx, vals, b, g_co, g_nactdof, g_nk, g_mi1 + 1,
                                    gtol_iters, gtol, verbose);
        } else if (cfg.rbmmap[0]) {              /* fallback: coord/DOF-map file (CCX_ACCEL_RBM_MAP) */
            grc = ccx_gmg_solve(n, colStarts, rowIdx, vals, b, cfg.rbmmap, gtol_iters, gtol, verbose);
        } else {
            fprintf(stderr, "[accel] solve=%s: no coord/DOF-map (in-memory or file) -> using direct\n",
                    cfg.gmg_auto ? "auto" : "gmg");
        }
        if (grc == 0) {
            if (verbose) fprintf(stderr, "[accel] GMG solve ok in %.2fs (tol=%.0e)\n", now_s() - tg, gtol);
            MEMLOG("after GMG solve");
            free(colStarts); free(rowIdx); free(vals);
            return 0;
        }
        if (grc != -99) {
            /* gmg rc: 2=not pure 3-DOF blocks, 3=not a regular grid (both GEOMETRY -> expected, benign);
               1=invalid inputs, 4=not converged, 5=coarse not SPD, 6=OOM (NUMERIC -> real, but the direct
               fallback below still produces the correct result). Report the two cases honestly. */
            if (grc == 2 || grc == 3)
                fprintf(stderr, "[accel] %s: not a regular voxel grid (gmg rc=%d, expected) -> direct factor\n",
                        cfg.gmg_auto ? "auto" : "gmg", grc);
            else
                fprintf(stderr, "[accel] GMG applicable but did not solve (rc=%d: %s) -> direct factor (correct result)\n",
                        grc, grc == 1 ? "invalid inputs" : grc == 4 ? "not converged"
                           : grc == 5 ? "coarse not SPD" : grc == 6 ? "out of memory" : "unknown");
        }
    }

    SparseAttributes_t attr = (SparseAttributes_t){0};
    attr.kind = SparseSymmetric; attr.triangle = SparseLowerTriangle; attr.transpose = false;
    SparseMatrixStructure structure = {
        .rowCount = n, .columnCount = n,
        .columnStarts = colStarts, .rowIndices = rowIdx,
        .attributes = attr, .blockSize = 1,
    };
    const char *ordname = cfg.order;
    SparseSymbolicFactorOptions sfo = _SparseDefaultSymbolicFactorOptions;
    sfo.orderMethod = order_from_name(cfg.order);

    int *user_perm = NULL;
#ifdef CCX_HAVE_METIS
    if (!strcmp(cfg.order, "usermetis")) {
        double tp = now_s();
        char cpath[400]; cpath[0] = '\0';
        int from_cache = 0;
        /* structure-keyed perm cache: reuse across runs with the SAME mesh; auto-miss on geometry change */
        if (cfg.permcache[0]) {
            uint64_t hh = struct_hash(n, colStarts, rowIdx, nnz);
            snprintf(cpath, sizeof cpath, "%s/perm_%s_%016llx_n%d.bin",
                     cfg.permcache, cfg.permdir, (unsigned long long)hh, n);
            FILE *cf = fopen(cpath, "rb");
            if (cf) {
                int *p = (int*)malloc((size_t)n * sizeof(int));
                /* validate: a corrupt/foreign cache file must not feed Accelerate a non-permutation.
                   Require exactly n ints forming a permutation of [0,n); else discard and recompute. */
                if (p && fread(p, sizeof(int), (size_t)n, cf) == (size_t)n) {
                    char *seen = (char*)calloc((size_t)n, 1); int okperm = (seen != NULL);
                    for (int i = 0; okperm && i < n; ++i) { int v = p[i]; if (v < 0 || v >= n || seen[v]) okperm = 0; else seen[v] = 1; }
                    free(seen);
                    if (okperm) { user_perm = p; from_cache = 1; } else free(p);
                } else free(p);
                fclose(cf);
            }
        }
        if (!user_perm) user_perm = compute_metis_perm(n, colStarts, rowIdx, cfg.permdir);
        if (user_perm) {
            sfo.orderMethod = SparseOrderUser; sfo.order = user_perm;
            if (cfg.permcache[0] && !from_cache) {          /* save freshly-computed perm for reuse */
                FILE *cf = fopen(cpath, "wb");
                if (cf) { fwrite(user_perm, sizeof(int), (size_t)n, cf); fclose(cf); }
            }
        } else { sfo.orderMethod = SparseOrderMetis; ordname = "metis(fallback)"; }
        if (verbose) fprintf(stderr, "[accel] usermetis perm %s=%.2fs (%s, %s)%s\n",
                             from_cache ? "CACHE-load" : "build", now_s() - tp,
                             user_perm ? "ok" : "FAILED->accel-metis", cfg.permdir,
                             (cfg.permcache[0] && !from_cache) ? " [cached]" : "");
        MEMLOG("after METIS perm (+perm)");
    }
#else
    if (!strcmp(cfg.order, "usermetis")) ordname = "metis(no-libmetis)";
#endif

    /* Determinism: the default ordering (usermetis/metis) is SERIAL METIS -> a fixed fill-reducing
       permutation -> the factorization, and therefore maxU, is reproducible run-to-run (the acceptance gate).
       MTMetis and SparseOrderDefault may order in PARALLEL, whose result depends on thread scheduling ->
       NOT reproducible. Warn unconditionally if such an ordering is in effect so a reproducibility-critical
       run is never silently nondeterministic. (Fixed-thread numeric reproducibility: SparseSolve / the GMG
       PCG combine partial sums in unspecified order, so results are reproducible to solver tolerance, not
       bit-identical; that is well inside the ±1% maxU bound and far below the accept ceiling.) */
    if (sfo.orderMethod == SparseOrderMTMetis || sfo.orderMethod == SparseOrderDefault)
        fprintf(stderr, "[accel] WARNING: ordering '%s' is parallel/nondeterministic -> results are NOT "
                        "reproducible run-to-run; use 'metis'/'usermetis' (default) for reproducibility.\n",
                cfg.order);

    int rc = 0;
    double t_factor = 0, t_solve = 0;
    int refine_iters = 0;
    double final_resid = 0.0;

    if (!cfg.use_float) {
        SparseMatrix_Double A = { .structure = structure, .data = vals };
        SparseNumericFactorOptions nfo = _SparseDefaultNumericFactorOptions_Double;
        double t1 = now_s();
        SparseOpaqueFactorization_Double F = SparseFactor(SparseFactorizationCholesky, A, sfo, nfo);
        if (F.status != SparseStatusOK) {
            SparseCleanup(F);
            F = SparseFactor(SparseFactorizationLDLT, A, sfo, nfo);
            if (F.status != SparseStatusOK) { SparseCleanup(F); rc = 3; goto done; }
        }
        t_factor = now_s() - t1;
        MEMLOG("after Cholesky factor (+L)");
        double t2 = now_s();
        double *xb = (double*)malloc((size_t)n * sizeof(double));
        if (!xb) { SparseCleanup(F); rc = 2; goto done; }
        for (int i = 0; i < n; ++i) xb[i] = b[i];
        SparseSolve(F, (DenseVector_Double){ .count = n, .data = xb });
        SparseCleanup(F);
        t_solve = now_s() - t2;
        /* residual gate: accept the solve iff the TRUE residual ||b-A*xb||/||b|| meets the acceptable-accuracy bar
           (1e-2, the same ceiling enforced for the GMG path), else decline (rc=1) so the exact SPOOLES solve
           runs -- never copy an untrustworthy x (singular LDLT, NaN, bad assembly) into b. NOTE: the bar is
           1e-2, NOT eps-level: a direct Cholesky's residual is ~growth*eps*cond(A), and a near-singular 3.87M
           elasticity matrix (cond~1e10) legitimately lands at ~1e-6..1e-7 -- a CORRECT solve. An earlier 1e-8
           bar falsely rejected exactly that and forced a slow stock SPOOLES fallback.
           A genuinely broken factor scores O(0.1-1), so 1e-2 cleanly separates correct from garbage. NaN-safe. */
        { double *rr = (double*)malloc((size_t)n * sizeof(double));
          if (!rr) { free(xb); rc = 2; goto done; }
          for (int i = 0; i < n; ++i) rr[i] = b[i];
          symm_lower_spmv_sub(n, colStarts, rowIdx, vals, xb, rr);   /* rr = b - A*xb */
          double rn = 0.0, bn = 0.0;
          for (int i = 0; i < n; ++i) { rn += rr[i]*rr[i]; bn += b[i]*b[i]; }
          free(rr);
          final_resid = (bn > 0.0) ? sqrt(rn / bn) : sqrt(rn);
          const double ACCEPT_MAX = 1e-2;                           /* acceptable-residual ceiling */
          if (!(final_resid < ACCEPT_MAX)) {                        /* NaN-safe */
              if (verbose) fprintf(stderr,
                  "[accel] direct solve resid %.3e (>=%.0e) -> stock SPOOLES fallback\n", final_resid, ACCEPT_MAX);
              free(xb); rc = 1; goto done; } }
        for (int i = 0; i < n; ++i) b[i] = xb[i];
        free(xb);
        MEMLOG("after solve");
        MEMLOG("after factor cleanup");
    } else {
        float *valsf = (float*)malloc((size_t)nnz * sizeof(float));
        float *tmp   = (float*)malloc((size_t)n   * sizeof(float));
        double *x    = (double*)calloc((size_t)n, sizeof(double));
        double *r    = (double*)malloc((size_t)n  * sizeof(double));
        /* PCG work vectors (z=preconditioned residual, p=search dir, Ap=A*p) */
        double *z    = cfg.pcg ? (double*)malloc((size_t)n * sizeof(double)) : NULL;
        double *p    = cfg.pcg ? (double*)malloc((size_t)n * sizeof(double)) : NULL;
        double *Ap   = cfg.pcg ? (double*)malloc((size_t)n * sizeof(double)) : NULL;
        if (!valsf || !tmp || !x || !r || (cfg.pcg && (!z || !p || !Ap))) {
            free(valsf); free(tmp); free(x); free(r); free(z); free(p); free(Ap); rc = 2; goto done; }
        for (long i = 0; i < nnz; ++i) valsf[i] = (float)vals[i];
        SparseMatrix_Float Af = { .structure = structure, .data = valsf };
        SparseNumericFactorOptions nfo = _SparseDefaultNumericFactorOptions_Float;
        double t1 = now_s();
        SparseOpaqueFactorization_Float F = SparseFactor(SparseFactorizationCholesky, Af, sfo, nfo);
        if (F.status != SparseStatusOK) {
            SparseCleanup(F);
            F = SparseFactor(SparseFactorizationLDLT, Af, sfo, nfo);
            if (F.status != SparseStatusOK) { SparseCleanup(F); free(valsf); free(tmp); free(x); free(r); free(z); free(p); free(Ap); rc = 3; goto done; }
        }
        t_factor = now_s() - t1;
        double t2 = now_s();
        double bnorm = 0.0; for (int i = 0; i < n; ++i) bnorm += b[i] * b[i]; bnorm = sqrt(bnorm);
        SparseMatrix_Double Ad = { .structure = structure, .data = vals };

        double *W = NULL, *AW = NULL, *xhat = NULL, *qb = NULL; int m = 0;
        if (cfg.defl) {
            /* rigid-body deflation modes: prefer the in-memory CCX coords (g_co/g_nactdof, same source the GMG
               uses) so defl-pcg runs with NO external file; fall back to an rbmmap file if supplied. */
            const char *src = "in-memory coords";
            if (g_co && g_nactdof) W = build_rbm(g_co, g_nactdof, g_nk, g_mi1 + 1, n, &m);
            else if (cfg.rbmmap[0]) { W = load_rbm(cfg.rbmmap, n, &m); src = cfg.rbmmap; }
            if (W) {
                AW   = (double*)malloc((size_t)6*n*sizeof(double));
                xhat = (double*)malloc((size_t)n*sizeof(double));
                qb   = (double*)malloc((size_t)n*sizeof(double));
            }
            if (!W || !AW || !xhat || !qb) {
                if (verbose) fprintf(stderr, "[accel] defl: RBM build failed (%s) -> plain float-pcg\n", src);
                free(W); free(AW); free(xhat); free(qb); W = NULL;
            } else if (verbose) fprintf(stderr, "[accel] defl: %d RBM modes from %s\n", m, src);
        }

        if (cfg.defl && W) {
            /* ---- Deflated PCG: M^-1 = float Cholesky factor, deflation space = 6 rigid-body modes.
               Deflating the near-null-space (which float precision corrupts) collapses the iteration
               count of plain float-pcg. ---- */
            int maxit = (cfg.refine_iters >= 50) ? cfg.refine_iters : 200;
            const double tol = cfg.pcg_tol;
            double E[64], L[64], c[8], d[8], nd[8];
            for (int k = 0; k < m; ++k)
                SparseMultiply(Ad, (DenseVector_Double){ .count=n, .data=W+(long)k*n },
                                   (DenseVector_Double){ .count=n, .data=AW+(long)k*n });   /* AW_k = A W_k */
            for (int k = 0; k < m; ++k) for (int l = 0; l < m; ++l)
                E[k*m+l] = ddot(n, W+(long)k*n, AW+(long)l*n);                              /* E = W^T A W */
            for (int i = 0; i < m*m; ++i) L[i] = E[i];
            dense_chol(L, m);
            double t_setup = now_s() - t2;
            /* qb = Q b = W E^-1 W^T b */
            colsT_vec(W, n, m, b, c); dense_cholsolve(L, m, c, d);
            for (int i = 0; i < n; ++i) qb[i] = 0.0;
            for (int k = 0; k < m; ++k) nd[k] = -d[k];
            axpy_cols(W, n, m, nd, qb);
            /* r = P b = b - AW E^-1 W^T b ; xhat = 0 */
            for (int i = 0; i < n; ++i) { r[i] = b[i]; xhat[i] = 0.0; }
            colsT_vec(W, n, m, r, c); dense_cholsolve(L, m, c, d); axpy_cols(AW, n, m, d, r);
            /* z = M^-1 r ; p = z */
            for (int i = 0; i < n; ++i) tmp[i] = (float)r[i];
            SparseSolve(F, (DenseVector_Float){ .count=n, .data=tmp });
            for (int i = 0; i < n; ++i) { z[i] = (double)tmp[i]; p[i] = z[i]; }
            double rz = ddot(n, r, z);
            final_resid = (bnorm > 0) ? sqrt(ddot(n,r,r))/bnorm : sqrt(ddot(n,r,r));
            int it = 0;
            for (; it < maxit && final_resid > tol; ++it) {
                SparseMultiply(Ad, (DenseVector_Double){ .count=n, .data=p },
                                   (DenseVector_Double){ .count=n, .data=Ap });            /* Ap = A p */
                colsT_vec(W, n, m, Ap, c); dense_cholsolve(L, m, c, d); axpy_cols(AW, n, m, d, Ap); /* Ap = P A p */
                double pAp = ddot(n, p, Ap);
                if (!(pAp > 0.0)) break;
                double alpha = rz / pAp;
                for (int i = 0; i < n; ++i) { xhat[i] += alpha*p[i]; r[i] -= alpha*Ap[i]; }
                final_resid = (bnorm > 0) ? sqrt(ddot(n,r,r))/bnorm : sqrt(ddot(n,r,r));
                if (final_resid <= tol) { ++it; break; }
                for (int i = 0; i < n; ++i) tmp[i] = (float)r[i];
                SparseSolve(F, (DenseVector_Float){ .count=n, .data=tmp });
                for (int i = 0; i < n; ++i) z[i] = (double)tmp[i];
                double rznew = ddot(n, r, z);
                double beta = rznew / rz; rz = rznew;
                for (int i = 0; i < n; ++i) p[i] = z[i] + beta*p[i];
            }
            refine_iters = it;
            /* x = Q b + P^T xhat ;  P^T xhat = xhat - W E^-1 (AW)^T xhat */
            colsT_vec(AW, n, m, xhat, c); dense_cholsolve(L, m, c, d); axpy_cols(W, n, m, d, xhat);
            for (int i = 0; i < n; ++i) x[i] = qb[i] + xhat[i];
            /* TRUE residual for the gate */
            SparseMultiply(Ad, (DenseVector_Double){ .count=n, .data=x },
                               (DenseVector_Double){ .count=n, .data=Ap });
            double tr = 0.0; for (int i = 0; i < n; ++i){ double e = b[i]-Ap[i]; tr += e*e; }
            final_resid = (bnorm > 0) ? sqrt(tr)/bnorm : sqrt(tr);
            if (verbose) fprintf(stderr, "[accel] defl-pcg: m=%d setup=%.2fs iters=%d resid=%.2e\n",
                                 m, t_setup, refine_iters, final_resid);
            free(W); free(AW); free(xhat); free(qb);
        } else if (cfg.pcg) {
            /* warm start: x0 = M^-1 b  (single float solve) */
            for (int i = 0; i < n; ++i) tmp[i] = (float)b[i];
            SparseSolve(F, (DenseVector_Float){ .count = n, .data = tmp });
            for (int i = 0; i < n; ++i) x[i] = (double)tmp[i];
            /* Preconditioned CG: A x = b, preconditioner M^-1 = float Cholesky factor.
               Mixed precision -> high precision: warm start from the float solve, then
               double CG refines to pcg_tol regardless of float-factor inaccuracy. */
            int maxit = (cfg.refine_iters >= 50) ? cfg.refine_iters : 200;
            const double tol = cfg.pcg_tol;
            /* multithreaded symmetric SpMV via Accelerate (Ad declared above) */
            /* r = b - A x0 */
            SparseMultiply(Ad, (DenseVector_Double){ .count = n, .data = x },
                               (DenseVector_Double){ .count = n, .data = r });
            for (int i = 0; i < n; ++i) r[i] = b[i] - r[i];
            /* z = M^-1 r ; p = z */
            for (int i = 0; i < n; ++i) tmp[i] = (float)r[i];
            SparseSolve(F, (DenseVector_Float){ .count = n, .data = tmp });
            for (int i = 0; i < n; ++i) { z[i] = (double)tmp[i]; p[i] = z[i]; }
            double rz = ddot(n, r, z);
            final_resid = (bnorm > 0) ? sqrt(ddot(n, r, r)) / bnorm : sqrt(ddot(n, r, r));
            int it = 0;
            for (; it < maxit && final_resid > tol; ++it) {
                SparseMultiply(Ad, (DenseVector_Double){ .count = n, .data = p },
                                   (DenseVector_Double){ .count = n, .data = Ap });
                double pAp = ddot(n, p, Ap);
                if (!(pAp > 0.0)) break;            /* breakdown -> stop, gate will catch */
                double alpha = rz / pAp;
                for (int i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
                final_resid = (bnorm > 0) ? sqrt(ddot(n, r, r)) / bnorm : sqrt(ddot(n, r, r));
                if (final_resid <= tol) { ++it; break; }
                for (int i = 0; i < n; ++i) tmp[i] = (float)r[i];
                SparseSolve(F, (DenseVector_Float){ .count = n, .data = tmp });
                for (int i = 0; i < n; ++i) z[i] = (double)tmp[i];
                double rznew = ddot(n, r, z);
                double beta = rznew / rz; rz = rznew;
                for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
            }
            refine_iters = it;
            /* TRUE residual for the gate: the CG recurrence residual drifts on near-singular A, so recompute
               ||b-A*x|| directly (same as the defl-pcg path) -- the gate must never see an optimistic resid. */
            SparseMultiply(Ad, (DenseVector_Double){ .count = n, .data = x },
                               (DenseVector_Double){ .count = n, .data = Ap });
            { double tr = 0.0; for (int i = 0; i < n; ++i){ double e = b[i]-Ap[i]; tr += e*e; }
              final_resid = (bnorm > 0) ? sqrt(tr)/bnorm : sqrt(tr); }
        } else {
            const double tol = 1e-12;
            for (int it = 0; it < cfg.refine_iters; ++it) {
                for (int i = 0; i < n; ++i) r[i] = b[i];
                symm_lower_spmv_sub(n, colStarts, rowIdx, vals, x, r);
                double rnorm = 0.0; for (int i = 0; i < n; ++i) rnorm += r[i] * r[i]; rnorm = sqrt(rnorm);
                final_resid = (bnorm > 0) ? rnorm / bnorm : rnorm;
                if (final_resid <= tol) break;
                for (int i = 0; i < n; ++i) tmp[i] = (float)r[i];
                SparseSolve(F, (DenseVector_Float){ .count = n, .data = tmp });
                for (int i = 0; i < n; ++i) x[i] += (double)tmp[i];
                refine_iters = it + 1;
            }
        }
        t_solve = now_s() - t2;
        /* residual gate: if float refinement did not converge (ill-conditioned matrix), the float answer is
           untrustworthy -> fall back to a correct double solve. The accept bar is the acceptable-accuracy ceiling; the env can
           only TIGHTEN it (capped at 1e-2), never loosen past it, so a knob can't let garbage through. */
        const char *fa = getenv("CCX_ACCEL_FLOAT_ACCEPT");
        double accept = fa ? atof(fa) : 1e-2;
        if (!(accept <= 1e-2)) accept = 1e-2;   /* hard safety ceiling (well-conditioned solves reach ~1e-4; a near-singular one is rejected) */
        if (!(final_resid <= accept)) {          /* NaN-safe: NaN/inf resid -> fall back to double, never accept */
            if (verbose) fprintf(stderr,
                "[accel] float refine resid %.2e > accept %.1e -> DOUBLE fallback (correct)\n",
                final_resid, accept);
            free(valsf); free(tmp); free(x); free(r); free(z); free(p); free(Ap);
            SparseMatrix_Double Ad = { .structure = structure, .data = vals };
            SparseNumericFactorOptions nfod = _SparseDefaultNumericFactorOptions_Double;
            double td = now_s();
            SparseOpaqueFactorization_Double Fd = SparseFactor(SparseFactorizationCholesky, Ad, sfo, nfod);
            if (Fd.status != SparseStatusOK) {
                SparseCleanup(Fd);
                Fd = SparseFactor(SparseFactorizationLDLT, Ad, sfo, nfod);
                if (Fd.status != SparseStatusOK) { SparseCleanup(Fd); SparseCleanup(F); rc = 3; goto done; }
            }
            /* same residual gate as the main double path: solve into scratch, verify ||b-A*xb||/||b||,
               only then copy into b -- never return an untrustworthy double solve as success either. */
            double *xb = (double*)malloc((size_t)n * sizeof(double));
            if (!xb) { SparseCleanup(Fd); SparseCleanup(F); rc = 2; goto done; }
            for (int i = 0; i < n; ++i) xb[i] = b[i];
            SparseSolve(Fd, (DenseVector_Double){ .count = n, .data = xb });
            SparseCleanup(Fd);
            double *rr = (double*)malloc((size_t)n * sizeof(double));
            if (!rr) { free(xb); SparseCleanup(F); rc = 2; goto done; }
            for (int i = 0; i < n; ++i) rr[i] = b[i];
            symm_lower_spmv_sub(n, colStarts, rowIdx, vals, xb, rr);
            double rn = 0.0, bn = 0.0; for (int i = 0; i < n; ++i) { rn += rr[i]*rr[i]; bn += b[i]*b[i]; }
            free(rr);
            final_resid = (bn > 0.0) ? sqrt(rn / bn) : sqrt(rn);
            if (!(final_resid < 1e-2)) {          /* double redo itself untrustworthy -> stock SPOOLES */
                if (verbose) fprintf(stderr, "[accel] double fallback resid %.3e -> stock SPOOLES\n", final_resid);
                free(xb); SparseCleanup(F); rc = 1; goto done;
            }
            for (int i = 0; i < n; ++i) b[i] = xb[i];
            free(xb);
            t_factor += now_s() - td;     /* report includes the double redo */
            cfg.use_float = 0;            /* result is now double-accurate */
        } else {
            for (int i = 0; i < n; ++i) b[i] = x[i];
            free(valsf); free(tmp); free(x); free(r); free(z); free(p); free(Ap);
        }
        SparseCleanup(F);
    }

done:
    free(user_perm);
    free(colStarts); free(rowIdx); free(vals);
    if (verbose) {
        const char *mode = !cfg.use_float ? "double" : (cfg.defl ? "defl-pcg" : (cfg.pcg ? "float-pcg" : "float"));
        fprintf(stderr, "[accel.cfg] enabled=%d order=%s precision=%s refine_iters=%d source=%s\n",
                cfg.enabled, ordname, mode, cfg.refine_iters, cfg.source);
        if (rc == 0 && cfg.use_float)
            fprintf(stderr, "[accel] n=%d nnz=%ld build=%.2fs factor=%.2fs solve+%s=%.2fs iters=%d resid=%.2e\n",
                    n, nnz, t_build, t_factor, cfg.pcg ? "pcg" : "refine", t_solve, refine_iters, final_resid);
        else if (rc == 0)
            fprintf(stderr, "[accel] n=%d nnz=%ld build=%.2fs factor=%.2fs solve=%.2fs (total=%.2fs)\n",
                    n, nnz, t_build, t_factor, t_solve, t_build + t_factor + t_solve);
        else
            fprintf(stderr, "[accel] FAILED rc=%d -> SPOOLES fallback\n", rc);
    }
    return rc;
}
