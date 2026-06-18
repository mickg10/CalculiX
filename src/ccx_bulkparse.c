/* ccx_bulkparse.c — bulk C parser for the *NODE / *ELEMENT data blocks. Tooling / APHYSICAL.
 *
 * The per-line Fortran path (getnewline copies inpc->text, splitline tokenizes text->textpart[16][132],
 * then nodes.f/elements.f read each field) costs ~5.9s on the row236 deck (2.5M lines). These routines do the
 * whole block in C: replicate getnewline's line-advance over the inp/ipoinp/ipoinpc structure, copy each line
 * once into a NUL-terminated buffer, tokenize in place (splitline.f-exact n) and parse with strtol/strtod.
 * They reproduce nodes.f/elements.f byte-for-byte for the common case (validated by dev/parse_integration_tests.sh)
 * and are called only under -DCCX_FAST_PARSE; callers fall back to the Fortran loop for the cases handled there
 * but not here (e.g. *NODE with NSET=, *ELEMENT user/substructure types).
 *
 * On return the routine leaves the terminating keyword line in (textpart,n,key=1,iline,ipol,inl) exactly as a
 * getnewline that found the next card would, so calinput's dispatch loop continues unchanged. istat=-1 on EOF.
 *
 * ITG is int (no -DLONGLONG); Fortran integer == int. Fortran passes hidden char lengths last (inpc=1, tp=132).
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern void ccxsplit_(const char *, char *, int *, long, long);
static double bp_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* front-end stage wall probe (env CCX_FE_TIMING); seconds since first call. diagnostic only. */
static double ccx_fe_t0 = 0.0;
void ccx_festamp(const char *tag) {
    if (!getenv("CCX_FE_TIMING")) return;
    double t = bp_now();
    if (ccx_fe_t0 == 0.0) ccx_fe_t0 = t;
    fprintf(stderr, "[fe] %-14s %8.3f\n", tag, t - ccx_fe_t0);
}

/* runtime toggle (default ON): set CCX_NO_BULK to fall back to the per-line Fortran loop (for A/B timing). */
int ccxbulk_on_(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("CCX_NO_BULK"); v = (e && *e) ? 0 : 1; }
    return v;
}

/* advance to the next line, byte-for-byte as getnewline.f. returns 0 ok / -1 EOF. updates *iline,*ipol,*inl.
 * Fortran 2D arrays are column-major: inp(r,k)=inp[(k-1)*3+(r-1)], ipoinp(r,p)=ipoinp[(p-1)*2+(r-1)]. */
static int bp_advance(int *iline, int *ipol, int *inl, const int *ipoinp, const int *inp) {
    enum { nentries = 19 };
    int il = *iline, ip = *ipol, in = *inl;
    if (il == inp[(in-1)*3 + 1]) {                 /* iline == inp(2,inl) */
        if (inp[(in-1)*3 + 2] == 0) {              /* inp(3,inl) == 0   */
            for (;;) {
                ip++;
                if (ip > nentries) { *ipol = ip; return -1; }
                if (ipoinp[(ip-1)*2 + 0] != 0) break;   /* ipoinp(1,ipol) != 0 */
            }
            in = ipoinp[(ip-1)*2 + 0];             /* inl = ipoinp(1,ipol) */
            il = inp[(in-1)*3 + 0];                /* iline = inp(1,inl)  */
        } else {
            in = inp[(in-1)*3 + 2];                /* inl = inp(3,inl)    */
            il = inp[(in-1)*3 + 0];                /* iline = inp(1,inl)  */
        }
    } else {
        il = il + 1;
    }
    *iline = il; *ipol = ip; *inl = in; return 0;
}

/* tokenize a NUL-terminated line into field-start pointers; n computed exactly like splitline.f
 * (comma separates, a space terminates the line, an empty trailing field is dropped). Writes NULs at commas. */
static int bp_tok(char *buf, char **fs, int maxf) {
    int n = 1, j = 0; char *p = buf; fs[0] = buf;
    for (;;) {
        char c = *p;
        if (c == '\0') break;
        if (c != ',') { if (c == ' ') break; j++; }
        else { *p = '\0'; if (n < maxf) fs[n] = p + 1; n++; j = 0; }
        p++;
    }
    *p = '\0';
    if (j == 0) n--;
    return n;
}

static double bp_d(const char *s) { while (*s == ' ') s++; return *s ? strtod(s, 0) : 0.0; }   /* blank -> 0 */

/* copy line `il` of inpc into buf (NUL-terminated); returns length. */
static long bp_line(const char *inpc, const int *ipoinpc, int il, char *buf, long cap) {
    long a = ipoinpc[il-1], b = ipoinpc[il], L = b - a;
    if (L < 0) L = 0; if (L > cap-1) L = cap-1;
    memcpy(buf, inpc + a, (size_t)L); buf[L] = '\0';
    return L;
}

/* fill (textpart,n,key=1) from a keyword line and return — mirrors getnewline returning the next card. */
static void bp_handback(char *buf, long L, char *textpart, int *n, int *key, int *istat, long tp_len) {
    char t[1320]; memset(t, ' ', sizeof t);
    if (L > 1320) L = 1320; memcpy(t, buf, (size_t)L);
    ccxsplit_(t, textpart, n, 1320, tp_len);
    *key = 1; *istat = 0;
}

/* parse one *NODE data line (id + up to 3 coords) into co[]; returns id, sets *err (1=bad id, 2=out of range). */
static long bp_node_line(const char *inpc, const int *ipoinpc, int il, double *co, int nkmax, int *err) {
    char buf[1400]; char *fs[20];
    bp_line(inpc, ipoinpc, il, buf, sizeof buf);
    int nf = bp_tok(buf, fs, 16);
    char *p = fs[0]; while (*p == ' ') p++;
    char *e; long id = *p ? strtol(p, &e, 10) : 0;          /* blank id -> 0 (Fortran i10) */
    if (*p && e == p) { *err = 1; return 0; }
    int i = (int)id;
    if (i < 1 || i > nkmax) { *err = 2; return id; }        /* out of range -> would OOB; flag (valid decks never hit) */
    double *c = co + (long)(i - 1) * 3;
    c[0] = (nf >= 2) ? bp_d(fs[1]) : 0.0;
    c[1] = (nf >= 3) ? bp_d(fs[2]) : 0.0;
    c[2] = (nf >= 4) ? bp_d(fs[3]) : 0.0;
    return id;
}

/* ===== *NODE (no NSET=): id + up to 3 coords, exactly as nodes.f =====
 * Enumerate the block's data-line numbers (cheap, sequential), then parse them in PARALLEL: each node id is
 * unique so co[] writes are disjoint (no race); nk is a max-reduction. Uses the ~15 cores idle during the
 * otherwise-serial front-end. Leaves the terminating keyword state exactly like getnewline. */
void ccxbulk_nodes_(char *inpc, double *co, int *nk, const int *nk_, int *ier,
                    int *iline, int *ipol, int *inl, const int *ipoinp, const int *inp,
                    const int *ipoinpc, char *textpart, int *n, int *key, int *istat,
                    long inpc_len, long tp_len) {
    (void)inpc_len;
    int cap = 1 << 20, nlines = 0;
    int *ils = (int *)malloc((size_t)cap * sizeof(int));
    for (;;) {
        if (bp_advance(iline, ipol, inl, ipoinp, inp) < 0) { *istat = -1; break; }
        long a = ipoinpc[*iline - 1];
        if (inpc[a] == '*' && inpc[a + 1] != '*') {        /* next keyword -> hand it back, block done */
            char buf[1400]; long L = bp_line(inpc, ipoinpc, *iline, buf, sizeof buf);
            bp_handback(buf, L, textpart, n, key, istat, tp_len); break;
        }
        if (!ils) {                                        /* OOM fallback: parse serially as we go */
            int e2 = 0; long id = bp_node_line(inpc, ipoinpc, *iline, co, *nk_, &e2);
            if ((int)id > *nk) *nk = (int)id;
            if (e2) { *ier = 1; }
            continue;
        }
        if (nlines >= cap) { cap *= 2; ils = (int *)realloc(ils, (size_t)cap * sizeof(int)); }
        ils[nlines++] = *iline;
    }
    if (ils) {
        int maxid = *nk, err = 0; int dbg = getenv("CCX_ACCEL_VERBOSE") != 0; double t0 = dbg ? bp_now() : 0;
        int nth = 1;
        #pragma omp parallel
        {
          #ifdef _OPENMP
          #pragma omp single
          nth = omp_get_num_threads();
          #endif
        }
        #pragma omp parallel for schedule(static) reduction(max:maxid)
        for (int k = 0; k < nlines; k++) {
            int e2 = 0; long id = bp_node_line(inpc, ipoinpc, ils[k], co, *nk_, &e2);
            if ((int)id > maxid) maxid = (int)id;
            if (e2) { err = e2; }                          /* benign race: any nonzero flags an error */
        }
        *nk = maxid;
        if (dbg) fprintf(stderr, "[bulk] *NODE: lines=%d threads=%d parse=%.3fs\n", nlines, nth, bp_now() - t0);
        free(ils);
        if (err) *ier = 1;
    }
    if (*nk > *nk_) { fprintf(stderr, "*ERROR reading *NODE: increase nk_\n"); *ier = 1; }
}
