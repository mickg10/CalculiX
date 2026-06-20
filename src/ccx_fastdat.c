/* ccx_fastdat.c — fast PARALLEL C buffered writer for the *EL PRINT,S stress section of the .dat file.
 * Replaces CalculiX's serial gfortran formatted write (i10,1x,i3,1p,6(1x,e13.6)), which dominates output on
 * a large stress deck (~15s for a ~1 GB .dat). Output is byte-format-identical to gfortran (verified over
 * 200k lines). The line is FIXED-WIDTH: i10(10)+1x(1)+i3(3)+6*(1x(1)+e13.6(13)) = 98 chars + '\n' = 99.
 * That lets us format all lines in parallel at exact offsets, then write in bounded chunks.
 *
 * Gating: active iff env CCX_ACCEL_OUT_DAT_FAST=<jobname.dat>. Per-element calls record (nelem,j,&stx) only;
 * ccx_fastdat_flush_() (called AFTER the Fortran flush(5) of node-print + the stress header) formats every
 * line with OpenMP and appends them.
 *
 * LIMITATION (mixed orientation): only the iorien==0 'S'/'SVF' path is redirected here; oriented stress
 * writes (the trailing-a20-label format) still go through Fortran unit 5 and are flushed BEFORE this buffer
 * is appended. A single *EL PRINT that mixes oriented and non-oriented printed elements would therefore
 * interleave out of order. This accelerator targets unoriented stress decks; for
 * mixed-orientation decks, omit CCX_ACCEL_OUT_DAT_FAST and use CalculiX's standard writer.
 *
 * Failure policy (opt-in tooling): FAIL-CLOSED. This path is requested explicitly via env, so if it cannot
 * deliver the COMPLETE stress block -- record-time OOM, format-buffer OOM, or a write/close error -- it prints
 * a loud diagnostic and exit(202)s. A missing or partial stress block must never be reported to a downstream
 * pipeline as a successful run (exit 0): a partial block looks complete but is wrong, and an absent block with
 * exit 0 looks valid but is incomplete -- both are silent-wrong-output. A line whose value does not fit the
 * fixed e13.6 width (NaN/Inf/|exp|>=100 -- only on a non-physical solution) likewise fails closed, rather
 * than emit the clamped (wrong) value that keeping rows aligned would require.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CCX_DAT_BODY 98   /* characters before the newline */
#define CCX_DAT_LINE 99   /* fixed bytes per stress line incl newline */

static int           *g_nelem = NULL;
static int           *g_j     = NULL;
static const double **g_s     = NULL;   /* pointer to the 6 contiguous stx components (valid until flush) */
static size_t         g_n = 0, g_cap = 0;
static int            g_active = -1;
static int            g_oom = 0;        /* sticky: a record-time alloc failed -> emit nothing, not a partial block */

int ccx_fastdat_active_(void) {
    if (g_active < 0) {
        const char *e = getenv("CCX_ACCEL_OUT_DAT_FAST");
        g_active = (e && *e) ? 1 : 0;
    }
    return g_active;
}

/* record one stress line: nelem (deck #), j (int. point), s -> &stx(1,j,nelel) (6 contiguous doubles). */
void ccx_fastdat_s_(const int *nelem, const int *j, const double *s) {
    if (g_oom) return;                              /* already degraded -> drop; flush warns + writes nothing */
    if (g_n >= g_cap) {
        size_t ncap = g_cap ? g_cap * 2 : ((size_t)1 << 22);
        /* realloc into temps so the live buffers are not leaked if one of the three fails */
        int           *ne = (int *)          realloc(g_nelem, ncap * sizeof(int));
        int           *nj = (int *)          realloc(g_j,     ncap * sizeof(int));
        const double **ns = (const double **)realloc(g_s,     ncap * sizeof(double *));
        if (ne) g_nelem = ne;
        if (nj) g_j     = nj;
        if (ns) g_s     = ns;
        if (!ne || !nj || !ns) {                    /* OOM: stop recording, flag for flush, do NOT exit */
            g_oom = 1;
            fprintf(stderr, "[accel] fast .dat: out of memory while recording (~%zu lines) -> stress block "
                            "will NOT be written; rerun without CCX_ACCEL_OUT_DAT_FAST\n", g_n);
            return;
        }
        g_cap = ncap;
    }
    g_nelem[g_n] = *nelem; g_j[g_n] = *j; g_s[g_n] = s; g_n++;
}

static void ccx_fastdat_reset(void) {
    free(g_nelem); free(g_j); free((void *)g_s);
    g_nelem = NULL; g_j = NULL; g_s = NULL; g_n = 0; g_cap = 0; g_oom = 0;
}

/* parallel-format every recorded line at its exact offset (bounded chunks), append to the .dat; reset. */
void ccx_fastdat_flush_(void) {
    const char *path = getenv("CCX_ACCEL_OUT_DAT_FAST");
    if (!path || !*path) { ccx_fastdat_reset(); return; }
    if (g_oom) {
        fprintf(stderr, "[accel] fast .dat: record-time OOM -> cannot write a complete stress block; failing "
                        "closed (exit 202). Rerun without CCX_ACCEL_OUT_DAT_FAST for complete output.\n");
        ccx_fastdat_reset(); exit(202);
    }
    if (g_n == 0) { ccx_fastdat_reset(); return; }

    FILE *f = fopen(path, "ab");
    if (!f) {                                /* cannot open the .dat -> stress block would be lost (the Fortran
                                                write is bypassed when this path is active) -> fail closed */
        perror("[accel] fast .dat fopen");
        fprintf(stderr, "[accel] fast .dat: cannot open '%s' -> stress block lost; failing closed (exit 202)\n", path);
        ccx_fastdat_reset(); exit(202);
    }
    setvbuf(f, NULL, _IOFBF, 1 << 22);

    /* bounded scratch (no single giant malloc): 1M lines (~99 MB), shrinking if memory is tight. */
    size_t chunk = (size_t)1 << 20;
    char *buf = NULL;
    while (chunk >= 4096 && !(buf = (char *)malloc(chunk * CCX_DAT_LINE))) chunk >>= 1;
    if (!buf) {
        fprintf(stderr, "[accel] fast .dat: cannot allocate format buffer -> incomplete stress block; "
                        "failing closed (exit 202)\n");
        fclose(f); ccx_fastdat_reset(); exit(202);
    }

    long malformed = 0; int werr = 0;
    for (size_t base = 0; base < g_n && !werr; base += chunk) {
        size_t cnt = (g_n - base < chunk) ? (g_n - base) : chunk;
        #pragma omp parallel for schedule(static) reduction(+:malformed)
        for (size_t k = 0; k < cnt; k++) {
            char tmp[160];
            const double *s = g_s[base + k];
            int r = snprintf(tmp, sizeof tmp, "%10d %3d %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E",
                             g_nelem[base + k], g_j[base + k], s[0], s[1], s[2], s[3], s[4], s[5]);
            char *p = buf + k * (size_t)CCX_DAT_LINE;
            /* expected width is exactly CCX_DAT_BODY; only a non-physical value (NaN/Inf/|exp|>=100) deviates.
               Clamp+pad here so the fixed-offset write stays in bounds, and count it -- any deviation makes the
               whole block fail closed after the loop (a clamped line is wrong; never report it as success). */
            int m = (r < 0) ? 0 : (r < CCX_DAT_BODY ? r : CCX_DAT_BODY);
            if (r != CCX_DAT_BODY) malformed++;
            memcpy(p, tmp, (size_t)m);
            for (int q = m; q < CCX_DAT_BODY; q++) p[q] = ' ';
            p[CCX_DAT_BODY] = '\n';
        }
        size_t want = cnt * (size_t)CCX_DAT_LINE;
        if (fwrite(buf, 1, want, f) != want) werr = 1;
    }
    free(buf);
    int cerr = (fclose(f) != 0);
    if (werr || cerr) {
        fprintf(stderr, "[accel] fast .dat: WRITE ERROR (%s) -> .dat stress block incomplete; failing "
                        "closed (exit 202)\n", werr ? "fwrite" : "fclose");
        ccx_fastdat_reset(); exit(202);
    }
    if (malformed) {  /* a value did not fit the fixed e13.6 width (NaN/Inf/|exp|>=100 -- a non-physical
                         solution): the line was clamped to stay row-aligned, so it is now WRONG. Fail closed
                         rather than report a clamped value as success; rerun without CCX_ACCEL_OUT_DAT_FAST
                         for CalculiX's standard writer. (Only reachable on a garbage solve.) */
        fprintf(stderr, "[accel] fast .dat: %ld stress line(s) not representable in the fixed e13.6 width "
                        "(NaN/Inf/huge stress?) -> failing closed (exit 202); rerun without "
                        "CCX_ACCEL_OUT_DAT_FAST and check the solution\n", malformed);
        ccx_fastdat_reset(); exit(202);
    }
    fprintf(stderr, "[accel] fast .dat: wrote %.1f MB stress (%zu lines, parallel C)\n",
            (double)g_n * CCX_DAT_LINE / 1e6, g_n);
    ccx_fastdat_reset();
}
