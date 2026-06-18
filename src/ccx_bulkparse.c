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

extern void ccxsplit_(const char *, char *, int *, long, long);

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

/* ===== *NODE (no NSET=): id + up to 3 coords, exactly as nodes.f ===== */
void ccxbulk_nodes_(char *inpc, double *co, int *nk, const int *nk_, int *ier,
                    int *iline, int *ipol, int *inl, const int *ipoinp, const int *inp,
                    const int *ipoinpc, char *textpart, int *n, int *key, int *istat,
                    long inpc_len, long tp_len) {
    char buf[1400]; char *fs[20];
    (void)inpc_len;
    for (;;) {
        if (bp_advance(iline, ipol, inl, ipoinp, inp) < 0) { *istat = -1; return; }
        long L = bp_line(inpc, ipoinpc, *iline, buf, sizeof buf);
        if (buf[0] == '*' && buf[1] != '*') { bp_handback(buf, L, textpart, n, key, istat, tp_len); return; }
        int nf = bp_tok(buf, fs, 16);
        char *p = fs[0]; while (*p == ' ') p++;
        char *e; long id = *p ? strtol(p, &e, 10) : 0;     /* blank id -> 0 (Fortran i10) */
        if (*p && e == p) { fprintf(stderr, "*ERROR reading *NODE\n"); *ier = 1; return; }
        int i = (int)id;
        if (i > *nk) *nk = i;
        if (*nk > *nk_) { fprintf(stderr, "*ERROR reading *NODE: increase nk_\n"); *ier = 1; return; }
        double *c = co + (long)(i - 1) * 3;
        c[0] = (nf >= 2) ? bp_d(fs[1]) : 0.0;
        c[1] = (nf >= 3) ? bp_d(fs[2]) : 0.0;
        c[2] = (nf >= 4) ? bp_d(fs[3]) : 0.0;
    }
}
