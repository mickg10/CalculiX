/* ccx_fastnum.c — fast integer/float field parsers for the input deck. Tooling / APHYSICAL.
 *
 * CalculiX parses NODE/ELEMENT bulk data with Fortran internal reads (read(textpart(k),'(f20.0)') x), ~16M
 * of them for a 1.3M-node / 1.2M-element deck (~5.5s, serial -- the format machinery per number is the cost).
 * These C helpers (strtol/strtod on the comma-split, space-padded textpart field) replace them, called from
 * nodes.f/elements.f only under -DCCX_ACCEL. Semantics match the Fortran reads: istat>0 iff no number parsed.
 * Equivalent for the deck's tokens (no embedded blanks -- fields are already comma-split), incl E-notation and
 * the f20.0 "implied integer" case (strtod("15")==15.0). Fortran passes the hidden string length last.
 */
#include <stdlib.h>
#include <string.h>

void ccxftoi_(const char *s, int *v, int *istat, long slen) {
    char b[140];
    long n = (slen < 139) ? slen : 139;
    if (n < 0) n = 0;
    memcpy(b, s, (size_t)n); b[n] = '\0';
    char *p = b; while (*p == ' ') p++;
    char *e; long x = strtol(p, &e, 10);
    *istat = (e == p) ? 1 : 0;     /* nothing parsed -> error, like the Fortran read */
    *v = (int)x;
}

void ccxftof_(const char *s, double *v, int *istat, long slen) {
    char b[140];
    long n = (slen < 139) ? slen : 139;
    if (n < 0) n = 0;
    memcpy(b, s, (size_t)n); b[n] = '\0';
    char *p = b; while (*p == ' ') p++;
    char *e; double x = strtod(p, &e);
    *istat = (e == p) ? 1 : 0;
    *v = x;
}
