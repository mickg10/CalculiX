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
#include <stdio.h>

void ccxftoi_(const char *s, int *v, int *istat, long slen) {
    char b[140];
    long n = (slen < 139) ? slen : 139;
    if (n < 0) n = 0;
    memcpy(b, s, (size_t)n); b[n] = '\0';
    char *p = b; while (*p == ' ') p++;
    if (*p == '\0') { *v = 0; *istat = 0; return; }  /* all-blank field -> 0, no error (matches Fortran i10) */
    char *e; long x = strtol(p, &e, 10);
    *istat = (e == p) ? 1 : 0;     /* non-numeric -> error, like the Fortran read */
    *v = (int)x;
}

void ccxftof_(const char *s, double *v, int *istat, long slen) {
    char b[140];
    long n = (slen < 139) ? slen : 139;
    if (n < 0) n = 0;
    memcpy(b, s, (size_t)n); b[n] = '\0';
    char *p = b; while (*p == ' ') p++;
    if (*p == '\0') { *v = 0.0; *istat = 0; return; }  /* all-blank field -> 0.0, no error (matches Fortran f20.0) */
    char *e; double x = strtod(p, &e);
    *istat = (e == p) ? 1 : 0;
    *v = x;
}

/* ccxsplit — byte-exact C port of splitline.f: split the input line `text` into `*np` comma-separated
 * fields in `textpart` (16 fields, each `ltp`=132 chars, left-justified, space-padded). A space terminates
 * the line (CalculiX convention: data fields have no embedded blanks). Replaces the per-char Fortran
 * substring loop (the #2 front-end hotspot, ~2.5M calls on the row236 deck). Fortran passes the hidden
 * char lengths last: lt = len(text) = 1320, ltp = element len(textpart) = 132. Called from splitline.f
 * only under -DCCX_FAST_PARSE; produces identical textpart bytes -> deck parse stays bit-exact. */
void ccxsplit_(const char *text, char *textpart, int *np, long lt, long ltp) {
    const long FW = ltp;            /* field width (132) */
    int n = 1;                      /* current field, 1-based */
    long j = 0;                     /* chars placed in current field */
    long i;
    for (i = 0; i < lt; i++) {
        char c = text[i];
        if (c != ',') {
            if (c == ' ') break;    /* space terminates the line */
            j++;
            if (j <= FW) textpart[(long)(n - 1) * FW + (j - 1)] = c;
        } else {
            for (long k = j; k < FW; k++) textpart[(long)(n - 1) * FW + k] = ' ';  /* pad field */
            j = 0;
            n++;
            if (n > 16) {           /* >16 entries: match splitline.f error handling exactly */
                int ierror = 0;
                for (long k = i + 1; k < lt; k++) {
                    if (text[k] == ',') continue;
                    if (text[k] == ' ') {
                        if (ierror == 0) break;
                        fprintf(stderr, "*ERROR in splitline: there should not\n"
                                        "       be more than 16 entries in a \n       line; \n");
                        exit(201);
                    }
                    ierror = 1;
                }
                break;
            }
        }
    }
    if (j == 0) {
        n = n - 1;
    } else {
        for (long k = j; k < FW; k++) textpart[(long)(n - 1) * FW + k] = ' ';
    }
    for (int f = n; f < 16; f++) {  /* clear unused fields n+1..16 (1-based) to spaces */
        char *pf = textpart + (long)f * FW;
        for (long k = 0; k < FW; k++) pf[k] = ' ';
    }
    *np = n;
}
