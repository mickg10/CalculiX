/* ccx_fastnum.c — fast integer/float field parsers for the input deck.
 *
 * CalculiX parses NODE/ELEMENT bulk data with Fortran internal reads (read(textpart(k),'(f20.0)') x), ~16M
 * of them on a 1.3M-node / 1.2M-element deck. These C helpers (strtol/strtod on the comma-split, space-padded
 * textpart field) replace them, called from nodes.f/elements.f/splitline.f only under -DCCX_FAST_PARSE (build
 * with FAST_PARSE_DEF= to A/B the stock Fortran reads). Measured front-end win ~4s on a large deck.
 *
 * Equivalence to the Fortran reads (validated bit-exact by dev/parse_tests.sh): the i10/f20.0 COLUMN WIDTH is
 * enforced (only the first 10 / 20 columns are read, exactly as read(textpart(k)(1:10),'(i10)') /
 * (1:20),'(f20.0)' do); blank field -> 0, no error; E-notation, Fortran D-exponent ("1.5D3") and bare-sign
 * exponent ("1.5+3"==1500); the f20.0 "implied integer" case (strtod("15")==15.0); trailing junk -> error.
 * Fortran passes the hidden string length of textpart(k) (132) last; the width cap is applied here.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>

/* These parsers write a 4-byte int into CalculiX's Fortran integer/ITG slots (node/element ids, kon). That is
   only valid for the default i4 build; an i8 build (-DLONGLONG / -fdefault-integer-8) has 8-byte integers, so
   the write would corrupt them and the INT_MAX overflow check would reject legitimately large ids. Trip the
   build, mirroring the accel backend guard. The fast parser is only reached under -DCCX_FAST_PARSE. */
#ifdef LONGLONG
#error "the fast deck parser requires the default i4 CalculiX build (ITG==int); incompatible with -DLONGLONG (i8)."
#endif

void ccxftoi_(const char *s, int *v, int *istat, long slen) {
    char b[16];
    long n = (slen < 10) ? slen : 10;   /* i10: the Fortran read consumes only the first 10 columns */
    if (n < 0) n = 0;
    memcpy(b, s, (size_t)n); b[n] = '\0';
    char *p = b; while (*p == ' ') p++;
    if (*p == '\0') { *v = 0; *istat = 0; return; }  /* all-blank field -> 0, no error (matches Fortran i10) */
    char *e; errno = 0; long x = strtol(p, &e, 10);
    if (e == p) { *v = 0; *istat = 1; return; }      /* nothing parsed -> error, like the Fortran read */
    while (*e == ' ') e++;
    /* error on trailing junk, or on a value that overflows a default (32-bit) integer -- the i10 Fortran read
       flags both via iostat (a 10-digit field can exceed INT_MAX even after the width cap). */
    if (*e != '\0' || errno == ERANGE || x < INT_MIN || x > INT_MAX) { *v = 0; *istat = 1; return; }
    *istat = 0; *v = (int)x;
}

void ccxftof_(const char *s, double *v, int *istat, long slen) {
    char b[32];                                        /* 20 cols + room for an inserted 'e' */
    long n = (slen < 20) ? slen : 20;                  /* f20.0: the Fortran read consumes only the first 20 columns */
    if (n < 0) n = 0;
    memcpy(b, s, (size_t)n); b[n] = '\0';
    char *p = b; while (*p == ' ') p++;
    if (*p == '\0') { *v = 0.0; *istat = 0; return; }  /* all-blank field -> 0.0, no error (matches Fortran f20.0) */
    for (char *q = p; *q; q++) if (*q == 'D' || *q == 'd') *q = 'e';  /* Fortran D-exponent -> C e-exponent */
    { char *q = p; if (*q == '+' || *q == '-') q++;    /* Fortran bare-sign exponent "1.5+3"==1500 -> "1.5e+3" */
      while ((*q >= '0' && *q <= '9') || *q == '.') q++;
      if (*q == '+' || *q == '-') { memmove(q + 1, q, strlen(q) + 1); *q = 'e'; } }
    char *e; double x = strtod(p, &e);
    if (e == p) { *v = 0.0; *istat = 1; return; }      /* nothing parsed -> error, like the Fortran read */
    while (*e == ' ') e++;
    *istat = (*e == '\0') ? 0 : 1;                     /* trailing junk after the float -> error, like f20.0 */
    *v = x;
}

/* ccxsplit — byte-exact C port of splitline.f: split the input line `text` into `*np` comma-separated
 * fields in `textpart` (16 fields, each `ltp`=132 chars, left-justified, space-padded). A space terminates
 * the line (CalculiX convention: data fields have no embedded blanks). Replaces the per-char Fortran
 * substring loop (the #2 front-end hotspot, ~2.5M calls on a large deck). Fortran passes the hidden
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
