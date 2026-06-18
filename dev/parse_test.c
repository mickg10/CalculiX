/* parse_test.c — differential red/green tests for the fast C deck parsers vs the Fortran reference.
 * Tooling / test-only. For every input, the C parser (ccxftoi/ccxftof/ccxsplit) must produce output
 * bit-identical to the Fortran reference (ftref_i/ftref_f = the exact read(...) in nodes.f/elements.f;
 * splitline_ = stock splitline.f compiled WITHOUT -DCCX_FAST_PARSE). Hand-corpus + deterministic fuzz.
 * Exit 0 = all green, 1 = any red. Build/run via dev/parse_tests.sh.
 *
 * Equivalence rules (matching how nodes.f/elements.f consume the fields):
 *  - integer: Fortran read(field(1:10),'(i10)') ; success iff istat<=0. Compare error-flag + value.
 *  - real   : Fortran read(field(1:20),'(f20.0)'); compare error-flag + the double BIT-for-bit.
 *  - tokenizer: splitline(text)->(n,textpart[16][132]); compare n + all 16*132 bytes.
 * Note: integer fields >10 digits and D-exponent floats are out of scope (never emitted by valid decks;
 * the i10/f20 width vs whole-field strtol diverge only there) and are excluded from the corpus.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>

extern void ftref_i_(const char*, int*, int*, long);
extern void ftref_f_(const char*, double*, int*, long);
extern void ccxftoi_(const char*, int*, int*, long);
extern void ccxftof_(const char*, double*, int*, long);
extern void splitline_(const char*, char*, int*, long, long);   /* stock Fortran reference */
extern void ccxsplit_(const char*, char*, int*, long, long);    /* C fast tokenizer */

static int g_pass = 0, g_fail = 0, g_shown = 0;
static void fail(const char *cat, const char *in, const char *detail) {
    g_fail++;
    if (g_shown++ < 30) fprintf(stderr, "  RED [%s] input=<%s>  %s\n", cat, in, detail);
}

/* place a token into a width-w field buffer (space-padded), as textpart(k) is presented to the readers */
static void field(char *buf, long w, const char *tok) {
    memset(buf, ' ', (size_t)w);
    long n = (long)strlen(tok); if (n > w) n = w;
    memcpy(buf, tok, (size_t)n);
}

static void test_int(const char *tok) {
    char b[132]; field(b, 132, tok);
    int vr, sr, vc, sc;
    ftref_i_(b, &vr, &sr, 132);
    ccxftoi_(b, &vc, &sc, 132);
    int ref_err = (sr > 0), c_err = (sc != 0);
    char d[160];
    if (ref_err != c_err) { snprintf(d, sizeof d, "err mismatch ref=%d c=%d (sr=%d sc=%d)", ref_err, c_err, sr, sc); fail("int", tok, d); return; }
    if (!ref_err && vr != vc) { snprintf(d, sizeof d, "value ref=%d c=%d", vr, vc); fail("int", tok, d); return; }
    g_pass++;
}

static void test_flt(const char *tok) {
    char b[132]; field(b, 132, tok);
    double vr, vc; int sr, sc;
    ftref_f_(b, &vr, &sr, 132);
    ccxftof_(b, &vc, &sc, 132);
    int ref_err = (sr > 0), c_err = (sc != 0);
    char d[160];
    if (ref_err != c_err) { snprintf(d, sizeof d, "err mismatch ref=%d c=%d", ref_err, c_err); fail("flt", tok, d); return; }
    if (!ref_err) {
        uint64_t ir, ic; memcpy(&ir, &vr, 8); memcpy(&ic, &vc, 8);
        if (ir != ic) {
            long ulp = (long)(ir > ic ? ir - ic : ic - ir);
            snprintf(d, sizeof d, "bits ref=%.17g c=%.17g ulp=%ld", vr, vc, ulp); fail("flt", tok, d); return;
        }
    }
    g_pass++;
}

static void test_split(const char *line) {
    char buf[1320]; memset(buf, ' ', sizeof buf);
    long n = (long)strlen(line); if (n > 1320) n = 1320; memcpy(buf, line, (size_t)n);
    char tpr[16*132], tpc[16*132]; int nr, nc;
    splitline_(buf, tpr, &nr, 1320, 132);
    ccxsplit_(buf, tpc, &nc, 1320, 132);
    char d[256];
    if (nr != nc) { snprintf(d, sizeof d, "n ref=%d c=%d", nr, nc); fail("split", line, d); return; }
    if (memcmp(tpr, tpc, 16*132) != 0) {
        int fi = -1; for (int i = 0; i < 16*132; i++) if (tpr[i] != tpc[i]) { fi = i; break; }
        snprintf(d, sizeof d, "textpart byte %d field %d (ref=%d c=%d)", fi, fi/132, tpr[fi], tpc[fi]);
        fail("split", line, d); return;
    }
    g_pass++;
}

/* fork a child that tokenizes a >16-field line; return its exit status (both impls must exit(201)) */
static int child_split_exit(int use_c) {
    pid_t pid = fork();
    if (pid == 0) {
        char buf[1320]; memset(buf, ' ', sizeof buf);
        const char *L = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18";
        memcpy(buf, L, strlen(L));
        char tp[16*132]; int n;
        if (use_c) ccxsplit_(buf, tp, &n, 1320, 132); else splitline_(buf, tp, &n, 1320, 132);
        _exit(0);  /* if it did NOT exit(201) itself, report 0 */
    }
    int st = 0; waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

/* deterministic LCG so the fuzz corpus is reproducible (red/green must be stable) */
static uint64_t rng = 0x123456789abcdef0ULL;
static unsigned r32(void){ rng = rng*6364136223846793005ULL + 1442695040888963407ULL; return (unsigned)(rng>>33); }
static int rin(int lo,int hi){ return lo + (int)(r32()%(unsigned)(hi-lo+1)); }

static void rand_int_tok(char *o){
    int neg = rin(0,3)==0, ndig = rin(1,9); char *p=o;
    if(neg)*p++='-'; for(int i=0;i<ndig;i++)*p++=(char)('0'+rin(0,9)); *p=0;
}
static void rand_flt_tok(char *o){
    char *p=o; if(rin(0,3)==0)*p++='-';
    int a=rin(0,6); for(int i=0;i<a;i++)*p++=(char)('0'+rin(0,9)); if(a==0)*p++='0';
    if(rin(0,4)){ *p++='.'; int b=rin(0,8); for(int i=0;i<b;i++)*p++=(char)('0'+rin(0,9)); }
    if(rin(0,2)==0){ *p++=(rin(0,1)?'E':'e'); if(rin(0,1))*p++=(rin(0,1)?'+':'-'); int e=rin(1,2); for(int i=0;i<e;i++)*p++=(char)('0'+rin(0,9)); }
    *p=0;
}

int main(void) {
    /* ---- integer corpus ---- */
    /* valid-deck integer fields: positive IDs <=10 digits (non-overflowing) + small negatives + blanks.
     * Out of scope (documented, never emitted by valid decks): >10-digit fields and INT_MIN's 11-char form,
     * where Fortran's i10 reads only the first 10 chars while ccxftoi reads the whole field. */
    const char *ints[] = {"0","1","-1","7","42","100","999","1000","12345","123456","1234567",
        "7654321","2147483647","  5","5  "," 123 ","007","+5","000","999999999",
        "1000000000","-7","   ","",
        "3.5","12x4","99z",                       /* trailing junk -> error (matches i10) */
        NULL};
    for (int i=0; ints[i]; i++) test_int(ints[i]);
    /* ---- float corpus ---- */
    const char *flts[] = {"0.0","1.0","-1.0","1.5","-2.5","3.14159265358979","0.1","0.2","0.3",
        "1.5E-3","1.5e+10","1E10","1e-10",".5","5.","-0.0","100.0","0.001","123456.789",
        "1.234567890123456E2","2.0","-999.999","0.","1.5E3","6.022e23","0.0001234","   ","",
        "1.5D3","-2.5d-2","1.0D-3","6.022D23",     /* Fortran D-exponent -> must match f20.0 */
        NULL};
    for (int i=0; flts[i]; i++) test_flt(flts[i]);
    /* ---- tokenizer corpus (deck-style: no embedded blanks) ---- */
    const char *lines[] = {"1,2,3","12345,1.5,2.5,3.5","100,1.5,-2.5,3.5","1",
        "1,2,3,4,5,6,7,8,9","1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16","1,2,","1,,3","",
        "42,3.14","1,2,3,4,5,6,7,8,9,10","-5,-6.5E-2,7","0,0.0,0.0,0.0",
        "999999,1.234567890123,-9.87654321E1,0.0",NULL};
    for (int i=0; lines[i]; i++) test_split(lines[i]);
    /* tokenizer: a field longer than 132 chars must be capped identically by both impls */
    { char ln[200]; ln[0]='1'; ln[1]=','; memset(ln+2,'7',150); ln[152]=','; ln[153]='3'; ln[154]=0; test_split(ln); }

    /* ---- fuzz ---- */
    int NF = 200000;
    char tok[64];
    for (int i=0;i<NF;i++){ rand_int_tok(tok); test_int(tok); }
    for (int i=0;i<NF;i++){ rand_flt_tok(tok); test_flt(tok); }
    char line[1100];
    for (int i=0;i<NF;i++){
        int nf=rin(1,16); line[0]=0; long off=0;
        for(int f=0;f<nf;f++){ if(f){line[off++]=',';} char t[64]; if(rin(0,1))rand_int_tok(t);else rand_flt_tok(t);
            long tl=(long)strlen(t); memcpy(line+off,t,tl); off+=tl; }
        line[off]=0; test_split(line);
    }

    /* ---- >16-field error path: both must exit(201) ---- */
    int ce = child_split_exit(1), fe = child_split_exit(0);
    if (ce == 201 && fe == 201) g_pass++;
    else { char d[80]; snprintf(d,sizeof d,"exit codes c=%d fortran=%d (want 201/201)",ce,fe); fail("split>16",">16 fields",d); }

    printf("parse_test: %d passed, %d failed (corpus + %d-case fuzz x3 + error-path)\n", g_pass, g_fail, NF);
    return g_fail ? 1 : 0;
}
