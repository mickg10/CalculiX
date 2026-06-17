/* ccx_zreader.c — transparent streaming reader for plain / gzip / zstd input files. Tooling / APHYSICAL.
 *
 * Lets CalculiX read compressed .inp and *INCLUDE decks (large meshes) WITHOUT decompressing to disk first
 * — important on space-constrained hosts (a 121 MB row236 deck is ~6-10 MB gz/zst). Used by readinput.c via
 * the CCX_FOPEN/CCX_FGETS/CCX_FCLOSE macros (drop-in fopen/fgets/fclose semantics), only under -DCCX_ACCEL.
 *
 * Detection is by magic bytes (gzip 1f 8b, zstd 28 b5 2f fd); ccx_zopen also tries <path>.gz then <path>.zst
 * when <path> itself is absent, so `ccx -i job` transparently picks up job.inp.gz / job.inp.zst.
 *
 * gz via zlib (-lz); zst via libzstd streaming (-lzstd). ccx_zgets matches fgets: up to n-1 chars or through
 * the next '\n', NUL-terminated, returns buf or NULL at EOF.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include <zstd.h>

typedef enum { CZ_PLAIN, CZ_GZ, CZ_ZST } cz_mode;

struct ccx_zfile {
    cz_mode mode;
    FILE   *fp;                 /* PLAIN + ZST (compressed source)            */
    gzFile  gz;                 /* GZ                                          */
    ZSTD_DStream *ds;           /* ZST streaming state                         */
    unsigned char *inbuf;  size_t in_cap,  in_size,  in_pos;   /* compressed   */
    unsigned char *outbuf; size_t out_cap, out_size, out_pos;  /* decompressed */
    int eof;                    /* compressed input fully read                 */
};
typedef struct ccx_zfile ccx_zfile;

ccx_zfile *ccx_zopen(const char *path) {
    char used[1200];
    FILE *fp = fopen(path, "rb");
    if (fp) { snprintf(used, sizeof used, "%s", path); }
    else {
        snprintf(used, sizeof used, "%s.gz", path);  fp = fopen(used, "rb");
        if (!fp) { snprintf(used, sizeof used, "%s.zst", path); fp = fopen(used, "rb"); }
        if (!fp) return NULL;
    }
    unsigned char m[4] = {0,0,0,0};
    size_t nm = fread(m, 1, 4, fp);
    cz_mode mode = CZ_PLAIN;
    if (nm >= 2 && m[0] == 0x1f && m[1] == 0x8b) mode = CZ_GZ;
    else if (nm >= 4 && m[0] == 0x28 && m[1] == 0xb5 && m[2] == 0x2f && m[3] == 0xfd) mode = CZ_ZST;

    ccx_zfile *z = (ccx_zfile *)calloc(1, sizeof *z);
    if (!z) { fclose(fp); return NULL; }
    z->mode = mode;

    if (mode == CZ_GZ) {
        fclose(fp);                          /* hand the file to zlib by name */
        z->gz = gzopen(used, "rb");
        if (!z->gz) { free(z); return NULL; }
    } else if (mode == CZ_ZST) {
        fseek(fp, 0, SEEK_SET);
        z->fp = fp;
        z->ds = ZSTD_createDStream();
        z->in_cap  = ZSTD_DStreamInSize();
        z->out_cap = ZSTD_DStreamOutSize();
        z->inbuf   = (unsigned char *)malloc(z->in_cap);
        z->outbuf  = (unsigned char *)malloc(z->out_cap);
        if (!z->ds || !z->inbuf || !z->outbuf) {
            if (z->ds) ZSTD_freeDStream(z->ds);
            free(z->inbuf); free(z->outbuf); fclose(fp); free(z); return NULL;
        }
        ZSTD_initDStream(z->ds);
    } else {
        fseek(fp, 0, SEEK_SET);
        z->fp = fp;
    }
    return z;
}

char *ccx_zgets(char *buf, int n, ccx_zfile *z) {
    if (!z || n <= 0) return NULL;
    if (z->mode == CZ_PLAIN) return fgets(buf, n, z->fp);
    if (z->mode == CZ_GZ)    return gzgets(z->gz, buf, n);

    /* CZ_ZST: assemble one line out of the streaming-decompressed buffer */
    int i = 0;
    for (;;) {
        while (z->out_pos < z->out_size && i < n - 1) {
            char c = (char)z->outbuf[z->out_pos++];
            buf[i++] = c;
            if (c == '\n') { buf[i] = '\0'; return buf; }
        }
        if (i >= n - 1) { buf[i] = '\0'; return buf; }      /* line longer than buffer (fgets behaviour) */

        /* output drained -> decompress more */
        if (z->in_pos >= z->in_size && !z->eof) {           /* refill compressed input */
            z->in_size = fread(z->inbuf, 1, z->in_cap, z->fp);
            z->in_pos = 0;
            if (z->in_size == 0) z->eof = 1;
        }
        if (z->in_pos >= z->in_size && z->eof) {            /* no more input */
            if (i > 0) { buf[i] = '\0'; return buf; }        /* final line without newline */
            return NULL;                                     /* true EOF */
        }
        ZSTD_inBuffer  in  = { z->inbuf,  z->in_size, z->in_pos };
        ZSTD_outBuffer out = { z->outbuf, z->out_cap, 0 };
        size_t r = ZSTD_decompressStream(z->ds, &out, &in);
        z->in_pos   = in.pos;
        z->out_size = out.pos;
        z->out_pos  = 0;
        if (ZSTD_isError(r)) {                              /* corrupt stream */
            fprintf(stderr, "[accel] ccx_zreader: zstd decode error: %s\n", ZSTD_getErrorName(r));
            if (i > 0) { buf[i] = '\0'; return buf; }
            return NULL;
        }
    }
}

void ccx_zclose(ccx_zfile *z) {
    if (!z) return;
    if (z->mode == CZ_GZ) { if (z->gz) gzclose(z->gz); }
    else                  { if (z->fp) fclose(z->fp); }
    if (z->ds) ZSTD_freeDStream(z->ds);
    free(z->inbuf); free(z->outbuf); free(z);
}
