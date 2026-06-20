/*     CalculiX - A 3-dimensional finite element program                 */
/*              Copyright (C) 1998-2024 Guido Dhondt                      */
/*     This program is free software; you can redistribute it and/or     */
/*     modify it under the terms of the GNU General Public License as     */
/*     published by the Free Software Foundation(version 2);              */
/*     This program is distributed in the hope that it will be useful,    */
/*     but WITHOUT ANY WARRANTY; see the GNU General Public License.      */

/* ccx_arrowout.c - self-contained Arrow IPC "file" (Feather v2) writer for the nodal solution field.
 * Tooling / APHYSICAL. DEPENDENCY-FREE pure C: a tiny FlatBuffers builder writes the Arrow metadata
 * (Message/Schema/RecordBatch/Footer), so there is NO nanoarrow/flatcc/libarrow to vendor or link -- the
 * file is GPL-clean and avoids the libc++/libstdc++ ABI clash a system libarrow would create.
 *
 * One record batch, fixed primitive schema, no nulls, little-endian. Columns (selected by co/stn != NULL):
 *   node:int32, [x,y,z:f64], ux,uy,uz:f64, [sxx,syy,szz,sxy,sxz,syz:f64].
 * Read zero-copy by pyarrow (pa.feather.read_table), polars (pl.read_ipc), duckdb. Called from frd.c when
 * CCX_ACCEL_OUT_ARROW=<path> is set (build with ARROW=1 -> -DCCX_ACCEL_ARROW).
 *
 * File layout: "ARROW1\0\0" | Schema msg | RecordBatch msg(meta+body) | EOS | Footer | int32 footer_len | "ARROW1"
 * each msg = uint32 0xFFFFFFFF | uint32 metadata_len | flatbuffer | pad-to-8 | body.
 * Validated against pyarrow by round-trip (read back + value compare).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---------------- minimal little-endian FlatBuffers builder (builds back-to-front) ---------------- */
/* Arrow metadata for one batch (<=16 cols) is a few hundred bytes; 64 KiB scratch is ample. FB_CAP is a
   multiple of 8 so "position = FB_CAP - len" is N-aligned exactly when len is. */
#define FB_CAP 65536
typedef struct { uint8_t b[FB_CAP]; uint32_t len; int minalign; } fb_t;
static uint32_t fb_slot[32]; static int fb_nslots; static uint32_t fb_tstart;

static void     fb_init(fb_t *f) { f->len = 0; f->minalign = 1; }
static uint8_t *fb_data(fb_t *f) { return f->b + FB_CAP - f->len; }

static void fb_prep(fb_t *f, int size, int additional) {   /* pad so (len+additional) % size == 0 */
    if (size > f->minalign) f->minalign = size;
    int pad = (int)((size - ((f->len + additional) % size)) % size);
    while (pad-- > 0) { f->len++; f->b[FB_CAP - f->len] = 0; }
}
static void fb_raw(fb_t *f, const void *p, uint32_t n) { f->len += n; memcpy(f->b + FB_CAP - f->len, p, n); }
static void fb_u8 (fb_t *f, uint8_t v)  { f->len += 1; f->b[FB_CAP - f->len] = v; }
static void fb_u16(fb_t *f, uint16_t v) { fb_prep(f, 2, 0); fb_raw(f, &v, 2); }
static void fb_u32(fb_t *f, uint32_t v) { fb_prep(f, 4, 0); fb_raw(f, &v, 4); }
static void fb_i64(fb_t *f, int64_t v)  { fb_prep(f, 8, 0); fb_raw(f, &v, 8); }
static void fb_uoffset(fb_t *f, uint32_t target) { fb_prep(f, 4, 0); uint32_t o = (f->len + 4) - target; fb_raw(f, &o, 4); }

static uint32_t fb_string(fb_t *f, const char *s) {       /* [u32 len][bytes][nul], all contiguous */
    uint32_t n = (uint32_t)strlen(s);
    fb_prep(f, 4, n + 1);          /* pre-align at the tail so the length stays 4-aligned with NO gap before bytes */
    fb_u8(f, 0); fb_raw(f, s, n); fb_raw(f, &n, 4);
    return f->len;
}
/* vector of uoffsets: [u32 count][off0..offN-1] */
static uint32_t fb_vec_off(fb_t *f, const uint32_t *offs, int count) {
    fb_prep(f, 4, count * 4);
    for (int i = count - 1; i >= 0; i--) fb_uoffset(f, offs[i]);
    fb_prep(f, 4, 0); fb_raw(f, &(uint32_t){ (uint32_t)count }, 4);
    return f->len;
}
/* vector of inline structs (8-aligned): [u32 count][struct0..] */
static uint32_t fb_vec_struct(fb_t *f, const void *data, int count, int structsize) {
    fb_prep(f, 8, count * structsize);
    fb_raw(f, data, (uint32_t)(count * structsize));
    fb_prep(f, 4, 0); fb_raw(f, &(uint32_t){ (uint32_t)count }, 4);
    return f->len;
}
static void fb_start(fb_t *f) { for (int i = 0; i < 32; i++) fb_slot[i] = 0; fb_nslots = 0; fb_tstart = f->len; }
static void fb_bump(int slot) { if (slot + 1 > fb_nslots) fb_nslots = slot + 1; }   /* track vtable width */
static void fb_add_u8 (fb_t *f, int slot, uint8_t v)  { fb_u8(f, v);  fb_slot[slot] = f->len; fb_bump(slot); }
static void fb_add_u16(fb_t *f, int slot, uint16_t v) { fb_u16(f, v); fb_slot[slot] = f->len; fb_bump(slot); }
static void fb_add_u32(fb_t *f, int slot, uint32_t v) { fb_u32(f, v); fb_slot[slot] = f->len; fb_bump(slot); }
static void fb_add_i64(fb_t *f, int slot, int64_t v)  { fb_i64(f, v); fb_slot[slot] = f->len; fb_bump(slot); }
static void fb_add_off(fb_t *f, int slot, uint32_t target) { if (!target) return; fb_uoffset(f, target); fb_slot[slot] = f->len; fb_bump(slot); }

static uint32_t fb_end(fb_t *f) {
    fb_prep(f, 4, 0); f->len += 4;            /* reserve the soffset (lowest-addr word of the table) */
    uint32_t table_tail = f->len;
    int n = fb_nslots;
    for (int s = n - 1; s >= 0; s--)          /* vtable field offsets, high slot -> low (back-to-front) */
        fb_u16(f, fb_slot[s] ? (uint16_t)(table_tail - fb_slot[s]) : 0);
    fb_u16(f, (uint16_t)(table_tail - fb_tstart));   /* table inline size (fields + soffset) */
    fb_u16(f, (uint16_t)(4 + 2 * n));                /* vtable size */
    int32_t soff = (int32_t)(f->len - table_tail);   /* soffset: table -> vtable (vtable is after, positive) */
    memcpy(f->b + FB_CAP - table_tail, &soff, 4);
    return table_tail;
}
static uint32_t fb_finish(fb_t *f, uint32_t root) {     /* root uoffset at the buffer start, minalign-aligned */
    fb_prep(f, f->minalign, 4);
    uint32_t o = (f->len + 4) - root; fb_raw(f, &o, 4);
    return f->len;
}

/* ---------------- Arrow FlatBuffers enums (Schema.fbs / Message.fbs / File.fbs) ---------------- */
enum { TYPE_INT = 2, TYPE_FLOAT = 3 };           /* Type union */
enum { HDR_SCHEMA = 1, HDR_RECORDBATCH = 3 };    /* MessageHeader union */
enum { MV_V5 = 4 };                              /* MetadataVersion */
enum { FP_DOUBLE = 2 };                          /* Precision */

/* build the Schema table (shared by the Schema message and the Footer); returns its offset */
static uint32_t build_schema(fb_t *f, const char *const *names, const int *is_int, int ncol) {
    uint32_t fields[16];
    for (int ci = 0; ci < ncol; ci++) {
        uint32_t name_off = fb_string(f, names[ci]);
        uint32_t type_off;
        if (is_int[ci]) { fb_start(f); fb_add_u8(f, 1, 1); fb_add_u32(f, 0, 32); type_off = fb_end(f); }  /* Int{bitWidth=32(int),is_signed} */
        else            { fb_start(f); fb_add_u16(f, 0, FP_DOUBLE);              type_off = fb_end(f); }  /* FloatingPoint{precision=DOUBLE} */
        fb_start(f);
        fb_add_off(f, 0, name_off);                                   /* name */
        fb_add_u8 (f, 1, 1);                                          /* nullable = true */
        fb_add_u8 (f, 2, (uint8_t)(is_int[ci] ? TYPE_INT : TYPE_FLOAT)); /* type_type */
        fb_add_off(f, 3, type_off);                                   /* type */
        fields[ci] = fb_end(f);
    }
    uint32_t fields_vec = fb_vec_off(f, fields, ncol);
    fb_start(f);
    fb_add_u16(f, 0, 0);            /* endianness = Little */
    fb_add_off(f, 1, fields_vec);   /* fields */
    return fb_end(f);
}

/* ---------------- entry: write the Arrow IPC file ---------------- */
static void put_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_pad(FILE *f, long n) { static const char z[8] = {0}; while (n > 0) { long k = n < 8 ? n : 8; fwrite(z, 1, k, f); n -= k; } }

int ccx_arrow_write(const char *path, long nk, int mt,
                    const double *co, const double *v, const double *stn) {
    if (!path || !*path || nk <= 0 || !v) return 1;
    /* assemble the column list */
    const char *names[16]; int is_int[16]; int ncol = 0;
    names[ncol] = "node"; is_int[ncol] = 1; ncol++;
    if (co)  { static const char *c3[3] = { "x", "y", "z" };    for (int j = 0; j < 3; j++) { names[ncol] = c3[j]; is_int[ncol] = 0; ncol++; } }
    { static const char *u3[3] = { "ux", "uy", "uz" };          for (int j = 0; j < 3; j++) { names[ncol] = u3[j]; is_int[ncol] = 0; ncol++; } }
    if (stn) { static const char *s6[6] = { "sxx","syy","szz","sxy","sxz","syz" }; for (int j = 0; j < 6; j++) { names[ncol] = s6[j]; is_int[ncol] = 0; ncol++; } }

    /* body layout: 2 buffers per column (validity len 0, then data); each data buffer 8-padded */
    int64_t boff[32], blen[32]; int64_t cur = 0;
    for (int ci = 0; ci < ncol; ci++) {
        int64_t bytes = (int64_t)nk * (is_int[ci] ? 4 : 8);
        boff[2*ci] = cur; blen[2*ci] = 0;            /* validity */
        boff[2*ci+1] = cur; blen[2*ci+1] = bytes;    /* data */
        cur += (bytes + 7) & ~(int64_t)7;
    }
    int64_t bodyLen = cur;
    int nbuf = 2 * ncol;

    /* ---- Schema message flatbuffer ---- */
    static fb_t fbs; fb_init(&fbs);
    { uint32_t sch = build_schema(&fbs, names, is_int, ncol);
      fb_start(&fbs);
      fb_add_u16(&fbs, 0, MV_V5);        /* version */
      fb_add_u8 (&fbs, 1, HDR_SCHEMA);   /* header_type */
      fb_add_off(&fbs, 2, sch);          /* header */
      fb_add_i64(&fbs, 3, 0);            /* bodyLength */
      fb_finish(&fbs, fb_end(&fbs)); }

    /* ---- RecordBatch message flatbuffer ---- */
    static fb_t fbr; fb_init(&fbr);
    { int64_t nodes[32]; for (int ci = 0; ci < ncol; ci++) { nodes[2*ci] = nk; nodes[2*ci+1] = 0; } /* FieldNode{length,null_count} */
      int64_t bufs[64];  for (int k = 0; k < nbuf; k++) { bufs[2*k] = boff[k]; bufs[2*k+1] = blen[k]; } /* Buffer{offset,length} */
      uint32_t nodes_vec = fb_vec_struct(&fbr, nodes, ncol, 16);
      uint32_t bufs_vec  = fb_vec_struct(&fbr, bufs,  nbuf, 16);
      fb_start(&fbr);
      fb_add_i64(&fbr, 0, nk);           /* RecordBatch.length */
      fb_add_off(&fbr, 1, nodes_vec);    /* nodes */
      fb_add_off(&fbr, 2, bufs_vec);     /* buffers */
      uint32_t rb = fb_end(&fbr);
      fb_start(&fbr);
      fb_add_u16(&fbr, 0, MV_V5);
      fb_add_u8 (&fbr, 1, HDR_RECORDBATCH);
      fb_add_off(&fbr, 2, rb);
      fb_add_i64(&fbr, 3, bodyLen);
      fb_finish(&fbr, fb_end(&fbr)); }

    uint32_t smeta = (fbs.len + 7u) & ~7u;     /* padded metadata lengths */
    uint32_t rmeta = (fbr.len + 7u) & ~7u;

    /* ---- write file ---- */
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[accel] arrow: cannot open %s\n", path); return 2; }
    fwrite("ARROW1\0\0", 1, 8, f);
    long pos = 8;
    /* schema message */
    put_u32(f, 0xFFFFFFFFu); put_u32(f, smeta); fwrite(fb_data(&fbs), 1, fbs.len, f); put_pad(f, smeta - fbs.len);
    pos += 8 + smeta;
    /* record-batch message (the Footer Block points here) */
    long rb_block_off = pos;
    put_u32(f, 0xFFFFFFFFu); put_u32(f, rmeta); fwrite(fb_data(&fbr), 1, fbr.len, f); put_pad(f, rmeta - fbr.len);
    long rb_meta_len = 8 + rmeta;
    /* body: data buffers in order, each padded to 8 (validity buffers are length 0) */
    for (int ci = 0; ci < ncol; ci++) {
        int64_t bytes = blen[2*ci+1];
        if (is_int[ci]) for (long i = 0; i < nk; i++) { int32_t nd = (int32_t)(i + 1); fwrite(&nd, 4, 1, f); }
        else {
            const double *src = NULL; long stride = 0, base = 0;
            if (ci >= 1 && ci <= 3 && co) { src = co; stride = 3; base = ci - 1; }                 /* x,y,z */
            else { int u = ci - (co ? 4 : 1);                                                      /* index among u/s */
                   if (u < 3) { src = v; stride = mt; base = 1 + u; }                              /* ux,uy,uz */
                   else { /* stress */ }
            }
            if (src) for (long i = 0; i < nk; i++) fwrite(&src[stride*i + base], 8, 1, f);
            else if (stn) { int sj = ci - (co ? 4 : 1) - 3; for (long i = 0; i < nk; i++) fwrite(&stn[6*i + sj], 8, 1, f); }
            else { double zero = 0.0; for (long i = 0; i < nk; i++) fwrite(&zero, 8, 1, f); }
        }
        put_pad(f, (long)(((bytes + 7) & ~(int64_t)7) - bytes));
    }
    /* EOS marker */
    put_u32(f, 0xFFFFFFFFu); put_u32(f, 0);

    /* ---- Footer flatbuffer (schema + one RecordBatch Block) ---- */
    static fb_t fbf; fb_init(&fbf);
    { uint32_t sch = build_schema(&fbf, names, is_int, ncol);
      uint8_t block[24]; int64_t off64 = rb_block_off, body64 = bodyLen; int32_t mdl = (int32_t)rb_meta_len;
      memcpy(block + 0, &off64, 8); memcpy(block + 8, &mdl, 4); memset(block + 12, 0, 4); memcpy(block + 16, &body64, 8);
      uint32_t rbvec = fb_vec_struct(&fbf, block, 1, 24);
      fb_start(&fbf);
      fb_add_u16(&fbf, 0, MV_V5);    /* version */
      fb_add_off(&fbf, 1, sch);      /* schema */
      fb_add_off(&fbf, 3, rbvec);    /* recordBatches */
      fb_finish(&fbf, fb_end(&fbf)); }
    fwrite(fb_data(&fbf), 1, fbf.len, f);
    put_u32(f, fbf.len);
    fwrite("ARROW1", 1, 6, f);

    int werr = ferror(f); int cerr = (fclose(f) != 0);
    if (werr || cerr) { fprintf(stderr, "[accel] arrow: write error on %s\n", path); return 3; }
    fprintf(stderr, "[accel] wrote Arrow field (%ld nodes, %d cols, body %.1f MB) -> %s\n",
            nk, ncol, (double)bodyLen / 1e6, path);
    return 0;
}
