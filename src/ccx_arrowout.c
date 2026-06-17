/* CCX Arrow IPC field output. Tooling / APHYSICAL.
 *
 * Writes the nodal solution fields as an uncompressed Arrow IPC (Feather) file via vendored
 * nanoarrow. Called from frd.c (which has co/v/stn in C) when CCX_ACCEL_OUT_ARROW=<path> is set.
 * This replaces the ~933 MB text .dat write (the e2e benchmark bottleneck) with a ~100 MB binary
 * Arrow file that pyarrow/polars read zero-copy -> fast benchmarks + fast cross-mode comparison.
 *
 * Columns: node (int32, 1-based), x,y,z (f64 coords), ux,uy,uz (f64 displacement),
 *          and sxx,syy,szz,sxy,sxz,syz (f64 nodal stress) if stn != NULL.
 *
 * Layout from frd.c: mt = mi[1]+1; displacement of node i (0-based) = v[mt*i + 1..3];
 *                    coord = co[3*i + 0..2]; nodal stress = stn[6*i + 0..5].
 */
#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"
#include <stdio.h>
#include <math.h>

int ccx_arrow_write(const char *path, long nk, int mt,
                    const double *co, const double *v, const double *stn)
{
    if (!path || !*path || nk <= 0 || !v) return 1;
    const int with_co = (co != NULL);
    const int with_s  = (stn != NULL);
    const int ncol = 1 + (with_co ? 3 : 0) + 3 + (with_s ? 6 : 0);

    struct ArrowError err; err.message[0] = '\0';
    struct ArrowSchema schema;
    ArrowSchemaInit(&schema);
    if (ArrowSchemaSetTypeStruct(&schema, ncol)) return 2;
    int c = 0;
    ArrowSchemaSetType(schema.children[c], NANOARROW_TYPE_INT32);
    ArrowSchemaSetName(schema.children[c], "node"); c++;
    if (with_co) {
        const char *cn[3] = {"x","y","z"};
        for (int j = 0; j < 3; j++) { ArrowSchemaSetType(schema.children[c], NANOARROW_TYPE_DOUBLE);
            ArrowSchemaSetName(schema.children[c], cn[j]); c++; }
    }
    const char *un[3] = {"ux","uy","uz"};
    for (int j = 0; j < 3; j++) { ArrowSchemaSetType(schema.children[c], NANOARROW_TYPE_DOUBLE);
        ArrowSchemaSetName(schema.children[c], un[j]); c++; }
    if (with_s) {
        const char *sn[6] = {"sxx","syy","szz","sxy","sxz","syz"};
        for (int j = 0; j < 6; j++) { ArrowSchemaSetType(schema.children[c], NANOARROW_TYPE_DOUBLE);
            ArrowSchemaSetName(schema.children[c], sn[j]); c++; }
    }

    struct ArrowArray array;
    if (ArrowArrayInitFromSchema(&array, &schema, &err)) { schema.release(&schema); return 3; }
    if (ArrowArrayStartAppending(&array)) { schema.release(&schema); return 3; }
    ArrowArrayReserve(&array, nk);   /* best-effort preallocation */

    for (long i = 0; i < nk; i++) {
        int cc = 0;
        ArrowArrayAppendInt(array.children[cc++], (int64_t)(i + 1));
        if (with_co) { ArrowArrayAppendDouble(array.children[cc++], co[3*i]);
                       ArrowArrayAppendDouble(array.children[cc++], co[3*i+1]);
                       ArrowArrayAppendDouble(array.children[cc++], co[3*i+2]); }
        ArrowArrayAppendDouble(array.children[cc++], v[(long)mt*i + 1]);
        ArrowArrayAppendDouble(array.children[cc++], v[(long)mt*i + 2]);
        ArrowArrayAppendDouble(array.children[cc++], v[(long)mt*i + 3]);
        if (with_s) for (int j = 0; j < 6; j++) ArrowArrayAppendDouble(array.children[cc++], stn[6*i + j]);
        if (ArrowArrayFinishElement(&array)) { schema.release(&schema); array.release(&array); return 4; }
    }
    if (ArrowArrayFinishBuildingDefault(&array, &err)) { schema.release(&schema); array.release(&array); return 5; }

    struct ArrowArrayStream stream;
    if (ArrowBasicArrayStreamInit(&stream, &schema, 1)) { schema.release(&schema); array.release(&array); return 6; }
    ArrowBasicArrayStreamSetArray(&stream, 0, &array);

    FILE *f = fopen(path, "wb");
    if (!f) { stream.release(&stream); return 7; }
    struct ArrowIpcOutputStream out;
    if (ArrowIpcOutputStreamInitFile(&out, f, 1)) { stream.release(&stream); return 8; }
    struct ArrowIpcWriter writer;
    if (ArrowIpcWriterInit(&writer, &out)) { stream.release(&stream); return 9; }
    int rc = 0;
    if (ArrowIpcWriterStartFile(&writer, &err)) rc = 10;
    else if (ArrowIpcWriterWriteArrayStream(&writer, &stream, &err)) rc = 11;
    else if (ArrowIpcWriterFinalizeFile(&writer, &err)) rc = 12;
    ArrowIpcWriterReset(&writer);
    stream.release(&stream);
    if (rc) fprintf(stderr, "[accel] arrow write rc=%d: %s\n", rc, err.message);
    else    fprintf(stderr, "[accel] wrote Arrow field (%ld nodes, %d cols) -> %s\n", nk, ncol, path);
    return rc;
}
