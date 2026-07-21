#!/usr/bin/env python3
"""Read a CCX Arrow IPC field file (written by ccx_arrowout.c when CCX_ACCEL_OUT_ARROW=<path>).

The file is a standard Arrow IPC "file"/Feather v2 with one record batch and columns:
  node:int32, [x,y,z], ux,uy,uz, [sxx,syy,szz,sxy,sxz,syz] : float64
so ANY Arrow reader works zero-copy -- this is just a convenience/validation example:

    import pyarrow.feather as f;  t = f.read_table("job.arrow")        # pyarrow
    import polars as pl;          df = pl.read_ipc("job.arrow")         # polars
    duckdb.sql("SELECT max(ux) FROM 'job.arrow'")                       # duckdb

Usage: arrow_read.py <file.arrow>   -> prints schema, row count, and field ranges.
"""
import sys
import math
import pyarrow.feather as feather


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    t = feather.read_table(sys.argv[1])
    d = t.to_pydict()
    print(f"{t.num_rows} rows, {len(t.schema.names)} cols: {t.schema.names}")
    if {"ux", "uy", "uz"} <= set(d):
        mag = max(math.sqrt(d["ux"][i] ** 2 + d["uy"][i] ** 2 + d["uz"][i] ** 2) for i in range(t.num_rows))
        print(f"max |U| = {mag:.10g}")
    for c in t.schema.names:
        col = d[c]
        print(f"  {c:5s} min={min(col):+.6g} max={max(col):+.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
