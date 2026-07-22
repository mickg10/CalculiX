#!/usr/bin/env python3
"""Versioned brick-major row236 operator format, serializer, and host oracle.

The old ``serialize_tiled.py`` appended active-node-order A27 data after a brick table.  It did not
pad/reorder coefficients into brick slots and did not contain an occupancy/local-slot mapping, so a
device could not perform the claimed resident-x brick shift.  This module defines the first format in
which the binary layout and the TT stream layout are the same thing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


MAGIC = b"TTB4SPV\0"
VERSION = 1
ENDIAN_MARKER = 0x01020304
HEADER_BYTES = 4096
BRICKS_PER_TILE = 16
TILE_ELEMENTS = 1024
OFFSETS = np.asarray(
    [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)],
    dtype=np.int8,
)

BASE_HEADER = struct.Struct("<8sIIII5Q8I6Q")
SECTION_HEADER = struct.Struct("<16sII4QQQ")

DTYPE_TO_CODE = {
    np.dtype("int8"): 1,
    np.dtype("uint8"): 2,
    np.dtype("<i4"): 3,
    np.dtype("<u4"): 4,
    np.dtype("<u8"): 5,
    np.dtype("<u2"): 6,
}
CODE_TO_DTYPE = {value: key for key, value in DTYPE_TO_CODE.items()}


def _align(value: int, alignment: int = 4096) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2")


def _bf16_values(bits: np.ndarray) -> np.ndarray:
    words = np.asarray(bits, dtype="<u2").astype("<u4") << np.uint32(16)
    return words.view(np.float32)


def split_bf16x3(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round-to-nearest-even fp32 -> hi/mid/lo bf16 residual expansion."""

    source = np.asarray(values, dtype=np.float32)
    hi_bits = _bf16_bits(source)
    hi = _bf16_values(hi_bits)
    mid_bits = _bf16_bits(source - hi)
    mid = _bf16_values(mid_bits)
    lo_bits = _bf16_bits(source - hi - mid)
    return hi_bits, mid_bits, lo_bits


@dataclass(frozen=True)
class FineDump:
    path: Path
    nb: int
    nblk: int
    nx: int
    ny: int
    nz: int
    ptr: np.memmap
    col: np.memmap
    val: np.memmap
    ijk: np.memmap


def open_fine_dump(path: Path) -> FineDump:
    path = path.expanduser().resolve()
    size = path.stat().st_size
    if size < 40:
        raise ValueError(f"{path}: shorter than the 40-byte fine-dump header")
    header = np.memmap(path, dtype="<i8", mode="r", offset=0, shape=(5,))
    nb, nblk, nx, ny, nz = (int(value) for value in header)
    if min(nb, nblk, nx, ny, nz) <= 0:
        raise ValueError(f"{path}: invalid header {(nb, nblk, nx, ny, nz)}")
    ptr_offset = 40
    col_offset = ptr_offset + 8 * (nb + 1)
    val_offset = col_offset + 4 * nblk
    ijk_offset = val_offset + 8 * nblk * 9
    minimum = ijk_offset + 8 * nb * 3
    if size < minimum:
        raise ValueError(f"{path}: truncated ({size} bytes; need at least {minimum})")
    ptr = np.memmap(path, dtype="<i8", mode="r", offset=ptr_offset, shape=(nb + 1,))
    col = np.memmap(path, dtype="<i4", mode="r", offset=col_offset, shape=(nblk,))
    val = np.memmap(path, dtype="<f8", mode="r", offset=val_offset, shape=(nblk, 9))
    ijk = np.memmap(path, dtype="<i8", mode="r", offset=ijk_offset, shape=(nb, 3))
    if int(ptr[0]) != 0 or int(ptr[-1]) != nblk or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError(f"{path}: invalid BCSR row pointer")
    if int(col.min()) < 0 or int(col.max()) >= nb:
        raise ValueError(f"{path}: BCSR column outside [0,{nb})")
    bounds = np.asarray((nx, ny, nz), dtype=np.int64)
    if np.any(ijk < 0) or np.any(ijk >= bounds):
        raise ValueError(f"{path}: ijk outside the declared box")
    return FineDump(path, nb, nblk, nx, ny, nz, ptr, col, val, ijk)


@dataclass(frozen=True)
class Section:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def pack(self) -> bytes:
        shape4 = self.shape + (0,) * (4 - len(self.shape))
        return SECTION_HEADER.pack(
            self.name.encode("ascii").ljust(16, b"\0"),
            DTYPE_TO_CODE[np.dtype(self.dtype)],
            len(self.shape),
            *shape4,
            self.offset,
            self.nbytes,
        )


def _make_sections(nb: int, nbrick_pad: int, nout: int) -> list[Section]:
    specifications = [
        ("offsets", np.dtype("int8"), (27, 3)),
        ("brick_coords", np.dtype("<i4"), (nbrick_pad, 3)),
        ("brick_nbr", np.dtype("<i4"), (nbrick_pad, 27)),
        ("occupancy", np.dtype("<u8"), (nbrick_pad,)),
        ("slot_to_node", np.dtype("<i4"), (nbrick_pad, 64)),
        ("node_to_pos", np.dtype("<u4"), (nb,)),
        ("coefficients", np.dtype("<u2"), (3, nout, 81, 1024)),
    ]
    result: list[Section] = []
    cursor = HEADER_BYTES
    for name, dtype, shape in specifications:
        cursor = _align(cursor)
        nbytes = int(math.prod(shape) * dtype.itemsize)
        result.append(Section(name, dtype, shape, cursor, nbytes))
        cursor += nbytes
    return result


def _write_header(
    handle,
    dump: FineDump,
    brick_size: int,
    nbrick: int,
    nbrick_pad: int,
    ngroup: int,
    nout: int,
    sections: Sequence[Section],
) -> None:
    base = BASE_HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        ENDIAN_MARKER,
        0,
        dump.nb,
        dump.nblk,
        dump.nx,
        dump.ny,
        dump.nz,
        brick_size,
        brick_size**3,
        27,
        3,
        3,
        3,
        BRICKS_PER_TILE,
        len(sections),
        nbrick,
        nbrick_pad,
        ngroup,
        nout,
        81,
        0,
    )
    table = b"".join(section.pack() for section in sections)
    if len(base) + len(table) > HEADER_BYTES:
        raise AssertionError("section table exceeds fixed header")
    handle.seek(0)
    handle.write(base)
    handle.write(table)
    handle.write(b"\0" * (HEADER_BYTES - len(base) - len(table)))


def _metadata(dump: FineDump, brick_size: int) -> tuple[np.ndarray, ...]:
    if brick_size != 4:
        raise ValueError("format v1 requires brick_size=4 (64 slots and 16 bricks per TT tile)")
    brick_coord_per_node = np.asarray(dump.ijk // brick_size, dtype="<i4")
    brick_coords, inverse = np.unique(brick_coord_per_node, axis=0, return_inverse=True)
    inverse = np.asarray(inverse, dtype="<i4")
    nbrick = len(brick_coords)
    local = np.asarray(dump.ijk % brick_size, dtype=np.int64)
    slot = ((local[:, 0] * brick_size + local[:, 1]) * brick_size + local[:, 2]).astype("<u4")
    positions = (inverse.astype(np.uint64) * 64 + slot.astype(np.uint64)).astype("<u4")
    if len(np.unique(positions)) != dump.nb:
        raise ValueError("duplicate active lattice coordinate")
    slot_to_node = np.full((nbrick, 64), -1, dtype="<i4")
    slot_to_node.reshape(-1)[positions] = np.arange(dump.nb, dtype="<i4")
    occupancy = np.zeros(nbrick, dtype="<u8")
    np.bitwise_or.at(occupancy, inverse, np.left_shift(np.uint64(1), slot.astype(np.uint64)))
    coord_to_id = {tuple(int(v) for v in coord): index for index, coord in enumerate(brick_coords)}
    brick_nbr = np.full((nbrick, 27), -1, dtype="<i4")
    for brick_id, coord in enumerate(brick_coords):
        ci, cj, ck = (int(value) for value in coord)
        for offset_id, (di, dj, dk) in enumerate(OFFSETS):
            brick_nbr[brick_id, offset_id] = coord_to_id.get((ci + int(di), cj + int(dj), ck + int(dk)), -1)
    return brick_coords.astype("<i4"), brick_nbr, occupancy, slot_to_node, positions


def _memmap_section(path: Path, section: Section, mode: str = "r+") -> np.memmap:
    return np.memmap(path, dtype=section.dtype, mode=mode, offset=section.offset, shape=section.shape)


def serialize(
    fine_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    *,
    brick_size: int = 4,
    row_chunk: int = 8192,
    force: bool = False,
    hash_files: bool = True,
) -> dict[str, Any]:
    started = time.time()
    dump = open_fine_dump(fine_path)
    output_path = output_path.expanduser().resolve()
    manifest_path = (manifest_path or output_path.with_suffix(output_path.suffix + ".json")).expanduser().resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} exists; pass --force to replace it")
    brick_coords, brick_nbr, occupancy, slot_to_node, node_to_pos = _metadata(dump, brick_size)
    nbrick = len(brick_coords)
    nbrick_pad = _align(nbrick, BRICKS_PER_TILE)
    ngroup = nbrick_pad // BRICKS_PER_TILE
    nout = 3 * ngroup
    sections = _make_sections(dump.nb, nbrick_pad, nout)
    by_name = {section.name: section for section in sections}
    total_bytes = _align(max(section.offset + section.nbytes for section in sections))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        partial.unlink()
    with partial.open("w+b") as handle:
        handle.truncate(total_bytes)
        _write_header(handle, dump, brick_size, nbrick, nbrick_pad, ngroup, nout, sections)

    offsets_mm = _memmap_section(partial, by_name["offsets"])
    offsets_mm[:] = OFFSETS
    offsets_mm.flush()
    coords_mm = _memmap_section(partial, by_name["brick_coords"])
    coords_mm[:] = -1
    coords_mm[:nbrick] = brick_coords
    coords_mm.flush()
    nbr_mm = _memmap_section(partial, by_name["brick_nbr"])
    nbr_mm[:] = -1
    nbr_mm[:nbrick] = brick_nbr
    nbr_mm.flush()
    occ_mm = _memmap_section(partial, by_name["occupancy"])
    occ_mm[:] = 0
    occ_mm[:nbrick] = occupancy
    occ_mm.flush()
    slots_mm = _memmap_section(partial, by_name["slot_to_node"])
    slots_mm[:] = -1
    slots_mm[:nbrick] = slot_to_node
    slots_mm.flush()
    pos_mm = _memmap_section(partial, by_name["node_to_pos"])
    pos_mm[:] = node_to_pos
    pos_mm.flush()
    del offsets_mm, coords_mm, nbr_mm, occ_mm, slots_mm, pos_mm

    coefficients = _memmap_section(partial, by_name["coefficients"])
    # A newly truncated file reads as zero. Only actual BCSR blocks are assigned below; padded/inactive lanes remain zero.
    assigned = 0
    for row0 in range(0, dump.nb, row_chunk):
        row1 = min(row0 + row_chunk, dump.nb)
        block0, block1 = int(dump.ptr[row0]), int(dump.ptr[row1])
        counts = np.diff(np.asarray(dump.ptr[row0 : row1 + 1], dtype=np.int64))
        rows = np.repeat(np.arange(row0, row1, dtype=np.int64), counts)
        if len(rows) != block1 - block0:
            raise AssertionError("row expansion does not match BCSR block span")
        cols = np.asarray(dump.col[block0:block1], dtype=np.int64)
        delta = np.asarray(dump.ijk[cols] - dump.ijk[rows], dtype=np.int64)
        if np.any(delta < -1) or np.any(delta > 1):
            bad = int(np.flatnonzero(np.any((delta < -1) | (delta > 1), axis=1))[0])
            raise ValueError(f"non-27-point block at global block {block0 + bad}: delta={delta[bad].tolist()}")
        offset_id = ((delta[:, 0] + 1) * 9 + (delta[:, 1] + 1) * 3 + (delta[:, 2] + 1)).astype(np.int64)
        keys = (rows - row0) * 27 + offset_id
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"duplicate geometric offset in BCSR rows [{row0},{row1})")
        position = node_to_pos[rows].astype(np.int64)
        group = position // TILE_ELEMENTS
        lane = position % TILE_ELEMENTS
        values = np.asarray(dump.val[block0:block1], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite coefficient in blocks [{block0},{block1})")
        for coefficient_id in range(9):
            out_component, in_component = divmod(coefficient_id, 3)
            out_tile = out_component * ngroup + group
            term = offset_id * 3 + in_component
            hi, mid, lo = split_bf16x3(values[:, coefficient_id])
            coefficients[0, out_tile, term, lane] = hi
            coefficients[1, out_tile, term, lane] = mid
            coefficients[2, out_tile, term, lane] = lo
        assigned += len(rows)
    coefficients.flush()
    del coefficients
    if assigned != dump.nblk:
        raise AssertionError(f"assigned {assigned} blocks, expected {dump.nblk}")
    os.replace(partial, output_path)

    manifest: dict[str, Any] = {
        "schema": "tt_gmg_brick_operator_manifest_v1",
        "format_magic": MAGIC.rstrip(b"\0").decode("ascii"),
        "format_version": VERSION,
        "created_unix_s": time.time(),
        "source_fine_dump": str(dump.path),
        "output": str(output_path),
        "source": {"nb": dump.nb, "nblk": dump.nblk, "box": [dump.nx, dump.ny, dump.nz]},
        "brick": {
            "size": brick_size,
            "slots": 64,
            "count": nbrick,
            "padded_count": nbrick_pad,
            "groups": ngroup,
            "covered_positions": nbrick * 64,
            "covered_to_active_ratio": nbrick * 64 / dump.nb,
            "occupancy_fraction": dump.nb / (nbrick * 64),
        },
        "operator": {
            "geometric_offsets": 27,
            "block_shape": [3, 3],
            "terms_per_output_component": 81,
            "bf16_splits": 3,
            "output_tiles": nout,
            "layout": "coefficients[split][output_component*brick_groups+group][offset*3+input_component][16_bricks*64_slots]",
            "payload_bytes": by_name["coefficients"].nbytes,
        },
        "sections": [
            {"name": section.name, "dtype": section.dtype.str, "shape": list(section.shape), "offset": section.offset, "bytes": section.nbytes}
            for section in sections
        ],
        "file_bytes": output_path.stat().st_size,
        "serialize_seconds": time.time() - started,
        "hashes": {},
    }
    if hash_files:
        manifest["hashes"] = {"source_sha256": _sha256(dump.path), "output_sha256": _sha256(output_path)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class BrickOperator:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        with self.path.open("rb") as handle:
            raw = handle.read(HEADER_BYTES)
        if len(raw) != HEADER_BYTES:
            raise ValueError(f"{self.path}: truncated header")
        values = BASE_HEADER.unpack_from(raw, 0)
        magic, version, header_bytes, endian, flags = values[:5]
        if magic != MAGIC or version != VERSION or header_bytes != HEADER_BYTES or endian != ENDIAN_MARKER:
            raise ValueError(f"{self.path}: incompatible header magic/version/endian")
        self.flags = flags
        self.nb, self.nblk, self.nx, self.ny, self.nz = (int(v) for v in values[5:10])
        (
            self.brick_size,
            self.slots,
            self.noff,
            self.block_rows,
            self.block_cols,
            self.splits,
            self.bricks_per_tile,
            section_count,
        ) = (int(v) for v in values[10:18])
        self.nbrick, self.nbrick_pad, self.ngroup, self.nout, self.nterms, _ = (int(v) for v in values[18:24])
        self.sections: dict[str, Section] = {}
        cursor = BASE_HEADER.size
        for _ in range(section_count):
            entry = SECTION_HEADER.unpack_from(raw, cursor)
            cursor += SECTION_HEADER.size
            name = entry[0].split(b"\0", 1)[0].decode("ascii")
            dtype = CODE_TO_DTYPE.get(entry[1])
            ndim = int(entry[2])
            if dtype is None or not 1 <= ndim <= 4:
                raise ValueError(f"{self.path}: invalid section {name}")
            shape = tuple(int(v) for v in entry[3 : 3 + ndim])
            offset, nbytes = int(entry[7]), int(entry[8])
            section = Section(name, dtype, shape, offset, nbytes)
            if math.prod(shape) * dtype.itemsize != nbytes:
                raise ValueError(f"{self.path}: section {name} byte count mismatch")
            self.sections[name] = section
        required = {"offsets", "brick_coords", "brick_nbr", "occupancy", "slot_to_node", "node_to_pos", "coefficients"}
        if set(self.sections) != required:
            raise ValueError(f"{self.path}: section set mismatch: {set(self.sections)}")
        ordered = sorted(self.sections.values(), key=lambda section: section.offset)
        cursor = HEADER_BYTES
        for section in ordered:
            if section.offset < _align(cursor):
                raise ValueError(f"{self.path}: overlapping/misaligned section {section.name}")
            cursor = section.offset + section.nbytes
        if self.path.stat().st_size < cursor:
            raise ValueError(f"{self.path}: truncated final section")

    def section(self, name: str) -> np.memmap:
        section = self.sections[name]
        return np.memmap(self.path, dtype=section.dtype, mode="r", offset=section.offset, shape=section.shape)


def _slot_shift_tables(brick_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    delta_id = np.empty((brick_size**3, 27), dtype=np.int8)
    source_slot = np.empty((brick_size**3, 27), dtype=np.uint8)
    for slot in range(brick_size**3):
        i = slot // (brick_size * brick_size)
        j = (slot // brick_size) % brick_size
        k = slot % brick_size
        for oid, offset in enumerate(OFFSETS):
            target = np.asarray((i, j, k), dtype=np.int64) + offset.astype(np.int64)
            brick_delta = np.floor_divide(target, brick_size)
            local = target - brick_delta * brick_size
            delta_id[slot, oid] = int((brick_delta[0] + 1) * 9 + (brick_delta[1] + 1) * 3 + brick_delta[2] + 1)
            source_slot[slot, oid] = int((local[0] * brick_size + local[1]) * brick_size + local[2])
    return delta_id, source_slot


def bcsr_apply(dump: FineDump, vector: np.ndarray, row_chunk: int = 8192) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float64).reshape(dump.nb, 3)
    y = np.zeros((dump.nb, 3), dtype=np.float64)
    for row0 in range(0, dump.nb, row_chunk):
        row1 = min(row0 + row_chunk, dump.nb)
        block0, block1 = int(dump.ptr[row0]), int(dump.ptr[row1])
        counts = np.diff(np.asarray(dump.ptr[row0 : row1 + 1], dtype=np.int64))
        products = np.einsum(
            "bij,bj->bi",
            np.asarray(dump.val[block0:block1], dtype=np.float64).reshape(-1, 3, 3),
            x[np.asarray(dump.col[block0:block1], dtype=np.int64)],
            optimize=True,
        )
        starts = np.cumsum(np.r_[0, counts[:-1]])
        y[row0:row1] = np.add.reduceat(products, starts, axis=0)
    return y


def _bf16_fidelity_phases(
    operand_a: np.ndarray,
    operand_b: np.ndarray,
    math_fidelity: int,
) -> tuple[np.ndarray, ...]:
    """Model Wormhole BF16 ELWMUL fidelity phases.

    Wormhole expands BF16 operands into a TF32-style source register. The
    four phases multiply top-A/top-B, low-A/top-B, top-A/low-B, and
    low-A/low-B significand partitions. HiFi2 retains the first two and
    HiFi3 retains the first three. HiFi4 uses the exact BF16 product so the
    established full-fidelity oracle remains bit-for-bit stable.
    """
    if math_fidelity not in (0, 2, 3, 4):
        raise ValueError("math_fidelity must be one of 0, 2, 3, or 4")
    a = np.asarray(operand_a, dtype=np.float32)
    b = np.asarray(operand_b, dtype=np.float32)
    if math_fidelity == 4:
        return (a * b,)

    # The first A phase keeps the implied bit plus four BF16 fraction bits;
    # the first B phase keeps the implied bit plus six. Clearing float32
    # mantissa bits is equivalent to the LLK TF32 significand masks for the
    # finite BF16 operands used by this operator.
    a_top = (a.view(np.uint32) & np.uint32(0xFFF80000)).view(np.float32)
    b_top = (b.view(np.uint32) & np.uint32(0xFFFE0000)).view(np.float32)
    a_low = a - a_top
    b_low = b - b_top
    phases = [a_top * b_top]
    if math_fidelity >= 2:
        phases.append(a_low * b_top)
    if math_fidelity >= 3:
        phases.append(a_top * b_low)
    return tuple(phases)


def _bf16_hifi2_exact_product_phases(
    operand_a: np.ndarray,
    operand_b: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Reconstruct one exact BF16 product with two fixed-HiFi2 calls.

    HiFi2 consumes both A significand phases but only the top B phase.  A
    BF16 operand can be split into a six-fraction-bit top value and a
    one-bit residual, and that residual is itself BF16.  Applying HiFi2 to
    each tile therefore emits the same four product phases as HiFi4 without
    changing the device math MOP.
    """
    b = np.asarray(operand_b, dtype=np.float32)
    b_top = (b.view(np.uint32) & np.uint32(0xFFFE0000)).view(np.float32)
    b_residual = b - b_top
    return (
        *_bf16_fidelity_phases(operand_a, b_top, 2),
        *_bf16_fidelity_phases(operand_a, b_residual, 2),
    )


def brick_apply(
    operator: BrickOperator,
    vector: np.ndarray,
    group_chunk: int = 8,
    *,
    b_splits: int = 3,
    b_low_terms: Sequence[int] | None = None,
    a_low_terms: Sequence[int] | None = None,
    a_low_affine_corner_fold: bool = False,
    accumulator_bias: float = 0.0,
    accumulator_chunk_terms: int = 0,
    term_order: str = "natural",
    math_fidelity: int = 4,
    hifi3_products: Sequence[tuple[int, int]] | None = None,
    hifi2_products: Sequence[tuple[int, int]] | None = None,
    fidelity_batch_terms: int = 0,
    fixed_hifi2_exact_leading: bool = False,
    compensated_product_order: Sequence[tuple[int, int]] | None = None,
    compensated_product_order_no_blow: Sequence[tuple[int, int]] | None = None,
    coefficient_quantum: float = 0.0,
    prefix_extrema: dict[str, float] | None = None,
) -> np.ndarray:
    if operator.brick_size != 4 or operator.slots != 64 or operator.bricks_per_tile != 16:
        raise ValueError("host oracle currently implements format-v1 4^3 bricks only")
    if b_splits not in (2, 3):
        raise ValueError("b_splits must be 2 or 3")
    low_term_set = None if b_low_terms is None else frozenset(int(term) for term in b_low_terms)
    if low_term_set is not None:
        if b_splits != 3:
            raise ValueError("b_low_terms requires b_splits=3")
        if any(term < 0 or term >= 81 for term in low_term_set):
            raise ValueError("b_low_terms entries must be in [0,80]")
    a_low_term_set = None if a_low_terms is None else frozenset(int(term) for term in a_low_terms)
    if a_low_term_set is not None and any(term < 0 or term >= 81 for term in a_low_term_set):
        raise ValueError("a_low_terms entries must be in [0,80]")
    corner_offset_order = tuple(
        oid
        for oid, offset in enumerate(OFFSETS)
        if sum(abs(int(value)) for value in offset) == 3
    )
    corner_offsets = frozenset(corner_offset_order)
    center_and_face_offsets = frozenset(
        oid
        for oid, offset in enumerate(OFFSETS)
        if sum(abs(int(value)) for value in offset) <= 1
    )
    if a_low_affine_corner_fold:
        retained_offsets = frozenset(term // 3 for term in (a_low_term_set or ()))
        if not center_and_face_offsets.issubset(retained_offsets):
            raise ValueError(
                "a_low_affine_corner_fold requires all center and face a-low terms"
            )
        if corner_offsets & retained_offsets:
            raise ValueError(
                "a_low_affine_corner_fold requires all corner a-low terms to be omitted"
            )
    if accumulator_chunk_terms < 0 or accumulator_chunk_terms > 81:
        raise ValueError("accumulator_chunk_terms must be in [0,81]")
    if math_fidelity not in (0, 2, 3, 4):
        raise ValueError("math_fidelity must be one of 0, 2, 3, or 4")
    hifi3_product_set = frozenset(hifi3_products or ())
    hifi2_product_set = frozenset(hifi2_products or ())
    valid_products = {
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (2, 0),
    }
    if not hifi3_product_set.issubset(valid_products):
        raise ValueError("hifi3_products contains an unknown compensated product")
    if not hifi2_product_set.issubset(valid_products):
        raise ValueError("hifi2_products contains an unknown compensated product")
    if (hifi3_product_set or hifi2_product_set) and math_fidelity != 4:
        raise ValueError("mixed product fidelity requires a HiFi4 base")
    if hifi3_product_set & hifi2_product_set:
        raise ValueError("hifi3_products and hifi2_products must be disjoint")
    if fidelity_batch_terms < 0 or fidelity_batch_terms > 81:
        raise ValueError("fidelity_batch_terms must be in [0,81]")
    if coefficient_quantum < 0.0:
        raise ValueError("coefficient_quantum must be nonnegative")
    if fidelity_batch_terms and not (hifi3_product_set or hifi2_product_set):
        raise ValueError("fidelity_batch_terms requires mixed product fidelity")
    if fidelity_batch_terms and (
        accumulator_chunk_terms
            or low_term_set is not None
            or a_low_term_set is not None
            or a_low_affine_corner_fold
        ):
        raise ValueError(
            "fidelity_batch_terms is incompatible with chunked accumulation or mixed low-term masks"
        )
    if fixed_hifi2_exact_leading:
        if b_splits != 3 or math_fidelity != 4:
            raise ValueError(
                "fixed_hifi2_exact_leading requires b_splits=3 and a HiFi4 oracle base"
            )
        if (
            low_term_set is not None
            or a_low_term_set is not None
            or a_low_affine_corner_fold
            or hifi3_product_set
            or hifi2_product_set
            or fidelity_batch_terms
        ):
            raise ValueError(
                "fixed_hifi2_exact_leading is incompatible with other reduced-fidelity schedules"
            )
    if term_order == "natural":
        ordered_terms = tuple(range(81))
    elif term_order == "center_first":
        ordered_terms = (39, 40, 41) + tuple(range(39)) + tuple(range(42, 81))
    elif term_order == "reverse":
        ordered_terms = tuple(reversed(range(81)))
    else:
        raise ValueError("term_order must be natural, center_first, or reverse")
    x = np.asarray(vector, dtype=np.float32).reshape(operator.nb, 3)
    node_to_pos = np.asarray(operator.section("node_to_pos"), dtype=np.int64)
    x_box = np.zeros((operator.nbrick_pad * 64, 3), dtype=np.float32)
    x_box[node_to_pos] = x
    x_hi_bits, x_mid_bits, x_lo_bits = split_bf16x3(x_box)
    x_split = tuple(_bf16_values(bits) for bits in (x_hi_bits, x_mid_bits, x_lo_bits)[:b_splits])
    # The five-term path is the intended resident-x ABI: x carries hi/mid while A keeps hi/mid/lo.
    # Keep this order identical to the compute kernel because fp32 destination accumulation is ordered.
    default_products = (
        ((0, 0), (0, 1), (1, 0), (2, 0), (1, 1))
        if b_splits == 2
        else ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))
    )
    products = tuple(compensated_product_order or default_products)
    if len(products) != len(default_products) or set(products) != set(default_products):
        raise ValueError("compensated_product_order must be a permutation of the active compensated products")
    default_no_blow_products = (
        ((0, 0), (0, 1), (1, 0), (2, 0), (1, 1))
        if b_splits == 3
        else default_products
    )
    no_blow_products = tuple(
        compensated_product_order_no_blow
        or (
            default_no_blow_products
            if compensated_product_order is None
            else tuple(product for product in products if product != (0, 2))
        )
    )
    if (
        len(no_blow_products) != len(default_no_blow_products)
        or set(no_blow_products) != set(default_no_blow_products)
    ):
        raise ValueError(
            "compensated_product_order_no_blow must be a permutation of the five-product T5 set"
        )
    brick_nbr = operator.section("brick_nbr")
    coefficients = operator.section("coefficients")
    delta_id, source_slot = _slot_shift_tables(4)
    local_slot = np.tile(np.arange(64, dtype=np.int64), 16)
    delta_lane = np.tile(delta_id[:, :].T[:, local_slot], (1, 1))
    source_lane = np.tile(source_slot[:, :].T[:, local_slot], (1, 1))
    bias = np.float32(accumulator_bias)
    chunked = accumulator_chunk_terms > 0
    y_box = np.zeros((operator.nbrick_pad * 64, 3), dtype=np.float32)
    scratch_box = (
        np.full((operator.nbrick_pad * 64, 3), bias, dtype=np.float32)
        if chunked
        else None
    )
    if not chunked:
        y_box.fill(bias)
    if prefix_extrema is not None:
        prefix_extrema.clear()
        prefix_extrema.update(minimum=float(bias), maximum=float(bias))
    for group0 in range(0, operator.ngroup, group_chunk):
        group1 = min(group0 + group_chunk, operator.ngroup)
        groups = np.arange(group0, group1, dtype=np.int64)
        bricks = groups[:, None] * 16 + np.arange(16, dtype=np.int64)[None, :]
        brick_for_lane = np.repeat(bricks, 64, axis=1)
        positions = (
            groups[:, None] * TILE_ELEMENTS
            + np.arange(TILE_ELEMENTS, dtype=np.int64)[None, :]
        ).reshape(-1)
        affine_corner_corrections: dict[tuple[int, int], np.ndarray] = {}
        if a_low_affine_corner_fold:
            for out_component in range(3):
                out_tiles = out_component * operator.ngroup + groups
                for in_component in range(3):
                    correction_by_offset = {
                        oid: np.zeros(
                            (group1 - group0, TILE_ELEMENTS),
                            dtype=np.float32,
                        )
                        for oid in center_and_face_offsets
                    }
                    for corner_oid in corner_offset_order:
                        di, dj, dk = (int(value) for value in OFFSETS[corner_oid])
                        corner_low = _bf16_values(
                            coefficients[
                                2,
                                out_tiles,
                                3 * corner_oid + in_component,
                                :,
                            ]
                        )
                        correction_by_offset[13] -= np.float32(2.0) * corner_low
                        correction_by_offset[(di + 1) * 9 + 4] += corner_low
                        correction_by_offset[9 + (dj + 1) * 3 + 1] += corner_low
                        correction_by_offset[12 + (dk + 1)] += corner_low
                    for oid, correction in correction_by_offset.items():
                        affine_corner_corrections[
                            (out_component, 3 * oid + in_component)
                        ] = correction
        if fidelity_batch_terms:
            # Model a candidate device schedule: retain a small CB batch,
            # accumulate all base-fidelity products for those terms, change
            # the math MOP once, then accumulate each reduced-fidelity product
            # set.  A batch of one is the run44-proven device order.  Although
            # larger batches preserve every accumulator's fp32 operation
            # order in this arithmetic oracle, batches of three corrupted the
            # same device lane in runs 45, 46, 48, and 49.  Full LLK
            # reinitialization, a complete Tensix pipeline drain, and an
            # explicit FP32 DEST clear did not change the failure.  This path
            # therefore remains a host scheduling oracle only; it is not
            # evidence that cross-term fidelity batching is device-safe.
            reduced_products = hifi3_product_set | hifi2_product_set
            fidelity_product_groups = (
                (
                    math_fidelity,
                    tuple(product for product in products if product not in reduced_products),
                ),
                (3, tuple(product for product in products if product in hifi3_product_set)),
                (2, tuple(product for product in products if product in hifi2_product_set)),
            )
            for batch0 in range(0, 81, fidelity_batch_terms):
                term_batch = ordered_terms[batch0 : batch0 + fidelity_batch_terms]
                prepared_terms = []
                for term in term_batch:
                    oid = term // 3
                    in_component = term - oid * 3
                    did = np.broadcast_to(delta_lane[oid], brick_for_lane.shape)
                    source_brick = np.asarray(brick_nbr[brick_for_lane, did], dtype=np.int64)
                    valid = source_brick >= 0
                    source_position = np.zeros_like(source_brick)
                    source_position[valid] = (
                        source_brick[valid] * 64
                        + np.broadcast_to(source_lane[oid], source_brick.shape)[valid]
                    )
                    gathered = []
                    for split in x_split:
                        values = np.zeros(source_brick.shape, dtype=np.float32)
                        values[valid] = split[source_position[valid], in_component]
                        gathered.append(values)
                    a_by_output = []
                    for out_component in range(3):
                        out_tiles = out_component * operator.ngroup + groups
                        a_by_output.append(
                            tuple(
                                _bf16_values(coefficients[level, out_tiles, term, :])
                                for level in range(3)
                            )
                        )
                    prepared_terms.append((tuple(gathered), tuple(a_by_output)))

                for product_fidelity, fidelity_products in fidelity_product_groups:
                    if not fidelity_products:
                        continue
                    for gathered, a_by_output in prepared_terms:
                        for out_component in range(3):
                            accumulator = y_box[positions, out_component].reshape(
                                group1 - group0,
                                TILE_ELEMENTS,
                            )
                            a_split = a_by_output[out_component]
                            for a_level, b_level in fidelity_products:
                                for phase in _bf16_fidelity_phases(
                                    a_split[a_level],
                                    gathered[b_level],
                                    product_fidelity,
                                ):
                                    accumulator = accumulator + phase
                                    if prefix_extrema is not None:
                                        prefix_extrema["minimum"] = min(
                                            prefix_extrema["minimum"],
                                            float(accumulator.min()),
                                        )
                                        prefix_extrema["maximum"] = max(
                                            prefix_extrema["maximum"],
                                            float(accumulator.max()),
                                        )
                            y_box[positions, out_component] = accumulator.reshape(-1)
            continue
        for stream_index, term in enumerate(ordered_terms):
            oid = term // 3
            in_component = term - oid * 3
            did = np.broadcast_to(delta_lane[oid], brick_for_lane.shape)
            source_brick = np.asarray(brick_nbr[brick_for_lane, did], dtype=np.int64)
            valid = source_brick >= 0
            source_position = np.zeros_like(source_brick)
            source_position[valid] = source_brick[valid] * 64 + np.broadcast_to(source_lane[oid], source_brick.shape)[valid]
            gathered = []
            for split in x_split:
                values = np.zeros(source_brick.shape, dtype=np.float32)
                values[valid] = split[source_position[valid], in_component]
                gathered.append(values)
            for out_component in range(3):
                out_tiles = out_component * operator.ngroup + groups
                ah = _bf16_values(coefficients[0, out_tiles, term, :])
                am = _bf16_values(coefficients[1, out_tiles, term, :])
                al = _bf16_values(coefficients[2, out_tiles, term, :])
                correction = affine_corner_corrections.get((out_component, term))
                if correction is not None:
                    al = _bf16_values(_bf16_bits(al + correction))
                a_split = (ah, am, al)
                if coefficient_quantum != 0.0:
                    canonical = (
                        np.rint(
                            (
                                ah.astype(np.float64)
                                + am.astype(np.float64)
                                + al.astype(np.float64)
                            )
                            / coefficient_quantum
                        )
                        * coefficient_quantum
                    ).astype(np.float32)
                    a_split = tuple(
                        _bf16_values(bits) for bits in split_bf16x3(canonical)
                    )
                # Match the LLK exactly: every cross-product is accumulated into the persistent fp32
                # destination before the next product/term. Summing a temporary contribution first changes
                # rounding under cancellation and is not a faithful device oracle.
                accumulator_store = scratch_box if chunked else y_box
                assert accumulator_store is not None
                accumulator = accumulator_store[positions, out_component].reshape(
                    group1 - group0,
                    TILE_ELEMENTS,
                )
                # The run52 five-product device path deliberately places
                # a_lo*x_hi before a_mid*x_mid when x-low is omitted.  Keep
                # that established T5 order, then independently remove the
                # a-low product for affine-fold terms.
                b_selected_products = (
                    products
                    if low_term_set is None or term in low_term_set
                    else no_blow_products
                )
                term_products = tuple(
                    product
                    for product in b_selected_products
                    if not (
                        product == (2, 0)
                        and a_low_term_set is not None
                        and term not in a_low_term_set
                    )
                )
                for a_level, b_level in term_products:
                    product = (a_level, b_level)
                    if fixed_hifi2_exact_leading and a_level + b_level <= 1:
                        product_phases = _bf16_hifi2_exact_product_phases(
                            a_split[a_level],
                            gathered[b_level],
                        )
                    else:
                        product_fidelity = (
                            2
                            if fixed_hifi2_exact_leading or product in hifi2_product_set
                            else 3
                            if product in hifi3_product_set
                            else math_fidelity
                        )
                        product_phases = _bf16_fidelity_phases(
                            a_split[a_level],
                            gathered[b_level],
                            product_fidelity,
                        )
                    for phase in product_phases:
                        accumulator = accumulator + phase
                        if prefix_extrema is not None:
                            prefix_extrema["minimum"] = min(
                                prefix_extrema["minimum"],
                                float(accumulator.min()),
                            )
                            prefix_extrema["maximum"] = max(
                                prefix_extrema["maximum"],
                                float(accumulator.max()),
                            )
                chunk_boundary = chunked and (
                    (stream_index + 1) % accumulator_chunk_terms == 0
                    or stream_index + 1 == 81
                )
                if chunk_boundary:
                    contribution = accumulator - bias
                    main = y_box[positions, out_component].reshape(
                        group1 - group0,
                        TILE_ELEMENTS,
                    )
                    y_box[positions, out_component] = (main + contribution).reshape(-1)
                    scratch_box[positions, out_component] = bias
                else:
                    accumulator_store[positions, out_component] = accumulator.reshape(-1)
    result = y_box[node_to_pos]
    return result if chunked else result - bias


def _vectors(dump: FineDump, names: Sequence[str], seed: int) -> Iterator[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    for name in names:
        if name == "random":
            yield name, rng.standard_normal((dump.nb, 3))
        elif name == "smooth":
            coords = np.asarray(dump.ijk, dtype=np.float64)
            scale = np.maximum(np.asarray((dump.nx - 1, dump.ny - 1, dump.nz - 1), dtype=np.float64), 1.0)
            q = 2.0 * coords / scale - 1.0
            yield name, np.column_stack((q[:, 0] + 0.25 * q[:, 1], q[:, 1] - 0.2 * q[:, 2], q[:, 2] + 0.1 * q[:, 0]))
        elif name == "boundary":
            degree = np.diff(np.asarray(dump.ptr, dtype=np.int64))
            vector = np.zeros((dump.nb, 3), dtype=np.float64)
            mask = degree < 27
            vector[mask] = rng.standard_normal((int(mask.sum()), 3))
            yield name, vector
        elif name == "constant":
            yield name, np.ones((dump.nb, 3), dtype=np.float64)
        else:
            raise ValueError(f"unknown verification vector {name!r}")


def verify(
    fine_path: Path,
    brick_path: Path,
    *,
    vector_names: Sequence[str] = ("random", "smooth", "boundary", "constant"),
    seed: int = 351,
    group_chunk: int = 8,
    b_splits: int = 3,
    b_low_terms: Sequence[int] | None = None,
    a_low_terms: Sequence[int] | None = None,
    a_low_affine_corner_fold: bool = False,
    accumulator_bias: float = 0.0,
    accumulator_bias_scale: float = 0.0,
    accumulator_chunk_terms: int = 0,
    term_order: str = "natural",
    math_fidelity: int = 4,
    hifi3_products: Sequence[tuple[int, int]] | None = None,
    hifi2_products: Sequence[tuple[int, int]] | None = None,
    fidelity_batch_terms: int = 0,
    fixed_hifi2_exact_leading: bool = False,
    compensated_product_order: Sequence[tuple[int, int]] | None = None,
    compensated_product_order_no_blow: Sequence[tuple[int, int]] | None = None,
    coefficient_quantum: float = 0.0,
    l2_tolerance: float = 2.0e-5,
    max_tolerance: float = 5.0e-5,
) -> dict[str, Any]:
    if accumulator_bias != 0.0 and accumulator_bias_scale != 0.0:
        raise ValueError("accumulator_bias and accumulator_bias_scale are mutually exclusive")
    if accumulator_bias_scale < 0.0:
        raise ValueError("accumulator_bias_scale must be nonnegative")
    started = time.time()
    dump = open_fine_dump(fine_path)
    operator = BrickOperator(brick_path)
    if (operator.nb, operator.nblk, operator.nx, operator.ny, operator.nz) != (dump.nb, dump.nblk, dump.nx, dump.ny, dump.nz):
        raise ValueError("brick operator header does not match fine dump")
    slot_to_node = np.asarray(operator.section("slot_to_node"), dtype=np.int64).reshape(-1)
    node_to_pos = np.asarray(operator.section("node_to_pos"), dtype=np.int64)
    active_positions = np.flatnonzero(slot_to_node >= 0)
    roundtrip_ok = (
        len(active_positions) == dump.nb
        and np.array_equal(slot_to_node[node_to_pos], np.arange(dump.nb, dtype=np.int64))
        and np.array_equal(np.sort(node_to_pos), active_positions)
    )
    occupancy = np.asarray(operator.section("occupancy"), dtype=np.uint64)
    # Keep the verifier usable with the Python shipped in older tt-metal
    # environments, where int.bit_count() is not available.
    occupancy_count = sum(bin(int(value)).count("1") for value in occupancy)
    metadata_ok = roundtrip_ok and occupancy_count == dump.nb
    vector_reports: list[dict[str, Any]] = []
    for name, vector in _vectors(dump, vector_names, seed):
        effective_bias = float(accumulator_bias)
        vector_max_absolute = float(np.max(np.abs(vector)))
        if accumulator_bias_scale != 0.0 and vector_max_absolute != 0.0:
            vector_scale = 2.0 ** math.ceil(math.log2(vector_max_absolute))
            effective_bias = float(np.float32(accumulator_bias_scale * vector_scale))
        t0 = time.time()
        reference = bcsr_apply(dump, vector)
        reference_seconds = time.time() - t0
        t1 = time.time()
        prefix_extrema: dict[str, float] = {}
        candidate = brick_apply(
            operator,
            vector,
            group_chunk=group_chunk,
            b_splits=b_splits,
            b_low_terms=b_low_terms,
            a_low_terms=a_low_terms,
            a_low_affine_corner_fold=a_low_affine_corner_fold,
            accumulator_bias=effective_bias,
            accumulator_chunk_terms=accumulator_chunk_terms,
            term_order=term_order,
            math_fidelity=math_fidelity,
            hifi3_products=hifi3_products,
            hifi2_products=hifi2_products,
            fidelity_batch_terms=fidelity_batch_terms,
            fixed_hifi2_exact_leading=fixed_hifi2_exact_leading,
            compensated_product_order=compensated_product_order,
            compensated_product_order_no_blow=compensated_product_order_no_blow,
            coefficient_quantum=coefficient_quantum,
            prefix_extrema=prefix_extrema,
        ).astype(np.float64)
        candidate_seconds = time.time() - t1
        difference = candidate - reference
        ref_l2 = float(np.linalg.norm(reference.ravel()))
        ref_max = float(np.max(np.abs(reference)))
        abs_l2 = float(np.linalg.norm(difference.ravel()))
        abs_max = float(np.max(np.abs(difference)))
        l2_relative = abs_l2 / max(ref_l2, 1.0e-300)
        max_relative = abs_max / max(ref_max, 1.0e-300)
        # Constant/rigid-like vectors may be true cancellation vectors; report a scale-aware absolute ratio too.
        source_scale = float(np.max(np.abs(vector))) * float(np.max(np.abs(dump.val))) * max(int(np.diff(dump.ptr).max()), 1)
        max_over_term_scale = abs_max / max(source_scale, 1.0e-300)
        passed = (l2_relative <= l2_tolerance and max_relative <= max_tolerance) or (
            name == "constant" and max_over_term_scale <= 2.0e-6
        )
        vector_reports.append(
            {
                "name": name,
                "l2_relative": l2_relative,
                "max_relative": max_relative,
                "max_absolute": abs_max,
                "max_error_over_term_scale": max_over_term_scale,
                "reference_seconds": reference_seconds,
                "brick_oracle_seconds": candidate_seconds,
                "vector_max_absolute": vector_max_absolute,
                "accumulator_bias": effective_bias,
                "accumulator_prefix_extrema": prefix_extrema,
                "pass": passed,
            }
        )
    return {
        "schema": "tt_gmg_brick_operator_verification_v1",
        "device_mac": {
            "a_bf16_splits": 3,
            "b_bf16_splits": b_splits,
            "cross_terms": (
                6
                if b_splits == 3
                and (b_low_terms is None or len(set(b_low_terms)) == 81)
                and (a_low_terms is None or len(set(a_low_terms)) == 81)
                else 5
                if b_splits == 2
                and (a_low_terms is None or len(set(a_low_terms)) == 81)
                else "mixed_low_products"
            ),
            "b_low_terms": None if b_low_terms is None else sorted(set(int(term) for term in b_low_terms)),
            "a_low_terms": None if a_low_terms is None else sorted(set(int(term) for term in a_low_terms)),
            "a_low_affine_corner_fold": a_low_affine_corner_fold,
            "accumulator_bias": accumulator_bias,
            "accumulator_bias_scale": accumulator_bias_scale,
            "accumulator_chunk_terms": accumulator_chunk_terms,
            "term_order": term_order,
            "math_fidelity": math_fidelity,
            "hifi3_products": (
                []
                if hifi3_products is None
                else [list(pair) for pair in sorted(set(hifi3_products))]
            ),
            "hifi2_products": (
                []
                if hifi2_products is None
                else [list(pair) for pair in sorted(set(hifi2_products))]
            ),
            "fidelity_batch_terms": fidelity_batch_terms,
            "fixed_hifi2_exact_leading": fixed_hifi2_exact_leading,
            "compensated_product_order": (
                None
                if compensated_product_order is None
                else [list(pair) for pair in compensated_product_order]
            ),
            "compensated_product_order_no_blow": (
                None
                if compensated_product_order_no_blow is None
                else [list(pair) for pair in compensated_product_order_no_blow]
            ),
            "coefficient_quantum": coefficient_quantum,
            "fixed_hifi2_mac_calls_per_compensated_term": (
                9 if fixed_hifi2_exact_leading else None
            ),
            "fixed_hifi2_fidelity_phases_per_compensated_term": (
                18 if fixed_hifi2_exact_leading else None
            ),
        },
        "fine_dump": str(dump.path),
        "brick_operator": str(operator.path),
        "metadata": {
            "roundtrip_ok": roundtrip_ok,
            "occupancy_count": occupancy_count,
            "active_nodes": dump.nb,
            "pass": metadata_ok,
        },
        "vectors": vector_reports,
        "thresholds": {"l2_relative": l2_tolerance, "max_relative": max_tolerance, "constant_max_error_over_term_scale": 2.0e-6},
        "pass": metadata_ok and all(item["pass"] for item in vector_reports),
        "verification_seconds": time.time() - started,
    }


def serialize_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serialize a fine BCSR dump into TT brick-major format v1")
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--brick-size", type=int, default=4)
    parser.add_argument("--row-chunk", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args(argv)
    manifest = serialize(
        args.fine,
        args.output,
        args.manifest,
        brick_size=args.brick_size,
        row_chunk=args.row_chunk,
        force=args.force,
        hash_files=not args.skip_hash,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def verify_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify TT brick-major format against the original BCSR dump")
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--brick", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--vectors", default="random,smooth,boundary,constant")
    parser.add_argument("--seed", type=int, default=351)
    parser.add_argument("--group-chunk", type=int, default=8)
    parser.add_argument("--b-splits", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--b-low-terms",
        help="comma-separated stream term ids that retain ah*x_lo; omitted means all terms",
    )
    parser.add_argument(
        "--a-low-terms",
        help="comma-separated stream term ids that retain a_lo*x_hi; omitted means all terms",
    )
    parser.add_argument(
        "--a-low-affine-corner-fold",
        action="store_true",
        help=(
            "fold omitted corner a-low residuals into signed face terms and "
            "the center so affine fields are preserved before BF16 re-rounding"
        ),
    )
    parser.add_argument("--accumulator-bias", type=float, default=0.0)
    parser.add_argument("--accumulator-bias-scale", type=float, default=0.0)
    parser.add_argument("--accumulator-chunk-terms", type=int, default=0)
    parser.add_argument(
        "--term-order",
        choices=("natural", "center_first", "reverse"),
        default="natural",
    )
    parser.add_argument(
        "--math-fidelity",
        type=int,
        choices=(0, 2, 3, 4),
        default=4,
        help="Wormhole BF16 ELWMUL fidelity phase count",
    )
    parser.add_argument(
        "--hifi3-products",
        help="comma-separated a_level:b_level products to evaluate at HiFi3",
    )
    parser.add_argument(
        "--hifi2-products",
        help="comma-separated a_level:b_level products to evaluate at HiFi2",
    )
    parser.add_argument(
        "--fidelity-batch-terms",
        type=int,
        default=0,
        help="group this many stream terms by fidelity before changing the math MOP",
    )
    parser.add_argument(
        "--fixed-hifi2-exact-leading",
        action="store_true",
        help=(
            "use one fixed HiFi2 MOP: reconstruct degree-0/1 products from "
            "top+residual B tiles and evaluate degree-2 products at HiFi2"
        ),
    )
    parser.add_argument(
        "--compensated-product-order",
        help=(
            "comma-separated a_level:b_level permutation used for all six "
            "bf16x3 products; intended for source-reuse scheduling oracles"
        ),
    )
    parser.add_argument(
        "--compensated-product-order-no-blow",
        help=(
            "comma-separated a_level:b_level permutation used by terms that "
            "omit A_high*B_low"
        ),
    )
    parser.add_argument(
        "--coefficient-quantum",
        type=float,
        default=0.0,
        help=(
            "round reconstructed coefficient scalars to this quantum, then "
            "resplit to BF16x3; zero preserves the serialized operator"
        ),
    )
    args = parser.parse_args(argv)
    b_low_terms = (
        None
        if args.b_low_terms is None
        else tuple(
            int(item.strip())
            for item in args.b_low_terms.split(",")
            if item.strip()
        )
    )
    a_low_terms = (
        None
        if args.a_low_terms is None
        else tuple(
            int(item.strip())
            for item in args.a_low_terms.split(",")
            if item.strip()
        )
    )
    hifi3_products = (
        None
        if args.hifi3_products is None
        else tuple(
            tuple(int(part) for part in item.strip().split(":"))
            for item in args.hifi3_products.split(",")
            if item.strip()
        )
    )
    if hifi3_products is not None and any(len(pair) != 2 for pair in hifi3_products):
        parser.error("--hifi3-products entries must use a_level:b_level")
    hifi2_products = (
        None
        if args.hifi2_products is None
        else tuple(
            tuple(int(part) for part in item.strip().split(":"))
            for item in args.hifi2_products.split(",")
            if item.strip()
        )
    )
    if hifi2_products is not None and any(len(pair) != 2 for pair in hifi2_products):
        parser.error("--hifi2-products entries must use a_level:b_level")
    compensated_product_order = (
        None
        if args.compensated_product_order is None
        else tuple(
            tuple(int(part) for part in item.strip().split(":"))
            for item in args.compensated_product_order.split(",")
            if item.strip()
        )
    )
    if compensated_product_order is not None and any(
        len(pair) != 2 for pair in compensated_product_order
    ):
        parser.error("--compensated-product-order entries must use a_level:b_level")
    compensated_product_order_no_blow = (
        None
        if args.compensated_product_order_no_blow is None
        else tuple(
            tuple(int(part) for part in item.strip().split(":"))
            for item in args.compensated_product_order_no_blow.split(",")
            if item.strip()
        )
    )
    if compensated_product_order_no_blow is not None and any(
        len(pair) != 2 for pair in compensated_product_order_no_blow
    ):
        parser.error("--compensated-product-order-no-blow entries must use a_level:b_level")
    report = verify(
        args.fine,
        args.brick,
        vector_names=[item.strip() for item in args.vectors.split(",") if item.strip()],
        seed=args.seed,
        group_chunk=args.group_chunk,
        b_splits=args.b_splits,
        b_low_terms=b_low_terms,
        a_low_terms=a_low_terms,
        a_low_affine_corner_fold=args.a_low_affine_corner_fold,
        accumulator_bias=args.accumulator_bias,
        accumulator_bias_scale=args.accumulator_bias_scale,
        accumulator_chunk_terms=args.accumulator_chunk_terms,
        term_order=args.term_order,
        math_fidelity=args.math_fidelity,
        hifi3_products=hifi3_products,
        hifi2_products=hifi2_products,
        fidelity_batch_terms=args.fidelity_batch_terms,
        fixed_hifi2_exact_leading=args.fixed_hifi2_exact_leading,
        compensated_product_order=compensated_product_order,
        compensated_product_order_no_blow=compensated_product_order_no_blow,
        coefficient_quantum=args.coefficient_quantum,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["pass"] else 1
