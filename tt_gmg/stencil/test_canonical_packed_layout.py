import unittest

import numpy as np

from canonical_packed_layout import (
    GROUP_BASE,
    GROUP_FALLBACK,
    GROUP_PADDING,
    LANES,
    _build_palette,
    build_packed_mapping,
)


class CanonicalPackedLayoutTest(unittest.TestCase):
    def test_mapping_is_bijective_and_separates_roles(self):
        source_lanes = 3 * LANES
        active = np.zeros(source_lanes, dtype=bool)
        active[: 2 * LANES + 17] = True
        base = np.zeros(source_lanes, dtype=np.uint8)
        base[np.flatnonzero(active)[::2]] = 1

        packed, inverse, roles, plan = build_packed_mapping(
            active,
            base,
            chips=2,
        )

        valid = packed.reshape(-1) >= 0
        sources = packed.reshape(-1)[valid]
        self.assertEqual(np.unique(sources).size, int(active.sum()))
        np.testing.assert_array_equal(
            inverse[sources],
            np.flatnonzero(valid).astype(np.int32),
        )
        self.assertEqual(plan["base_lanes"], int(base.sum()))
        self.assertEqual(
            plan["fallback_lanes"],
            int(active.sum() - base.sum()),
        )
        for chip in range(2):
            for group in range(packed.shape[1]):
                selected = packed[chip, group]
                selected = selected[selected >= 0]
                if roles[chip, group] == GROUP_BASE:
                    self.assertTrue(np.all(base[selected] == 1))
                elif roles[chip, group] == GROUP_FALLBACK:
                    self.assertTrue(np.all(base[selected] == 0))
                else:
                    self.assertEqual(roles[chip, group], GROUP_PADDING)

    def test_palette_round_trips_term_output_signature(self):
        signature = np.zeros((27, 3, 3), dtype="<i4")
        signature[0] = np.asarray(
            [[1, 0, -2], [0, 1, 0], [-2, 0, 3]],
            dtype="<i4",
        )
        palette, schedule, tiles = _build_palette(signature)
        reconstructed = palette[schedule].reshape(27, 3, 3).transpose(0, 2, 1)
        np.testing.assert_array_equal(reconstructed, signature)
        self.assertIn(0, palette)
        self.assertEqual(len(tiles), 3)
        self.assertEqual(tiles[0].shape, (palette.size, LANES))
        for tile_set in tiles:
            self.assertTrue(np.all(tile_set == tile_set[:, :1]))


if __name__ == "__main__":
    unittest.main()
