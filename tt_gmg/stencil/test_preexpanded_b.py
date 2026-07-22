import unittest

import numpy as np

from build_preexpanded_b import (
    BRICKS_PER_GROUP,
    HALO_PLANE_VALUES,
    HALO_WORDS_PER_BRICK,
    MATERIALIZE_WORD_OFFSETS,
    TERMS,
    TILE_VALUES,
    _term_word_indices,
)


class PreexpandedBTest(unittest.TestCase):
    def test_vectorized_word_map_matches_scalar_materializer(self) -> None:
        plane_words = np.arange(HALO_PLANE_VALUES // 4, dtype="<u8")
        indices = _term_word_indices()
        self.assertEqual(indices.shape, (TERMS, BRICKS_PER_GROUP, 16))
        for term in (0, 1, 13 * 3 + 2, TERMS - 1):
            offset = term // 3
            di = offset // 9
            dj = (offset // 3) % 3
            dk = offset % 3
            window = (di * 6 + dj) * 3 + dk
            scalar = []
            for brick in range(BRICKS_PER_GROUP):
                source = brick * HALO_WORDS_PER_BRICK + window
                scalar.extend(
                    plane_words[source + MATERIALIZE_WORD_OFFSETS].tolist()
                )
            vectorized = plane_words[indices[term].reshape(-1)]
            np.testing.assert_array_equal(vectorized, np.asarray(scalar, dtype="<u8"))
            self.assertEqual(vectorized.view("<u2").size, TILE_VALUES)


if __name__ == "__main__":
    unittest.main()
