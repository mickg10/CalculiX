import unittest

from canonical_packed_oracle import (
    PRODUCTS_WITH_BLOW,
    PRODUCTS_WITHOUT_BLOW,
    retains_blow,
)


class CanonicalPackedOracleTest(unittest.TestCase):
    def test_selective_blow_matches_run52_schedule(self):
        retained = [term for term in range(81) if retains_blow(term)]
        self.assertEqual(len(retained), 63)
        self.assertEqual(len(PRODUCTS_WITH_BLOW), 6)
        self.assertEqual(len(PRODUCTS_WITHOUT_BLOW), 5)
        self.assertEqual(
            set(PRODUCTS_WITH_BLOW) - set(PRODUCTS_WITHOUT_BLOW),
            {(0, 2)},
        )
        self.assertEqual(
            PRODUCTS_WITHOUT_BLOW,
            ((0, 0), (0, 1), (1, 0), (2, 0), (1, 1)),
        )


if __name__ == "__main__":
    unittest.main()
