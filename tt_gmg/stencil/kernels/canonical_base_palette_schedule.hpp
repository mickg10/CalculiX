#pragma once

#include <cstdint>

// Deterministic q1e-7 canonical-stencil schedule for Row236.
//
// Source signature:
//   canonical_base_signature_q1e7.i32
//   sha256 689ece8f730617ead3afc09459f721f531dee3008f06f88056525c339f3b5ac8
// Derived palette:
//   [-59656973, -22371364, -15361670, -5592841, -3952274, 0,
//      5592841,   7009694,  22371364, 29828486, 126472778] * 1e-7
//
// canonical_packed_layout.py and test_canonical_packed_kernel_contract.py
// independently reproduce and verify this table from the exported signature.
namespace canonical_packed_base {

constexpr uint32_t kTerms = 81;
constexpr uint32_t kOutputComponents = 3;
constexpr uint32_t kPaletteSize = 11;
constexpr uint32_t kPaletteSplits = 3;
constexpr uint32_t kPaletteTiles = kPaletteSize * kPaletteSplits;
constexpr uint8_t kZeroPaletteIndex = 5;
constexpr uint32_t kNonzeroScalarPlanes = 153;
constexpr uint32_t kRun52ProductsPerGroup = 864;

constexpr uint8_t kPaletteIndex[kTerms][kOutputComponents] = {
    {4, 3, 3},  // term 0
    {3, 4, 3},  // term 1
    {3, 3, 4},  // term 2
    {2, 1, 5},  // term 3
    {1, 2, 5},  // term 4
    {5, 5, 7},  // term 5
    {4, 3, 6},  // term 6
    {3, 4, 6},  // term 7
    {6, 6, 4},  // term 8
    {2, 5, 1},  // term 9
    {5, 7, 5},  // term 10
    {1, 5, 2},  // term 11
    {0, 5, 5},  // term 12
    {5, 9, 5},  // term 13
    {5, 5, 9},  // term 14
    {2, 5, 8},  // term 15
    {5, 7, 5},  // term 16
    {8, 5, 2},  // term 17
    {4, 6, 3},  // term 18
    {6, 4, 6},  // term 19
    {3, 6, 4},  // term 20
    {2, 8, 5},  // term 21
    {8, 2, 5},  // term 22
    {5, 5, 7},  // term 23
    {4, 6, 6},  // term 24
    {6, 4, 3},  // term 25
    {6, 3, 4},  // term 26
    {7, 5, 5},  // term 27
    {5, 2, 1},  // term 28
    {5, 1, 2},  // term 29
    {9, 5, 5},  // term 30
    {5, 0, 5},  // term 31
    {5, 5, 9},  // term 32
    {7, 5, 5},  // term 33
    {5, 2, 8},  // term 34
    {5, 8, 2},  // term 35
    {9, 5, 5},  // term 36
    {5, 9, 5},  // term 37
    {5, 5, 0},  // term 38
    {10, 5, 5},  // term 39
    {5, 10, 5},  // term 40
    {5, 5, 10},  // term 41
    {9, 5, 5},  // term 42
    {5, 9, 5},  // term 43
    {5, 5, 0},  // term 44
    {7, 5, 5},  // term 45
    {5, 2, 8},  // term 46
    {5, 8, 2},  // term 47
    {9, 5, 5},  // term 48
    {5, 0, 5},  // term 49
    {5, 5, 9},  // term 50
    {7, 5, 5},  // term 51
    {5, 2, 1},  // term 52
    {5, 1, 2},  // term 53
    {4, 6, 6},  // term 54
    {6, 4, 3},  // term 55
    {6, 3, 4},  // term 56
    {2, 8, 5},  // term 57
    {8, 2, 5},  // term 58
    {5, 5, 7},  // term 59
    {4, 6, 3},  // term 60
    {6, 4, 6},  // term 61
    {3, 6, 4},  // term 62
    {2, 5, 8},  // term 63
    {5, 7, 5},  // term 64
    {8, 5, 2},  // term 65
    {0, 5, 5},  // term 66
    {5, 9, 5},  // term 67
    {5, 5, 9},  // term 68
    {2, 5, 1},  // term 69
    {5, 7, 5},  // term 70
    {1, 5, 2},  // term 71
    {4, 3, 6},  // term 72
    {3, 4, 6},  // term 73
    {6, 6, 4},  // term 74
    {2, 1, 5},  // term 75
    {1, 2, 5},  // term 76
    {5, 5, 7},  // term 77
    {4, 3, 3},  // term 78
    {3, 4, 3},  // term 79
    {3, 3, 4},  // term 80
};

}  // namespace canonical_packed_base
