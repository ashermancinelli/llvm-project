//===-- flang/unittests/RuntimeGTest/CharacterTest.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../runtime/character.h"
#include "gtest/gtest.h"
#include <cstring>
#include <functional>
#include <tuple>
#include <vector>

using namespace Fortran::runtime;

TEST(CharacterTests, AppendAndPad) {
  static constexpr int limitMax{8};
  static char buffer[limitMax];
  static std::size_t offset{0};
  for (std::size_t limit{0}; limit < limitMax; ++limit, offset = 0) {
    std::memset(buffer, 0, sizeof buffer);

    // Ensure appending characters does not overrun the limit
    offset = RTNAME(CharacterAppend1)(buffer, limit, offset, "abc", 3);
    offset = RTNAME(CharacterAppend1)(buffer, limit, offset, "DE", 2);
    ASSERT_LE(offset, limit) << "offset " << offset << ">" << limit;

    // Ensure whitespace padding does not overrun limit, the string is still
    // null-terminated, and string matches the expected value up to the limit.
    RTNAME(CharacterPad1)(buffer, limit, offset);
    EXPECT_EQ(buffer[limit], '\0')
        << "buffer[" << limit << "]='" << buffer[limit] << "'";
    buffer[limit] = buffer[limit] ? '\0' : buffer[limit];
    ASSERT_EQ(std::memcmp(buffer, "abcDE   ", limit), 0)
        << "buffer = '" << buffer << "'";
  }
}

TEST(CharacterTests, CharacterAppend1Overrun) {
  static constexpr int bufferSize{4};
  static constexpr std::size_t limit{2};
  static char buffer[bufferSize];
  static std::size_t offset{0};
  std::memset(buffer, 0, sizeof buffer);
  offset = RTNAME(CharacterAppend1)(buffer, limit, offset, "1234", bufferSize);
  ASSERT_EQ(offset, limit) << "CharacterAppend1 did not halt at limit = "
                           << limit << ", but at offset = " << offset;
}

//------------------------------------------------------------------------------
/// Tests and infrastructure for character comparison functions
//------------------------------------------------------------------------------

template <typename CHAR>
using COMPARISON_FUNC =
    std::function<int(const CHAR *, const CHAR *, std::size_t, std::size_t)>;

static std::tuple<COMPARISON_FUNC<char>, COMPARISON_FUNC<char16_t>,
    COMPARISON_FUNC<char32_t>>
    ComparisonFuncs{
        RTNAME(CharacterCompareScalar1),
        RTNAME(CharacterCompareScalar2),
        RTNAME(CharacterCompareScalar4),
    };

// Types of _values_ over which comparison tests are parameterized
template <typename CHAR>
using COMPARISON_PARAMETER =
    std::tuple<const CHAR *, const CHAR *, int, int, int>;

static std::tuple<std::vector<COMPARISON_PARAMETER<char>>,
    std::vector<COMPARISON_PARAMETER<char16_t>>,
    std::vector<COMPARISON_PARAMETER<char32_t>>>
    ComparisonParameters{{
                             std::make_tuple("abc", "abc", 3, 3, 0),
                             std::make_tuple("abc", "def", 3, 3, -1),
                             std::make_tuple("ab ", "abc", 3, 2, 0),
                             std::make_tuple("abc", "abc", 2, 3, -1),
                         },
        {
            std::make_tuple(u"abc", u"abc", 3, 3, 0),
            std::make_tuple(u"abc", u"def", 3, 3, -1),
            std::make_tuple(u"ab ", u"abc", 3, 2, 0),
            std::make_tuple(u"abc", u"abc", 2, 3, -1),
        },
        {
            std::make_tuple(U"abc", U"abc", 3, 3, 0),
            std::make_tuple(U"abc", U"def", 3, 3, -1),
            std::make_tuple(U"ab ", U"abc", 3, 2, 0),
            std::make_tuple(U"abc", U"abc", 2, 3, -1),
        }};

template <typename CHAR>
struct CharacterComparisonTests : public ::testing::Test {
  CharacterComparisonTests()
      : parameters{std::get<std::vector<COMPARISON_PARAMETER<CHAR>>>(
            ComparisonParameters)},
        characterComparisonFunc{
            std::get<COMPARISON_FUNC<CHAR>>(ComparisonFuncs)} {}
  std::vector<COMPARISON_PARAMETER<CHAR>> parameters;
  COMPARISON_FUNC<CHAR> characterComparisonFunc;
};

using CHARACTERS = ::testing::Types<char, char16_t, char32_t>;
TYPED_TEST_CASE(CharacterComparisonTests, CHARACTERS);

TYPED_TEST(CharacterComparisonTests, CompareCharacters) {
  for (auto &[x, y, xBytes, yBytes, expect] : this->parameters) {
    int cmp{this->characterComparisonFunc(x, y, xBytes, yBytes)};
    TypeParam buf[2][8];
    std::memset(buf, 0, sizeof buf);
    std::memcpy(buf[0], x, xBytes);
    std::memcpy(buf[1], y, yBytes);
    ASSERT_EQ(cmp, expect) << "compare '" << x << "'(" << xBytes << ") to '"
                           << y << "'(" << yBytes << "), got " << cmp
                           << ", should be " << expect << '\n';

    // Perform the same test with the parameters reversed and the difference
    // negated
    std::swap(x, y);
    std::swap(xBytes, yBytes);
    expect = -expect;

    cmp = this->characterComparisonFunc(x, y, xBytes, yBytes);
    std::memset(buf, 0, sizeof buf);
    std::memcpy(buf[0], x, xBytes);
    std::memcpy(buf[1], y, yBytes);
    ASSERT_EQ(cmp, expect) << "compare '" << x << "'(" << xBytes << ") to '"
                           << y << "'(" << yBytes << "), got " << cmp
                           << ", should be " << expect << '\n';
  }
}

//------------------------------------------------------------------------------
/// Tests and infrastructure for Scan functions
//------------------------------------------------------------------------------

template <typename CHAR>
using SCAN_FUNC = std::function<int(
    const CHAR *, std::size_t, const CHAR *, std::size_t, bool)>;

static std::tuple<SCAN_FUNC<char>, SCAN_FUNC<char16_t>, SCAN_FUNC<char32_t>>
    ScanFuncs{
        RTNAME(Scan1),
        RTNAME(Scan2),
        RTNAME(Scan4),
    };

// Types of _values_ over which tests are parameterized
template <typename CHAR>
using SCAN_PARAMETER = std::tuple<const CHAR *, const CHAR *, bool, int>;

static std::tuple<std::vector<SCAN_PARAMETER<char>>,
    std::vector<SCAN_PARAMETER<char16_t>>,
    std::vector<SCAN_PARAMETER<char32_t>>>
    ScanParameters{{
                       std::make_tuple("abc", "abc", false, 1),
                       std::make_tuple("abc", "abc", true, 3),
                       std::make_tuple("abc", "cde", false, 3),
                       std::make_tuple("abc", "cde", true, 3),
                       std::make_tuple("abc", "x", false, 0),
                       std::make_tuple("", "x", false, 0),
                   },
        {
            std::make_tuple(u"abc", u"abc", false, 1),
            std::make_tuple(u"abc", u"abc", true, 3),
            std::make_tuple(u"abc", u"cde", false, 3),
            std::make_tuple(u"abc", u"cde", true, 3),
            std::make_tuple(u"abc", u"x", false, 0),
            std::make_tuple(u"", u"x", false, 0),
        },
        {
            std::make_tuple(U"abc", U"abc", false, 1),
            std::make_tuple(U"abc", U"abc", true, 3),
            std::make_tuple(U"abc", U"cde", false, 3),
            std::make_tuple(U"abc", U"cde", true, 3),
            std::make_tuple(U"abc", U"x", false, 0),
            std::make_tuple(U"", U"x", false, 0),
        }};

template <typename CHAR> struct CharacterScanTests : public ::testing::Test {
  CharacterScanTests()
      : parameters{std::get<std::vector<SCAN_PARAMETER<CHAR>>>(ScanParameters)},
        characterScanFunc{std::get<SCAN_FUNC<CHAR>>(ScanFuncs)} {}
  std::vector<SCAN_PARAMETER<CHAR>> parameters;
  SCAN_FUNC<CHAR> characterScanFunc;
};

// Type-parameterized over the same character types as CharacterComparisonTests
TYPED_TEST_CASE(CharacterScanTests, CHARACTERS);

TYPED_TEST(CharacterScanTests, ScanCharacters) {
  for (auto const &[str, set, back, expect] : this->parameters) {
    auto res{
        this->characterScanFunc(str, std::char_traits<TypeParam>::length(str),
            set, std::char_traits<TypeParam>::length(set), back)};
    ASSERT_EQ(res, expect) << "Scan(" << str << ',' << set << ",back=" << back
                           << "): got " << res << ", should be " << expect;
  }
}

//------------------------------------------------------------------------------
/// Tests and infrastructure for Verify functions
//------------------------------------------------------------------------------
template <typename CHAR>
using VERIFY_FUNC = std::function<int(
    const CHAR *, std::size_t, const CHAR *, std::size_t, bool)>;

static std::tuple<VERIFY_FUNC<char>, VERIFY_FUNC<char16_t>,
    VERIFY_FUNC<char32_t>>
    VerifyFuncs{
        RTNAME(Verify1),
        RTNAME(Verify2),
        RTNAME(Verify4),
    };

// Types of _values_ over which tests are parameterized
template <typename CHAR>
using VERIFY_PARAMETER = std::tuple<const CHAR *, const CHAR *, bool, int>;

static std::tuple<std::vector<VERIFY_PARAMETER<char>>,
    std::vector<VERIFY_PARAMETER<char16_t>>,
    std::vector<VERIFY_PARAMETER<char32_t>>>
    VerifyParameters{{
                         std::make_tuple("abc", "abc", false, 0),
                         std::make_tuple("abc", "abc", true, 0),
                         std::make_tuple("abc", "cde", false, 1),
                         std::make_tuple("abc", "cde", true, 2),
                         std::make_tuple("abc", "x", false, 1),
                         std::make_tuple("", "x", false, 0),
                     },
        {
            std::make_tuple(u"abc", u"abc", false, 0),
            std::make_tuple(u"abc", u"abc", true, 0),
            std::make_tuple(u"abc", u"cde", false, 1),
            std::make_tuple(u"abc", u"cde", true, 2),
            std::make_tuple(u"abc", u"x", false, 1),
            std::make_tuple(u"", u"x", false, 0),
        },
        {
            std::make_tuple(U"abc", U"abc", false, 0),
            std::make_tuple(U"abc", U"abc", true, 0),
            std::make_tuple(U"abc", U"cde", false, 1),
            std::make_tuple(U"abc", U"cde", true, 2),
            std::make_tuple(U"abc", U"x", false, 1),
            std::make_tuple(U"", U"x", false, 0),
        }};

template <typename CHAR> struct CharacterVerifyTests : public ::testing::Test {
  CharacterVerifyTests()
      : parameters{std::get<std::vector<VERIFY_PARAMETER<CHAR>>>(
            VerifyParameters)},
        characterVerifyFunc{std::get<VERIFY_FUNC<CHAR>>(VerifyFuncs)} {}
  std::vector<VERIFY_PARAMETER<CHAR>> parameters;
  VERIFY_FUNC<CHAR> characterVerifyFunc;
};

// Type-parameterized over the same character types as CharacterComparisonTests
TYPED_TEST_CASE(CharacterVerifyTests, CHARACTERS);

TYPED_TEST(CharacterVerifyTests, VerifyCharacters) {
  for (auto const &[str, set, back, expect] : this->parameters) {
    auto res{
        this->characterVerifyFunc(str, std::char_traits<TypeParam>::length(str),
            set, std::char_traits<TypeParam>::length(set), back)};
    ASSERT_EQ(res, expect) << "Verify(" << str << ',' << set << ",back=" << back
                           << "): got " << res << ", should be " << expect;
  }
}
