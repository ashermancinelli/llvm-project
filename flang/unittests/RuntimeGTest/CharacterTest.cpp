// Basic sanity tests of CHARACTER API; exhaustive testing will be done
// in Fortran.

#include "../../runtime/character.h"
#include "RuntimeTesting.h"
#include "gtest/gtest.h"
#include <array>
#include <cstring>
#include <tuple>

using namespace Fortran::runtime;

struct CharacterTests : RuntimeTestFixture {};

TEST_F(CharacterTests, AppendAndPad) {
  for (std::size_t limit{0}; limit < 8; ++limit) {
    char x[8];
    std::size_t xLen{0};
    std::memset(x, 0, sizeof x);
    xLen = RTNAME(CharacterAppend1)(x, limit, xLen, "abc", 3);
    xLen = RTNAME(CharacterAppend1)(x, limit, xLen, "DE", 2);
    RTNAME(CharacterPad1)(x, limit, xLen);
    ASSERT_LE(xLen, limit) << "xLen " << xLen << ">" << limit;
    if (x[limit]) {
      EXPECT_TRUE(false) << "x[" << limit << "]='" << x[limit] << "'\n";
      x[limit] = '\0';
    }
    ASSERT_FALSE(std::memcmp(x, "abcDE   ", limit)) << "x = '" << x << "'";
  }
}

using ParamT = std::tuple<const char *, const char *, int, int, int>;

struct CharacterComparisonTestsFixture
    : public RuntimeTestFixture,
      public ::testing::WithParamInterface<ParamT> {
  void SetUp() {
    RuntimeTestFixture::SetUp();
    std::tie(x, y, xBytes, yBytes, expect) = GetParam();
  }
  void SwapParams() {
    std::swap(x, y);
    std::swap(xBytes, yBytes);
    expect = -expect;
  }
  void DoCharacterComparison() {
    int cmp{RTNAME(CharacterCompareScalar1)(x, y, xBytes, yBytes)};
    char buf[2][8];
    std::memset(buf, 0, sizeof buf);
    std::memcpy(buf[0], x, xBytes);
    std::memcpy(buf[1], y, yBytes);
    ASSERT_EQ(cmp, expect) << "compare '" << buf[0] << "'(" << xBytes
                           << ") to '" << buf[1] << "'(" << yBytes << "), got "
                           << cmp << ", should be " << expect << '\n';
  }
  const char *x;
  const char *y;
  int xBytes;
  int yBytes;
  int expect;
};

TEST_P(CharacterComparisonTestsFixture, CompareCharacters) {
  DoCharacterComparison();
  SwapParams();
  DoCharacterComparison();
}

INSTANTIATE_TEST_CASE_P(CharacterComparisonTests,
    CharacterComparisonTestsFixture,
    ::testing::Values(std::make_tuple("abc", "abc", 3, 3, 0),
        std::make_tuple("abc", "def", 3, 3, -1),
        std::make_tuple("ab ", "abc", 3, 2, 0),
        std::make_tuple("abc", "abc", 2, 3, -1)), );
