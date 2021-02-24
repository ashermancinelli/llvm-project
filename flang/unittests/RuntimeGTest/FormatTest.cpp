// Tests basic FORMAT string traversal

#include "RuntimeTesting.h"
#include "gtest/gtest.h"
#include "../runtime/format-implementation.h"
#include "../runtime/io-error.h"
#include <cstdarg>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;
using namespace std::literals::string_literals;

using Results = std::vector<std::string>;

// A test harness context for testing FormatControl
class TestFormatContext : public IoErrorHandler {
public:
  using CharType = char;
  TestFormatContext() : IoErrorHandler{"format.cpp", 1} {}
  bool Emit(const char *, std::size_t);
  bool Emit(const char16_t *, std::size_t);
  bool Emit(const char32_t *, std::size_t);
  bool AdvanceRecord(int = 1);
  void HandleRelativePosition(std::int64_t);
  void HandleAbsolutePosition(std::int64_t);
  void Report(const DataEdit &);
  void Check(Results &);

  Results results;
  MutableModes &mutableModes() { return mutableModes_; }

private:
  MutableModes mutableModes_;
};

bool TestFormatContext::Emit(const char *s, std::size_t len) {
  std::string str{s, len};
  results.push_back("'"s + str + '\'');
  return true;
}
bool TestFormatContext::Emit(const char16_t *, std::size_t) {
  Crash("TestFormatContext::Emit(const char16_t *) called");
  return false;
}
bool TestFormatContext::Emit(const char32_t *, std::size_t) {
  Crash("TestFormatContext::Emit(const char32_t *) called");
  return false;
}

bool TestFormatContext::AdvanceRecord(int n) {
  while (n-- > 0) {
    results.emplace_back("/");
  }
  return true;
}

void TestFormatContext::HandleAbsolutePosition(std::int64_t n) {
  results.push_back("T"s + std::to_string(n));
}

void TestFormatContext::HandleRelativePosition(std::int64_t n) {
  if (n < 0) {
    results.push_back("TL"s + std::to_string(-n));
  } else {
    results.push_back(std::to_string(n) + 'X');
  }
}

void TestFormatContext::Report(const DataEdit &edit) {
  std::string str{edit.descriptor};
  if (edit.repeat != 1) {
    str = std::to_string(edit.repeat) + '*' + str;
  }
  if (edit.variation) {
    str += edit.variation;
  }
  if (edit.width) {
    str += std::to_string(*edit.width);
  }
  if (edit.digits) {
    str += "."s + std::to_string(*edit.digits);
  }
  if (edit.expoDigits) {
    str += "E"s + std::to_string(*edit.expoDigits);
  }
  // modes?
  results.push_back(str);
}

void TestFormatContext::Check(Results &expect) {
  std::stringstream message;
  message << "expected:";
  for (const std::string &s : expect) {
    message << ' ' << s;
  }
  message << ", got:";
  for (const std::string &s : results) {
    message << ' ' << s;
  }
  EXPECT_EQ(expect, results) << message.str();
  expect.clear();
  results.clear();
}

// Tuple of types over which the test cases are parameterized
using Params = std::tuple<int, const char *, Results, int>;

// Fixture which constructs needed components for each test, includeing a
// TestFormatContext
struct FormatTestsFixture : ::testing::TestWithParam<Params> {
  virtual void SetUp() {
    StartTests();
    std::tie(n, format, expect, repeat) = GetParam();
  }
  virtual void TearDown() {
    ASSERT_NO_CRASHES() << " In teardown of FormatTestsFixture " << __FILE__
                        << __LINE__;
    EndTests();
  }

protected:
  int n;
  const char *format;
  Results expect;
  int repeat;
  TestFormatContext context;
};

TEST_P(FormatTestsFixture, FormatStringTraversal) {
  FormatControl<decltype(context)> control{
      context, format, std::strlen(format)};

  for (int j{0}; j < n; ++j) {
    context.Report(/*edit=*/control.GetNextDataEdit(context, repeat));
    ASSERT_NO_CRASHES() << __FILE__ << " " << __LINE__;
  }

  control.Finish(context);
  int iostat{context.GetIoStat()};
  ASSERT_FALSE(iostat) << "GetIoStat() == " << iostat;

  context.Check(expect);
  ASSERT_NO_CRASHES() << __FILE__ << " " << __LINE__;
}

// Trailing ',' in this macro suppresses a warning which is emmitted even when
// the macro is used correctly
INSTANTIATE_TEST_CASE_P(FormatTests, FormatTestsFixture,
    ::testing::Values(
        std::make_tuple(1, "('PI=',F9.7)", Results{"'PI='", "F9.7"}, 1),
        std::make_tuple(1, "(3HPI=F9.7)", Results{"'PI='", "F9.7"}, 1),
        std::make_tuple(1, "(3HPI=/F9.7)", Results{"'PI='", "/", "F9.7"}, 1),
        std::make_tuple(2, "('PI=',F9.7)",
            Results{"'PI='", "F9.7", "/", "'PI='", "F9.7"}, 1),
        std::make_tuple(2, "(2('PI=',F9.7),'done')",
            Results{"'PI='", "F9.7", "'PI='", "F9.7", "'done'"}, 1),
        std::make_tuple(2, "(3('PI=',F9.7,:),'tooFar')",
            Results{"'PI='", "F9.7", "'PI='", "F9.7"}, 1),
        std::make_tuple(2, "(*('PI=',F9.7,:),'tooFar')",
            Results{"'PI='", "F9.7", "'PI='", "F9.7"}, 1),
        std::make_tuple(1, "(3F9.7)", Results{"2*F9.7"}, 2)), );
