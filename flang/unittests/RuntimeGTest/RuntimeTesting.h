#ifndef LLVM_FLANG_UNITTESTS_RUNTIMEGTEST_RUNTIMETESTING_H
#define LLVM_FLANG_UNITTESTS_RUNTIMEGTEST_RUNTIMETESTING_H

#include "gtest/gtest.h"
#include <cstddef>

struct RuntimeTestFixture : ::testing::Test {
  RuntimeTestFixture();
  void SetUp();
  void TearDown();

private:
  bool IsCrashHandlerRegistered;
};

// Defines a CHARACTER object with padding when needed
void SetCharacter(char *, std::size_t, const char *);

#endif // LLVM_FLANG_UNITTESTS_RUNTIMEGTEST_RUNTIMETESTING_H
