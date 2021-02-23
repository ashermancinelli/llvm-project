#ifndef FORTRAN_TEST_RUNTIME_TESTING_H_
#define FORTRAN_TEST_RUNTIME_TESTING_H_

#include "gtest/gtest.h"
#include <cstddef>

// Number of runtime crashes discoverd by the crash handler registered in the
// `StartTests` function below.
int GetNumRuntimeCrashes();

// Registers a callback in the fortran runtime to register errors in tests.
// When `rethrow` is false, you must check for runtime crashes yourself.
void StartTests();

// Resets static variables that should not carry over between tests
void EndTests();

// Defines a CHARACTER object with padding when needed
void SetCharacter(char *, std::size_t, const char *);

// Convenience assertion to ensure no crashes from within the fortran runtime
// have occured.
#define ASSERT_NO_CRASHES() \
  ASSERT_EQ(GetNumRuntimeCrashes(), 0) \
      << "Encountered " << GetNumRuntimeCrashes() << " runtime crashes! "

#endif // FORTRAN_TEST_RUNTIME_TESTING_H_
