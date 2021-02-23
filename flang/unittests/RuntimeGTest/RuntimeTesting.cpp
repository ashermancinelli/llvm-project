#include "RuntimeTesting.h"
#include "../../runtime/terminator.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

static int failures{0};

static bool hasRunStartTests{false};

// Override the Fortran runtime's Crash() for testing purposes
static void CatchCrash(
    const char *sourceFile, int sourceLine, const char *message, va_list &ap) {
  char buffer[1000];
  std::vsnprintf(buffer, sizeof buffer, message, ap);
  va_end(ap);
  llvm::errs() << (sourceFile ? sourceFile : "unknown source file") << '('
               << sourceLine << "): CRASH: " << buffer << '\n';
  failures++;
}

int GetNumRuntimeCrashes() {
  return failures;
}

void StartTests() {
  if (hasRunStartTests)
    return;
  Fortran::runtime::Terminator::RegisterCrashHandler(CatchCrash);
  hasRunStartTests = true;
}

void EndTests() {
  ASSERT_NO_CRASHES();
  failures = 0;
}

void SetCharacter(char *to, std::size_t n, const char *from) {
  auto len{std::strlen(from)};
  std::memcpy(to, from, std::min(len, n));
  if (len < n) {
    std::memset(to + len, ' ', n - len);
  }
}
