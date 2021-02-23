// Basic sanity tests for list-directed input

#include "RuntimeTesting.h"
#include "gtest/gtest.h"
#include "../../runtime/descriptor.h"
#include "../../runtime/io-api.h"
#include "../../runtime/io-error.h"
#include <algorithm>
#include <cstring>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

TEST(InputTest, ListInput) {
  StartTests();
  char buffer[4][32];
  int j{0};
  for (const char *p : {"1 2 2*3  ,", ",6,,8,1*",
           "2*'abcdefghijklmnopqrstuvwxyzABC", "DEFGHIJKLMNOPQRSTUVWXYZ'"}) {
    SetCharacter(buffer[j++], sizeof buffer[0], p);
  }
  for (; j < 4; ++j) {
    SetCharacter(buffer[j], sizeof buffer[0], "");
  }

  StaticDescriptor<1> staticDescriptor;
  Descriptor &whole{staticDescriptor.descriptor()};
  SubscriptValue extent[]{4};
  whole.Establish(TypeCode{CFI_type_char}, sizeof buffer[0], &buffer, 1, extent,
      CFI_attribute_pointer);
  whole.Dump();
  whole.Check();

  auto cookie{IONAME(BeginInternalArrayListInput)(whole)};
  std::int64_t n[9]{-1, -2, -3, -4, 5, -6, 7, -8, 9};
  std::int64_t want[9]{1, 2, 3, 3, 5, 6, 7, 8, 9};
  for (j = 0; j < 9; ++j) {
    IONAME(InputInteger)(cookie, n[j]);
  }

  char asc[2][54]{};
  IONAME(InputAscii)(cookie, asc[0], sizeof asc[0] - 1);
  IONAME(InputAscii)(cookie, asc[1], sizeof asc[1] - 1);

  auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_FALSE((bool)status) << "list-directed input failed, status "
    << static_cast<int>(status) << '\n';
  ASSERT_NO_CRASHES() << __FILE__ << ' ' << __LINE__;

  for (j = 0; j < 9; ++j) {
    ASSERT_EQ(n[j], want[j]) << "wanted n[" << j << "]==" << want[j]
      << ", got " << n[j] << '\n';
  }
  ASSERT_NO_CRASHES() << __FILE__ << ' ' << __LINE__;

  for (j = 0; j < 2; ++j) {
    ASSERT_EQ(std::strcmp(asc[j],
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "), 0)
      << "wanted asc[" << j << "]=alphabets, got '" << asc[j] << "'\n";
  }
  ASSERT_NO_CRASHES() << __FILE__ << ' ' << __LINE__;
  EndTests();
}
