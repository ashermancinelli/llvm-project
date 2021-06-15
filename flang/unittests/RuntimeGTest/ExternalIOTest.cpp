//===-- flang/unittests/RuntimeGTest/ExternalIOTest.cpp ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../runtime/io-api.h"
#include "../../runtime/main.h"
#include "../../runtime/stop.h"
#include "llvm/Support/raw_ostream.h"
#include "CrashHandlerFixture.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace Fortran::runtime::io;

struct ExternalIOTests : public CrashHandlerFixture {};

TEST(ExternalIOTests, TestDirectUnformatted) {
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH')
  Cookie io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "DIRECT", 6)) << "SetAccess(DIRECT)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";

  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk) << "EndIoStatement() for OpenNewUnit";

  static constexpr int records{10};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,REC=j) j
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';

    buffer = j;
    ASSERT_TRUE(IONAME(OutputUnformattedBlock)
        (io, reinterpret_cast<const char *>(&buffer), recl, recl))
      << "OutputUnformattedBlock()";

    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OutputUnformattedBlock";
  }

  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,REC=j) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    ASSERT_TRUE(IONAME(InputUnformattedBlock)(io, reinterpret_cast<char *>(&buffer), recl, recl))
        << "InputUnformattedBlock()";

    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for InputUnformattedBlock";

    ASSERT_EQ(buffer, j) << "Read back " << buffer << " from direct unformatted record "
      << j << ", expected " << j << '\n';
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk) << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestDirectUnformattedSwapped) {
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH',CONVERT='NATIVE')
  auto io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "DIRECT", 6)) << "SetAccess(DIRECT)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";
  ASSERT_TRUE(IONAME(SetConvert)(io, "NATIVE", 6)) << "SetConvert(NATIVE)";

  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk) << "EndIoStatement() for OpenNewUnit";

  static constexpr int records{10};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,REC=j) j
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    buffer = j;
    ASSERT_TRUE(IONAME(OutputUnformattedBlock)(io, reinterpret_cast<const char *>(&buffer), recl, recl)) << "OutputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk) << "EndIoStatement() for OutputUnformattedBlock";
  }

  // OPEN(UNIT=unit,STATUS='OLD',CONVERT='SWAP')
  io = IONAME(BeginOpenUnit)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "OLD", 3)) << "SetStatus(OLD)";
  ASSERT_TRUE(IONAME(SetConvert)(io, "SWAP", 4)) << "SetConvert(SWAP)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk) << "EndIoStatement() for OpenUnit";

  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,REC=j) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    ASSERT_TRUE(IONAME(InputUnformattedBlock)(io, reinterpret_cast<char *>(&buffer), recl, recl)) << "InputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk) << "EndIoStatement() for InputUnformattedBlock";
    ASSERT_EQ(buffer >> 56, j) << "Read back " << (buffer >> 56)
      << " from direct unformatted record " << j << ", expected " << j
      << '\n';
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk) << "EndIoStatement() for Close";
}
