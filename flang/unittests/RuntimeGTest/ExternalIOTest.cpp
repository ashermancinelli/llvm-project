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
  // IONAME(SetRecl)(io, recl) || (ASSERT_TRUE(false) << "SetRecl()");
  // IONAME(SetStatus)(io, "SCRATCH", 7) || (ASSERT_TRUE(false) << "SetStatus(SCRATCH)");
  // int unit{-1};
  // IONAME(GetNewUnit)(io, unit) || (ASSERT_TRUE(false) << "GetNewUnit()");
  // IONAME(EndIoStatement)
  // (io) == IostatOk || (ASSERT_TRUE(false) << "EndIoStatement() for OpenNewUnit");
  // static constexpr int records{10};
  // for (int j{1}; j <= records; ++j) {
  //   // WRITE(UNIT=unit,REC=j) j
  //   io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
  //   IONAME(SetRec)(io, j) || (ASSERT_TRUE(false) << "SetRec(" << j << ')');
  //   buffer = j;
  //   IONAME(OutputUnformattedBlock)
  //   (io, reinterpret_cast<const char *>(&buffer), recl, recl) ||
  //       (ASSERT_TRUE(false) << "OutputUnformattedBlock()");
  //   IONAME(EndIoStatement)
  //   (io) == IostatOk ||
  //       (ASSERT_TRUE(false) << "EndIoStatement() for OutputUnformattedBlock");
  // }
  // for (int j{records}; j >= 1; --j) {
  //   // READ(UNIT=unit,REC=j) n
  //   io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
  //   IONAME(SetRec)
  //   (io, j) || (ASSERT_TRUE(false) << "SetRec(" << j << ')');
  //   IONAME(InputUnformattedBlock)
  //   (io, reinterpret_cast<char *>(&buffer), recl, recl) ||
  //       (ASSERT_TRUE(false) << "InputUnformattedBlock()");
  //   IONAME(EndIoStatement)
  //   (io) == IostatOk ||
  //       (ASSERT_TRUE(false) << "EndIoStatement() for InputUnformattedBlock");
  //   if (buffer != j) {
  //     ASSERT_TRUE(false) << "Read back " << buffer << " from direct unformatted record "
  //            << j << ", expected " << j << '\n';
  //   }
  // }
  // // CLOSE(UNIT=unit,STATUS='DELETE')
  // io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  // IONAME(SetStatus)(io, "DELETE", 6) || (ASSERT_TRUE(false) << "SetStatus(DELETE)");
  // IONAME(EndIoStatement)
  // (io) == IostatOk || (ASSERT_TRUE(false) << "EndIoStatement() for Close");
}
