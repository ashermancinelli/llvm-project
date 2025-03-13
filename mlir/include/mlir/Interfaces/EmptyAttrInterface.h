//===- EmptyAttrInterface.h - Empty Attribute Interface ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the empty attribute interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_EMPTYATTRINTERFACE_H
#define MLIR_INTERFACES_EMPTYATTRINTERFACE_H

#include "mlir/IR/Attributes.h"

namespace mlir {

/// Include the generated interface declarations.
#include "mlir/Interfaces/EmptyAttrInterface.h.inc"

} // namespace mlir

#endif // MLIR_INTERFACES_EMPTYATTRINTERFACE_H 
