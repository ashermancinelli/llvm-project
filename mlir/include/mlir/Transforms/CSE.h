//===- CSE.h - Common Subexpression Elimination -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares methods for eliminating common subexpressions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CSE_H_
#define MLIR_TRANSFORMS_CSE_H_

#include "mlir/Pass/Pass.h"
namespace mlir {

class DominanceInfo;
class Operation;
class RewriterBase;

/// Eliminate common subexpressions within the given operation. This transform
/// looks for and deduplicates equivalent operations.
///
/// `changed` indicates whether the IR was modified or not.
void eliminateCommonSubExpressions(RewriterBase &rewriter,
                                   DominanceInfo &domInfo, Operation *op,
                                   bool *changed = nullptr);

using ModRefCheckFn =
    llvm::function_ref<bool(mlir::Value read, mlir::Operation *maybeWrite)>;
std::unique_ptr<Pass> createCSEPass(ModRefCheckFn modRefCheckFn);

} // namespace mlir

#endif // MLIR_TRANSFORMS_CSE_H_
