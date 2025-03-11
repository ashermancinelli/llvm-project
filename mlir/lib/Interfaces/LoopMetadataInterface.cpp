//===- LoopMetadataInterface.cpp - Loop metadata interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface for operations that carry loop metadata.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LoopMetadataInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

#include "mlir/Interfaces/LoopMetadataInterface.cpp.inc"

llvm::SmallVector<NamedAttribute> 
LLVMLoopMetadataOps::extractLLVMLoopMetadata(Operation *op) {
  llvm::SmallVector<NamedAttribute> llvmAttrs;
  llvm::copy_if(op->getAttrs(), std::back_inserter(llvmAttrs),
                [](const NamedAttribute &attr) {
                  return isLLVMDialectAttribute(attr);
                });
  return llvmAttrs;
}

bool LLVMLoopMetadataOps::isLLVMDialectAttribute(const NamedAttribute &attr) {
  return attr.getValue().getDialect() && 
         isa<LLVM::LLVMDialect>(attr.getValue().getDialect());
} 
