//===- LoopMetadataInterface.td - Loop Metadata Interface ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for operations that carry loop metadata.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LOOPMETADATAINTERFACE
#define MLIR_INTERFACES_LOOPMETADATAINTERFACE

include "mlir/IR/OpBase.td"

def LoopMetadataInterface : OpInterface<"LoopMetadataInterface"> {
  let description = [{
    Interface for operations that can carry loop metadata attributes.
    
    Operations implementing this interface can pass metadata across dialect
    conversions without creating explicit dependencies between dialects.
    This includes metadata like vectorization hints, unrolling factors,
    and other loop optimization directives.
  }];
  
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns loop metadata attributes (or empty if none exist).",
      /*returnType=*/"::llvm::SmallVector<::mlir::NamedAttribute>",
      /*methodName=*/"getLoopMetadata",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        return ::llvm::SmallVector<::mlir::NamedAttribute>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Sets loop metadata attributes on this operation.",
      /*returnType=*/"void",
      /*methodName=*/"setLoopMetadata",
      /*args=*/(ins "::llvm::ArrayRef<::mlir::NamedAttribute>":$metadata),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        this->getOperation()->setDiscardableAttrs(metadata);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Transfers loop metadata from another operation to this one.",
      /*returnType=*/"void",
      /*methodName=*/"transferLoopMetadataFrom",
      /*args=*/(ins "::mlir::Operation *":$fromOp),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        if (auto interface = ::llvm::dyn_cast<LoopMetadataInterface>(fromOp))
          this->setLoopMetadata(interface.getLoopMetadata());
      }]
    >
  ];
}

#endif // MLIR_INTERFACES_LOOPMETADATAINTERFACE 
