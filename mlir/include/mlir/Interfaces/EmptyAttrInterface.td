//===- EmptyAttrInterface.td - Empty Attribute Interface ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines an empty attribute interface that prescribes no methods.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EMPTYATTRINTERFACE
#define MLIR_EMPTYATTRINTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute interface
//===----------------------------------------------------------------------===//

def EmptyAttrInterface : AttrInterface<"EmptyAttrInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    An empty attribute interface that prescribes no methods.
    
    This interface can be used as a marker interface for attributes where no
    additional functionality is required through the interface.
  }];
  
  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the access groups attribute or nullptr",
      /*returnType=*/  "::mlir::ArrayAttr",
      /*methodName=*/  "getAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op;
      }]
      >,
  ];
}

#endif // MLIR_EMPTYATTRINTERFACE 
