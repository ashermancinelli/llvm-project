//===- LoopMetadataInterface.h - Loop metadata interface ---------*- C++ -*-===//
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

#ifndef MLIR_INTERFACES_LOOPMETADATAINTERFACE_H_
#define MLIR_INTERFACES_LOOPMETADATAINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

/// Interface for operations that can carry loop metadata.
///
/// This interface provides methods to get and set loop metadata attributes
/// on operations that might need to preserve such information during conversion
/// between dialects.
class LoopMetadataInterface;

namespace detail {
class LoopMetadataInterfaceInterfaceTraits {
public:
  struct Concept {
    virtual ~Concept() = default;
    virtual llvm::SmallVector<NamedAttribute> getLoopMetadata(Operation *) = 0;
    virtual void setLoopMetadata(Operation *, llvm::ArrayRef<NamedAttribute>) = 0;
    virtual void transferLoopMetadataFrom(Operation *, Operation *) = 0;
  };
  
  template <typename ConcreteOp>
  class Model : public Concept {
  public:
    llvm::SmallVector<NamedAttribute> getLoopMetadata(Operation *op) final {
      return cast<ConcreteOp>(op).getLoopMetadata();
    }
    
    void setLoopMetadata(Operation *op, 
                         llvm::ArrayRef<NamedAttribute> metadata) final {
      return cast<ConcreteOp>(op).setLoopMetadata(metadata);
    }
    
    void transferLoopMetadataFrom(Operation *op, Operation *fromOp) final {
      return cast<ConcreteOp>(op).transferLoopMetadataFrom(fromOp);
    }
  };
};
} // namespace detail

/// Implements a specific case of loop metadata propagation - extract and
/// propagate attributes from/to LLVM dialect. This can be used by operations
/// that implement LoopMetadataInterface to extract or set LLVM-specific
/// loop optimization attributes.
class LLVMLoopMetadataOps {
public:
  /// Extracts LLVM dialect attributes from an operation
  static llvm::SmallVector<NamedAttribute> extractLLVMLoopMetadata(Operation *op);
  
  /// Checks if an attribute is an LLVM dialect attribute
  static bool isLLVMDialectAttribute(const NamedAttribute &attr);
};

#include "mlir/Interfaces/LoopMetadataInterface.h.inc"

} // namespace mlir

#endif // MLIR_INTERFACES_LOOPMETADATAINTERFACE_H_ 
