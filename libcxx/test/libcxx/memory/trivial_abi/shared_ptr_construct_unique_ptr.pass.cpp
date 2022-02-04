//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <memory>

template <typename T>
struct move_only_deleter {
  move_only_deleter(move_only_deleter&&) {}
  move_only_deleter() = default;
  move_only_deleter(move_only_deleter&) = delete;
  move_only_deleter(move_only_deleter const&) = delete;
  void operator()(T* v) { delete v; }
};

// https://cplusplus.github.io/LWG/issue3548 clarifies that constructing a
// shared pointer from a unique pointer only requires the deleter of the
// unique pointer to be move constructible, therefore a shared pointer must
// be constructible from a unique pointer with a move-only deleter.
int main() {
  std::unique_ptr<int, move_only_deleter<int>> up(new int(5), move_only_deleter<int>());
  std::shared_ptr sp(std::move(up));
  return 0;
}
