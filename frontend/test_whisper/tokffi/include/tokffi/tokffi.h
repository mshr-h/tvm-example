#pragma once

#include <memory>
#include <string>

#include <tokenizers_cpp.h>

#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/container/array.h>

namespace tokffi {

namespace ffi = tvm::ffi;

class TokenizerObj : public ffi::Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tokffi.Tokenizer", TokenizerObj, ffi::Object);

  std::unique_ptr<tokenizers::Tokenizer> impl;
};

class Tokenizer : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tokenizer, ffi::ObjectRef, TokenizerObj);

  static Tokenizer FromHFJSONBytes(const std::string& blob);
  static Tokenizer FromSentencePieceBytes(const std::string& blob);
  static Tokenizer FromPath(const std::string& path);
};

std::string LoadBytesFromFile(const std::string& path);

}  // namespace tokffi