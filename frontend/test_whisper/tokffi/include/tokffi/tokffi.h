#ifndef TOKFFI_TOKFFI_H_
#define TOKFFI_TOKFFI_H_

#include <memory>
#include <string>
#include <vector>

#include <tokenizers_cpp.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>

namespace tokffi
{

  class TokenizerObj : public tvm::ffi::Object
  {
  public:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;

    static void RegisterReflection();

    std::vector<int32_t> Encode(const std::string &text) const;
    std::string Decode(const std::vector<int32_t> &ids) const;

    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tokffi.Tokenizer", TokenizerObj, tvm::ffi::Object);
  };

  class Tokenizer : public tvm::ffi::ObjectRef
  {
  public:
    TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tokenizer, tvm::ffi::ObjectRef, TokenizerObj);

    static Tokenizer FromHFJSONBytes(const tvm::ffi::String &blob);
    static Tokenizer FromSentencePieceBytes(const tvm::ffi::String &blob);
    static Tokenizer FromPath(const tvm::ffi::String &path);
  };

  class TextStreamerObj : public tvm::ffi::Object
  {
  public:
    explicit TextStreamerObj(Tokenizer tokenizer);

    std::string Put(const std::vector<int32_t> &delta_tokens);
    std::string Finish();

    static void RegisterReflection();

    static constexpr const char *kReplacementCharacter = "\xef\xbf\xbd";
    static constexpr bool _type_has_method_sequal_reduce = false;
    static constexpr bool _type_has_method_shash_reduce = false;
    static constexpr bool _type_mutable = true;
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tokffi.TextStreamer", TextStreamerObj, tvm::ffi::Object);

  private:
    Tokenizer tokenizer_;
    std::vector<int32_t> prefix_tokens_;
    std::vector<int32_t> pending_tokens_;
    bool finished_ = false;
  };

  class TextStreamer : public tvm::ffi::ObjectRef
  {
  public:
    explicit TextStreamer(Tokenizer tokenizer);
    TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TextStreamer, tvm::ffi::ObjectRef, TextStreamerObj);
  };

  tvm::ffi::String LoadFileToString(const std::string &path);

} // namespace tokffi

#endif // TOKFFI_TOKFFI_H_
