#include "tokffi/tokffi.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tokffi {

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

Tokenizer Tokenizer::FromHFJSONBytes(const std::string& blob) {
  auto n = ffi::make_object<TokenizerObj>();
  n->impl = tokenizers::Tokenizer::FromBlobJSON(blob);

  Tokenizer ret;
  ret.data_ = std::move(n);
  return ret;
}

Tokenizer Tokenizer::FromSentencePieceBytes(const std::string& blob) {
  auto n = ffi::make_object<TokenizerObj>();
  n->impl = tokenizers::Tokenizer::FromBlobSentencePiece(blob);

  Tokenizer ret;
  ret.data_ = std::move(n);
  return ret;
}

Tokenizer Tokenizer::FromPath(const std::string& path) {
  namespace fs = std::filesystem;
  fs::path p(path);

  if (fs::is_directory(p)) {
    fs::path tok_json = p / "tokenizer.json";
    if (fs::exists(tok_json)) {
      return FromHFJSONBytes(LoadBytesFromFile(tok_json.string()));
    }

    fs::path tok_model = p / "tokenizer.model";
    if (fs::exists(tok_model)) {
      return FromSentencePieceBytes(LoadBytesFromFile(tok_model.string()));
    }

    throw std::runtime_error(
        "Tokenizer directory does not contain tokenizer.json or tokenizer.model: " + path);
  }

  if (p.extension() == ".json") {
    return FromHFJSONBytes(LoadBytesFromFile(path));
  }
  if (p.extension() == ".model") {
    return FromSentencePieceBytes(LoadBytesFromFile(path));
  }

  throw std::runtime_error(
      "Unsupported tokenizer path. Expected directory, *.json, or *.model: " + path);
}

}  // namespace tokffi

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  using tokffi::Tokenizer;
  using tvm::ffi::Array;

  refl::GlobalDef()
      .def("tokffi.TokenizerFromHFJSONBytes",
           [](const std::string& blob) {
             return Tokenizer::FromHFJSONBytes(blob);
           })
      .def("tokffi.TokenizerFromSentencePieceBytes",
           [](const std::string& blob) {
             return Tokenizer::FromSentencePieceBytes(blob);
           })
      .def("tokffi.TokenizerFromPath",
           [](const std::string& path) {
             return Tokenizer::FromPath(path);
           })
      .def("tokffi.TokenizerEncode",
           [](const Tokenizer& tok, const std::string& text) {
             std::vector<int32_t> ids32 = tok->impl->Encode(text);
             Array<int64_t> ids64;
             ids64.reserve(ids32.size());
             for (int32_t x : ids32) {
               ids64.push_back(static_cast<int64_t>(x));
             }
             return ids64;
           })
      .def("tokffi.TokenizerDecode",
           [](const Tokenizer& tok, const Array<int64_t>& ids64) {
             std::vector<int32_t> ids32;
             ids32.reserve(ids64.size());
             for (int64_t x : ids64) {
               ids32.push_back(static_cast<int32_t>(x));
             }
             return tok->impl->Decode(ids32);
           });
}