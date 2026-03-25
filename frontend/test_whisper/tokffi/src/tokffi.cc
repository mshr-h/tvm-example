#include "tokffi/tokffi.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace tokffi
{

  namespace fs = std::filesystem;
  namespace refl = tvm::ffi::reflection;

  void TokenizerObj::RegisterReflection()
  {
    refl::ObjectDef<TokenizerObj>();
  }

  std::vector<int32_t> TokenizerObj::Encode(const std::string &text) const
  {
    return tokenizer->Encode(text);
  }

  std::string TokenizerObj::Decode(const std::vector<int32_t> &ids) const
  {
    return tokenizer->Decode(ids);
  }

  tvm::ffi::String LoadFileToString(const std::string &path)
  {
    std::ifstream fs(path, std::ios::binary);
    TVM_FFI_ICHECK(fs.good()) << "Unable to open file: " << path;
    std::string blob((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    return tvm::ffi::String(blob);
  }

  Tokenizer Tokenizer::FromHFJSONBytes(const tvm::ffi::String &blob)
  {
    auto obj = tvm::ffi::make_object<TokenizerObj>();
    obj->tokenizer = tokenizers::Tokenizer::FromBlobJSON(static_cast<std::string>(blob));
    return Tokenizer(obj);
  }

  Tokenizer Tokenizer::FromSentencePieceBytes(const tvm::ffi::String &blob)
  {
    auto obj = tvm::ffi::make_object<TokenizerObj>();
    obj->tokenizer = tokenizers::Tokenizer::FromBlobSentencePiece(static_cast<std::string>(blob));
    return Tokenizer(obj);
  }

  Tokenizer Tokenizer::FromPath(const tvm::ffi::String &path_str)
  {
    fs::path path{static_cast<std::string>(path_str)};
    TVM_FFI_ICHECK(fs::exists(path)) << "Tokenizer path does not exist: " << path;

    fs::path hf_json;
    fs::path sp_model;

    if (fs::is_directory(path))
    {
      hf_json = path / "tokenizer.json";
      sp_model = path / "tokenizer.model";
    }
    else
    {
      if (path.filename() == "tokenizer.json" || path.extension() == ".json")
      {
        hf_json = path;
      }
      else if (path.filename() == "tokenizer.model" || path.extension() == ".model")
      {
        sp_model = path;
      }
      else
      {
        hf_json = path.parent_path() / "tokenizer.json";
        sp_model = path.parent_path() / "tokenizer.model";
      }
    }

    if (fs::exists(hf_json))
    {
      return FromHFJSONBytes(LoadFileToString(hf_json.string()));
    }
    if (fs::exists(sp_model))
    {
      return FromSentencePieceBytes(LoadFileToString(sp_model.string()));
    }

    TVM_FFI_THROW(ValueError)
        << "Cannot find supported tokenizer files under path: " << path
        << ". Expected tokenizer.json or tokenizer.model.";
  }

  TextStreamerObj::TextStreamerObj(Tokenizer tokenizer) : tokenizer_(std::move(tokenizer)) {}

  TextStreamer::TextStreamer(Tokenizer tokenizer)
  {
    data_ = tvm::ffi::make_object<TextStreamerObj>(std::move(tokenizer));
  }

  void TextStreamerObj::RegisterReflection()
  {
    refl::ObjectDef<TextStreamerObj>()
        .def("finish", &TextStreamerObj::Finish);
  }

  std::string TextStreamerObj::Put(const std::vector<int32_t> &delta_tokens)
  {
    TVM_FFI_ICHECK(!finished_) << "`put` is not expected to be invoked after finish.";
    if (delta_tokens.empty())
    {
      return "";
    }

    std::string ret;
    for (int32_t delta_token : delta_tokens)
    {
      pending_tokens_.push_back(delta_token);

      std::vector<int32_t> all_tokens;
      all_tokens.reserve(prefix_tokens_.size() + pending_tokens_.size());
      all_tokens.insert(all_tokens.end(), prefix_tokens_.begin(), prefix_tokens_.end());
      all_tokens.insert(all_tokens.end(), pending_tokens_.begin(), pending_tokens_.end());

      std::string prefix_str = prefix_tokens_.empty()
                                   ? std::string()
                                   : tokenizer_->tokenizer->Decode(prefix_tokens_);
      std::string full_str = tokenizer_->tokenizer->Decode(all_tokens);

      std::string validated_str;
      std::vector<int32_t> new_pending_tokens;

      if (full_str.compare(0, prefix_str.length(), prefix_str) == 0)
      {
        validated_str = full_str.substr(prefix_str.length());
        while (!pending_tokens_.empty() &&
               validated_str.length() >= 3 &&
               validated_str.compare(validated_str.length() - 3, 3, kReplacementCharacter) == 0)
        {
          new_pending_tokens.push_back(pending_tokens_.back());
          pending_tokens_.pop_back();
          all_tokens.pop_back();
          validated_str = tokenizer_->tokenizer->Decode(all_tokens).substr(prefix_str.length());
        }
      }
      else
      {
        if (static_cast<int>(pending_tokens_.size()) < 3)
        {
          continue;
        }
        bool get_valid_full_str = false;
        while (!pending_tokens_.empty() && static_cast<int>(new_pending_tokens.size()) < 3)
        {
          new_pending_tokens.push_back(pending_tokens_.back());
          pending_tokens_.pop_back();
          all_tokens.pop_back();
          full_str = tokenizer_->tokenizer->Decode(all_tokens);
          if (full_str.compare(0, prefix_str.length(), prefix_str) == 0)
          {
            get_valid_full_str = true;
            break;
          }
        }
        if (get_valid_full_str)
        {
          validated_str = full_str.substr(prefix_str.length());
        }
        else
        {
          validated_str = tokenizer_->tokenizer->Decode(pending_tokens_);
        }
      }

      if (!pending_tokens_.empty())
      {
        prefix_tokens_ = pending_tokens_;
      }
      std::reverse(new_pending_tokens.begin(), new_pending_tokens.end());
      pending_tokens_ = std::move(new_pending_tokens);
      ret += validated_str;
    }
    return ret;
  }

  std::string TextStreamerObj::Finish()
  {
    std::vector<int32_t> all_tokens;
    all_tokens.reserve(prefix_tokens_.size() + pending_tokens_.size());
    all_tokens.insert(all_tokens.end(), prefix_tokens_.begin(), prefix_tokens_.end());
    all_tokens.insert(all_tokens.end(), pending_tokens_.begin(), pending_tokens_.end());

    std::string prefix_str = prefix_tokens_.empty()
                                 ? std::string()
                                 : tokenizer_->tokenizer->Decode(prefix_tokens_);
    std::string full_str = all_tokens.empty() ? std::string() : tokenizer_->tokenizer->Decode(all_tokens);
    finished_ = true;
    if (full_str.compare(0, prefix_str.length(), prefix_str) == 0)
    {
      return full_str.substr(prefix_str.length());
    }
    return tokenizer_->tokenizer->Decode(pending_tokens_);
  }

  TVM_FFI_STATIC_INIT_BLOCK()
  {
    TokenizerObj::RegisterReflection();
    TextStreamerObj::RegisterReflection();

    refl::GlobalDef()
        .def("tokffi.TokenizerFromHFJSONBytes",
             [](const tvm::ffi::String &blob)
             { return Tokenizer::FromHFJSONBytes(blob); })
        .def("tokffi.TokenizerFromSentencePieceBytes",
             [](const tvm::ffi::String &blob)
             { return Tokenizer::FromSentencePieceBytes(blob); })
        .def("tokffi.TokenizerFromPath",
             [](const tvm::ffi::String &path)
             { return Tokenizer::FromPath(path); })
        .def("tokffi.TokenizerEncode",
             [](const Tokenizer &tok, const tvm::ffi::String &text) -> tvm::ffi::Shape
             {
               auto ids = tok->Encode(static_cast<std::string>(text));
               std::vector<int64_t> ids64(ids.begin(), ids.end());
               return tvm::ffi::Shape(ids64.begin(), ids64.end());
             })
        .def("tokffi.TokenizerDecode",
             [](const Tokenizer &tok, const tvm::ffi::Shape &ids) -> tvm::ffi::String
             {
               std::vector<int32_t> ids32;
               ids32.reserve(ids->size);
               for (size_t i = 0; i < ids->size; ++i)
               {
                 ids32.push_back(static_cast<int32_t>(ids->data[i]));
               }
               return tvm::ffi::String(tok->Decode(ids32));
             })
        .def("tokffi.TextStreamer",
             [](Tokenizer tokenizer) -> TextStreamer
             { return TextStreamer(std::move(tokenizer)); })
        .def("tokffi.TextStreamerPut",
             [](TextStreamer streamer, const tvm::ffi::Shape &delta_tokens) -> tvm::ffi::String
             {
               std::vector<int32_t> tokens;
               tokens.reserve(delta_tokens->size);
               for (size_t i = 0; i < delta_tokens->size; ++i)
               {
                 tokens.push_back(static_cast<int32_t>(delta_tokens->data[i]));
               }
               return tvm::ffi::String(streamer->Put(tokens));
             })
        .def("tokffi.TextStreamerPutOne",
             [](TextStreamer streamer, int64_t token_id) -> tvm::ffi::String
             {
               return tvm::ffi::String(streamer->Put({static_cast<int32_t>(token_id)}));
             })
        .def_method("tokffi.TextStreamerFinish", &TextStreamerObj::Finish);
  }

} // namespace tokffi
