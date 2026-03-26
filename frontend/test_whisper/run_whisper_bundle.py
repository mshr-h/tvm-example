import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import tokenizers_tvm_ffi
import tvm
from scipy.signal import resample_poly
from tvm import relax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Whisper TVM bundle with tokenizers_tvm_ffi-based decoding."
    )
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--task", choices=["transcribe", "translate"])
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--no-prev-text", action="store_true")
    return parser.parse_args()


def choose_device(name: str):
    if name == "cuda":
        dev = tvm.cuda(0)
        if not dev.exist:
            raise RuntimeError("CUDA was requested, but tvm.cuda(0).exist is False.")
        return dev
    if name == "cpu":
        return tvm.cpu(0)
    return tvm.cuda(0) if tvm.cuda(0).exist else tvm.cpu(0)


def to_tvm(x, dev):
    if isinstance(x, np.ndarray):
        return tvm.runtime.tensor(x, dev)
    if hasattr(x, "copyto") and hasattr(x, "numpy"):
        return x.copyto(target=dev)
    raise TypeError(f"Unsupported tensor input type: {type(x)}")


def unwrap(x):
    while not hasattr(x, "numpy"):
        x = x[0]
    return x


def unwrap_many(x):
    if hasattr(x, "numpy"):
        return x
    return [unwrap_many(x[i]) for i in range(len(x))]


def load_audio(path: Path, sr: int) -> np.ndarray:
    audio, src_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if src_sr != sr:
        g = math.gcd(src_sr, sr)
        audio = resample_poly(audio, sr // g, src_sr // g)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def slice_window(audio: np.ndarray, start: int, n_samples: int):
    chunk = np.asarray(audio[start : start + n_samples], dtype=np.float32)
    valid = int(chunk.shape[0])
    if valid < n_samples:
        chunk = np.pad(chunk, (0, n_samples - valid))
    return chunk[None, :], np.asarray([valid], dtype=np.int32)


def make_keep_mask(past_len: int, max_past_len: int) -> np.ndarray:
    mask = np.full((1, 1, 1, max_past_len), -1e9, dtype=np.float32)
    if past_len > 0:
        mask[..., :past_len] = 0.0
    return mask


def load_params(params_path: Path, dev, meta: dict[str, Any]):
    by_name = {
        str(k): v
        for k, v in tvm.runtime.load_param_dict_from_file(str(params_path)).items()
    }
    names = [str(x) for x in meta["param_names"]]
    if len(names) != int(meta["param_count"]):
        raise ValueError("metadata param count mismatch")
    return [to_tvm(by_name[name], dev) for name in names]


def resolve_task(task: str | None, meta: dict[str, Any]) -> str:
    task = str(task or meta.get("default_task", "transcribe")).lower()
    if task not in {"transcribe", "translate"}:
        raise ValueError(f"Unsupported task: {task}")
    if task == "translate" and not bool(meta.get("supports_translate", True)):
        raise ValueError("This bundle does not support translate.")
    return task


def resolve_language(language: str, meta: dict[str, Any]) -> str | None:
    if not bool(meta.get("is_multilingual", False)):
        return str(meta.get("default_language_code", "en"))
    language = str(language).strip().lower()
    if language in {"", "auto"}:
        return None
    aliases = {
        str(k).lower(): str(v)
        for k, v in meta.get("language_alias_to_code", {}).items()
    }
    if language in aliases:
        return aliases[language]
    if language in meta.get("language_token_ids", {}):
        return language
    raise ValueError(f"Unsupported language: {language}")


def logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("-inf")
    m = float(np.max(x))
    if np.isneginf(m):
        return float("-inf")
    return float(m + np.log(np.exp(x - m).sum()))


def ts(seconds: float) -> str:
    ms = int(round(max(0.0, float(seconds)) * 1000.0))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class WhisperBundleRunner:
    def __init__(self, artifacts_dir: Path, tokenizer_json: Path, device: str):
        meta_path = artifacts_dir / "whisper_bundle_metadata.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dev = choose_device(device)
        self.vm = relax.VirtualMachine(
            tvm.runtime.load_module(str(artifacts_dir / self.meta["lib_name"])),
            self.dev,
        )
        self.params = load_params(
            artifacts_dir / self.meta["params_name"], self.dev, self.meta
        )
        self.tok = tokenizers_tvm_ffi.Tokenizer.from_json_bytes(
            tokenizer_json.read_bytes()
        )
        self.special_ids = {int(x) for x in self.meta["special_ids"]}
        self.timestamp_begin = int(self.meta["timestamp_begin"])
        self.eos = int(self.meta["eos_token_id"])
        self.vocab_size = int(self.meta["vocab_size"])
        self.suppress = np.asarray(
            [
                int(x)
                for x in self.meta.get("suppress_tokens", [])
                if 0 <= int(x) < self.vocab_size
            ],
            dtype=np.int64,
        )
        self.begin_suppress = np.asarray(
            [
                int(x)
                for x in self.meta.get("begin_suppress_tokens", [])
                if 0 <= int(x) < self.vocab_size
            ],
            dtype=np.int64,
        )
        self.blank_suppress = np.asarray(
            [
                int(x)
                for x in self.meta.get("blank_suppress_tokens", [])
                if 0 <= int(x) < self.vocab_size
            ],
            dtype=np.int64,
        )
        self.no_timestamps = (
            int(self.meta["no_timestamps_token_id"])
            if self.meta.get("no_timestamps_token_id") is not None
            and 0 <= int(self.meta["no_timestamps_token_id"]) < self.vocab_size
            else None
        )
        self.max_past_len = int(self.meta["max_dec_len_compiled"]) - 1
        self.self_shape = (
            int(self.meta["decoder_layers"]),
            1,
            int(self.meta["decoder_attention_heads"]),
            self.max_past_len,
            int(self.meta["head_dim"]),
        )

    def decode_text(self, ids: Sequence[int]) -> str:
        ids = [
            int(x)
            for x in ids
            if int(x) < self.timestamp_begin and int(x) not in self.special_ids
        ]
        return "" if not ids else str(self.tok.decode(ids))

    def build_prompt(
        self,
        language_code: str | None,
        task: str,
        timestamps: bool,
        history: Sequence[int],
    ):
        ids = [int(self.meta["decoder_start_token_id"])]
        lang_ids = {
            str(k): int(v) for k, v in self.meta.get("language_token_ids", {}).items()
        }
        if language_code is not None and language_code in lang_ids:
            ids.append(lang_ids[language_code])
        ids.append(
            int(
                self.meta["translate_token_id"]
                if task == "translate"
                else self.meta["transcribe_token_id"]
            )
        )
        if not timestamps and self.meta.get("no_timestamps_token_id") is not None:
            ids.append(int(self.meta["no_timestamps_token_id"]))
        if history and self.meta.get("startofprev_token_id") is not None:
            limit = max(0, int(self.meta["max_target_positions"]) // 2 - len(ids) - 1)
            tail = list(history)[-limit:] if limit > 0 else []
            ids = [int(self.meta["startofprev_token_id"]), *tail, *ids]
        return ids

    def max_new_tokens(self, requested: int | None, prompt_len: int) -> int:
        available = int(self.meta["max_target_positions"]) - int(prompt_len)
        if available <= 0:
            raise ValueError("Prompt leaves no room for new tokens.")
        if requested is None:
            return available
        if requested <= 0:
            raise ValueError("--max-new-tokens must be positive.")
        return min(int(requested), available)

    def load_window(self, audio: np.ndarray, start: int):
        wave, valid = slice_window(audio, start, int(self.meta["n_samples"]))
        pp = self.vm["preprocess"](to_tvm(wave, self.dev), to_tvm(valid, self.dev))
        feats, _mask = unwrap_many(pp)
        enc = unwrap(self.vm["encode"](feats, self.params))
        cross_k, cross_v = unwrap_many(self.vm["cross_kv"](enc, self.params))
        return cross_k, cross_v, int(valid[0])

    def step(
        self,
        token_id: int,
        pos: int,
        self_k: np.ndarray,
        self_v: np.ndarray,
        cross_k,
        cross_v,
    ):
        out = self.vm["decode_step"](
            to_tvm(np.asarray([[token_id]], dtype=np.int32), self.dev),
            to_tvm(np.asarray([[pos]], dtype=np.int32), self.dev),
            to_tvm(self_k, self.dev),
            to_tvm(self_v, self.dev),
            to_tvm(make_keep_mask(pos, self.max_past_len), self.dev),
            cross_k,
            cross_v,
            self.params,
        )
        logits, new_k, new_v = unwrap_many(out)
        return (
            unwrap(logits).numpy().astype(np.float32),
            unwrap(new_k).numpy().astype(np.float32),
            unwrap(new_v).numpy().astype(np.float32),
        )

    def prime(self, prompt_ids: Sequence[int], cross_k, cross_v):
        self_k = np.zeros(self.self_shape, dtype=np.float32)
        self_v = np.zeros_like(self_k)
        logits = None
        for pos, token_id in enumerate(prompt_ids):
            logits, new_k, new_v = self.step(
                int(token_id), pos, self_k, self_v, cross_k, cross_v
            )
            self_k[:, :, :, pos : pos + 1, :] = new_k
            self_v[:, :, :, pos : pos + 1, :] = new_v
        if logits is None:
            raise RuntimeError("Failed to prime decoder.")
        return logits, self_k, self_v, len(prompt_ids)

    def detect_language(self, cross_k, cross_v) -> str:
        if not bool(self.meta.get("is_multilingual", False)):
            return str(self.meta.get("default_language_code", "en"))
        lang_ids = {
            str(k): int(v) for k, v in self.meta.get("language_token_ids", {}).items()
        }
        if not lang_ids:
            return str(self.meta.get("default_language_code", "en"))
        logits, *_ = self.prime(
            [int(self.meta["decoder_start_token_id"])], cross_k, cross_v
        )
        cand = np.asarray(sorted(lang_ids.values()), dtype=np.int64)
        return {v: k for k, v in lang_ids.items()}.get(
            int(cand[int(np.argmax(logits[0][cand]))]),
            str(self.meta.get("default_language_code", "en")),
        )

    def filter_logits(
        self, logits: np.ndarray, generated: Sequence[int], timestamps: bool
    ) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32).copy()
        if self.suppress.size:
            logits[self.suppress] = -np.inf
        if not generated:
            if self.begin_suppress.size:
                logits[self.begin_suppress] = -np.inf
            if self.blank_suppress.size:
                logits[self.blank_suppress] = -np.inf
        if not timestamps:
            return logits
        if self.no_timestamps is not None:
            logits[self.no_timestamps] = -np.inf
        seq = [int(x) for x in generated]
        last_is_ts = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
        prev_is_ts = len(seq) < 2 or seq[-2] >= self.timestamp_begin
        if last_is_ts:
            if prev_is_ts:
                logits[self.timestamp_begin :] = -np.inf
            else:
                logits[: self.eos] = -np.inf
        ts_tokens = [x for x in seq if x >= self.timestamp_begin]
        if ts_tokens:
            floor = (
                ts_tokens[-1] if last_is_ts and not prev_is_ts else ts_tokens[-1] + 1
            )
            logits[self.timestamp_begin : floor] = -np.inf
        if not seq:
            logits[: self.timestamp_begin] = -np.inf
            limit = self.timestamp_begin + int(
                round(1.0 / float(self.meta["time_precision"]))
            )
            if limit + 1 < logits.shape[0]:
                logits[limit + 1 :] = -np.inf
        ln = logsumexp(logits)
        if np.isfinite(ln):
            lp = logits.astype(np.float64) - ln
            ts_lp = logsumexp(lp[self.timestamp_begin :])
            txt_lp = (
                float(np.max(lp[: self.timestamp_begin]))
                if self.timestamp_begin > 0
                else float("-inf")
            )
            if ts_lp > txt_lp:
                logits[: self.timestamp_begin] = -np.inf
        return logits

    def decode_window(
        self,
        cross_k,
        cross_v,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        timestamps: bool,
    ):
        logits, self_k, self_v, pos = self.prime(prompt_ids, cross_k, cross_v)
        out: list[int] = []
        for _ in range(max_new_tokens):
            scores = self.filter_logits(logits[0], out, timestamps)
            if np.all(np.isneginf(scores)):
                break
            next_id = int(np.argmax(scores))
            if not 0 <= next_id < self.vocab_size:
                raise RuntimeError(f"Out-of-range token id: {next_id}")
            out.append(next_id)
            if next_id == self.eos:
                break
            logits, new_k, new_v = self.step(
                next_id, pos, self_k, self_v, cross_k, cross_v
            )
            self_k[:, :, :, pos : pos + 1, :] = new_k
            self_v[:, :, :, pos : pos + 1, :] = new_v
            pos += 1
        while out and out[-1] == self.eos:
            out.pop()
        return out

    def build_segments(
        self, ids: Sequence[int], offset_s: float, valid_samples: int, timestamps: bool
    ):
        ids = [int(x) for x in ids]
        if not ids:
            return [], valid_samples
        if not timestamps:
            text = self.decode_text(ids)
            seg = {
                "start": float(offset_s),
                "end": float(
                    offset_s + valid_samples / float(self.meta["sample_rate"])
                ),
                "text": text,
                "tokens": ids,
            }
            return ([seg] if text.strip() else []), valid_samples

        arr = np.asarray(ids, dtype=np.int64)
        ts_mask = arr >= self.timestamp_begin
        pair_ends = np.where(ts_mask[:-1] & ts_mask[1:])[0] + 1
        tail_is_single_ts = arr.size >= 2 and ts_mask[-2:].tolist() == [False, True]
        segments: list[dict[str, Any]] = []
        advance = valid_samples

        if len(pair_ends) > 0:
            cuts = pair_ends.tolist() + ([len(arr)] if tail_is_single_ts else [])
            last = 0
            for cur in cuts:
                part = arr[last:cur]
                last = cur
                if len(part) <= 1:
                    continue
                a = float(
                    offset_s
                    + (int(part[0]) - self.timestamp_begin)
                    * float(self.meta["time_precision"])
                )
                b = float(
                    offset_s
                    + (int(part[-1]) - self.timestamp_begin)
                    * float(self.meta["time_precision"])
                )
                text = self.decode_text(part.tolist())
                if b > a and text.strip():
                    segments.append(
                        {
                            "start": a,
                            "end": b,
                            "text": text,
                            "tokens": [int(x) for x in part],
                        }
                    )
            if not tail_is_single_ts and last > 0:
                advance = int(
                    (int(arr[last - 1]) - self.timestamp_begin)
                    * int(self.meta["samples_per_timestamp"])
                )
        else:
            end_s = float(offset_s + valid_samples / float(self.meta["sample_rate"]))
            ts_only = arr[ts_mask]
            if len(ts_only) > 0 and int(ts_only[-1]) != self.timestamp_begin:
                end_s = float(
                    offset_s
                    + (int(ts_only[-1]) - self.timestamp_begin)
                    * float(self.meta["time_precision"])
                )
            text = self.decode_text(ids)
            if end_s > offset_s and text.strip():
                segments.append(
                    {
                        "start": float(offset_s),
                        "end": end_s,
                        "text": text,
                        "tokens": ids,
                    }
                )

        if advance <= 0:
            advance = min(
                valid_samples, max(1, int(self.meta["samples_per_timestamp"]))
            )
        return segments, int(advance)

    def run(
        self,
        audio: np.ndarray,
        language: str | None,
        task: str,
        timestamps: bool,
        max_new_tokens: int | None,
        condition_on_previous_text: bool,
    ):
        sr = int(self.meta["sample_rate"])
        seek = 0
        history: list[int] = []
        segments: list[dict[str, Any]] = []
        detected = language
        while seek < len(audio) or (len(audio) == 0 and not segments):
            cross_k, cross_v, valid = self.load_window(audio, seek)
            if valid <= 0 and len(audio) > 0:
                break
            if detected is None:
                detected = self.detect_language(cross_k, cross_v)
            prompt = self.build_prompt(
                detected,
                task,
                timestamps,
                history if condition_on_previous_text else [],
            )
            ids = self.decode_window(
                cross_k,
                cross_v,
                prompt,
                self.max_new_tokens(max_new_tokens, len(prompt)),
                timestamps,
            )
            window_segments, advance = self.build_segments(
                ids, seek / float(sr), valid, timestamps
            )
            for s in window_segments:
                s["id"] = len(segments)
                segments.append(s)
                if condition_on_previous_text:
                    history.extend(int(x) for x in s["tokens"])
            if advance <= 0:
                advance = min(valid, int(self.meta["samples_per_timestamp"]))
            seek += int(advance if timestamps else valid)
            if len(audio) == 0:
                break
        detected = detected or str(self.meta.get("default_language_code", "en"))
        return {
            "text": "".join(str(s["text"]) for s in segments).strip(),
            "language": detected,
            "language_name": self.meta.get("language_code_to_name", {}).get(
                detected, detected
            ),
            "task": task,
            "timestamps": bool(timestamps),
            "segments": segments,
        }


def main():
    args = parse_args()
    tokenizer_json = args.tokenizer_json or (args.artifacts_dir / "tokenizer.json")
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_json}")
    runner = WhisperBundleRunner(args.artifacts_dir, tokenizer_json, args.device)
    audio = load_audio(args.audio, int(runner.meta["sample_rate"]))
    result = runner.run(
        audio=audio,
        language=resolve_language(args.language, runner.meta),
        task=resolve_task(args.task, runner.meta),
        timestamps=args.timestamps,
        max_new_tokens=args.max_new_tokens,
        condition_on_previous_text=not args.no_prev_text,
    )
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.timestamps:
        for s in result["segments"]:
            print(f"[{ts(s['start'])} --> {ts(s['end'])}] {str(s['text']).strip()}")
    else:
        print(result["text"])


if __name__ == "__main__":
    main()
