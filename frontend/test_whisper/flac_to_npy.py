import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def download_file(url: str, filepath: Path, overwrite=True):
    import requests

    if not overwrite and filepath.exists():
        return

    with filepath.open(mode="wb") as file:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


src = Path("jfk.flac")
dst = Path("audio_16khz_mono.npy")
target_sr = 16000

download_file("https://github.com/openai/whisper/raw/refs/heads/main/tests/jfk.flac", src, False)
audio, sr = sf.read(src, dtype="float32", always_2d=True)

audio = audio.mean(axis=1)

if sr != target_sr:
    g = math.gcd(sr, target_sr)
    audio = resample_poly(audio, target_sr // g, sr // g)

audio = np.asarray(audio, dtype=np.float32)

np.save(dst, audio)

print(f"saved: {dst}")
print(f"shape={audio.shape}, dtype={audio.dtype}, sr={target_sr}")
