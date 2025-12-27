# src/mvp_anomaly_v1.py
import os
import glob
import joblib
import numpy as np
import soundfile as sf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ========== config ==========
NORMAL_DIR = "data/wav/normal"
TEST_DIR   = "data/wav/test"
MODEL_PATH = "models/ocsvm.joblib"
SCORES_CSV = "outputs/scores.csv"

SR_TARGET = 16000
WIN_SEC = 1.0
HOP_SEC = 0.5

N_MFCC = 20

# OneClassSVM params
NU = 0.05
GAMMA = "scale"     # OK now because we StandardScale first
KERNEL = "rbf"
# ============================


def load_wav_mono(path, sr_target=SR_TARGET):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    # simple resample if needed (no librosa dependency here)
    if sr != sr_target:
        # linear resample (fast MVP) – later可以換成 librosa.resample
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        x_new = np.linspace(0, 1, num=int(len(y) * sr_target / sr), endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = sr_target
    return y.astype(np.float32), sr


def extract_frame_features(y, sr, win_sec=WIN_SEC, hop_sec=HOP_SEC, n_mfcc=N_MFCC):
    """
    回傳 X: (num_frames, 21)
    21 = 20 MFCC mean + 1 RMS
    注意：這裡不依賴 librosa，MFCC 我們用簡化版 FFT + DCT MVP（夠用來驗證流程）
    如果你想用 librosa 的 mfcc 也可以，但要確保 train/test 兩邊完全一致。
    """
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)

    # pre-emphasis optional
    # y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    feats = []
    for i in range(0, max(0, len(y) - win + 1), hop):
        seg = y[i:i + win]
        if len(seg) < win:
            continue

        # RMS
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))

        # power spectrum
        spec = np.fft.rfft(seg * np.hanning(len(seg)))
        pow_spec = (np.abs(spec) ** 2).astype(np.float32)

        # log energies (very rough mel-ish compression for MVP)
        # bucket to fixed bins then DCT -> "mfcc-like"
        n_bins = 40
        if len(pow_spec) < n_bins:
            pad = np.zeros(n_bins, dtype=np.float32)
            pad[:len(pow_spec)] = pow_spec
            pow_spec = pad
        else:
            pow_spec = pow_spec[:n_bins]

        loge = np.log(pow_spec + 1e-12)
        # DCT (type-II) simple
        mfcc_like = np.real(np.fft.rfft(loge, n=2*(n_bins-1)))[:n_mfcc]
        mfcc_like = mfcc_like.astype(np.float32)

        v = np.concatenate([mfcc_like, np.array([rms], dtype=np.float32)], axis=0)
        feats.append(v)

    if not feats:
        return np.zeros((0, n_mfcc + 1), dtype=np.float32)
    return np.vstack(feats).astype(np.float32)


def list_wavs(folder):
    exts = ("*.wav", "*.WAV")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    ensure_dir("models")
    ensure_dir("outputs")

    normal_files = list_wavs(NORMAL_DIR)
    test_files = list_wavs(TEST_DIR)

    print(f"[info] normal files: {len(normal_files)}")

    X_list = []
    for f in normal_files:
        y, sr = load_wav_mono(f)
        X = extract_frame_features(y, sr)
        if len(X) == 0:
            continue
        X_list.append(X)

    if not X_list:
        raise RuntimeError("No normal frames extracted. Check your normal wav files.")

    Xn = np.vstack(X_list)
    print(f"[info] normal frames = {len(Xn)} , feature_dim={Xn.shape[1]}")
    print(f"[debug] normal feature std mean ≈ {float(np.std(Xn, axis=0).mean()):.5f}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel=KERNEL, nu=NU, gamma=GAMMA)),
    ])
    pipe.fit(Xn)

    joblib.dump(pipe, MODEL_PATH)
    print(f"[ok] saved model -> {MODEL_PATH}")

    # score test files for quick check
    rows = [("file", "min", "p10", "median", "max")]
    print("\n=== quick score summary (lower = more anomalous) ===")
    for f in test_files:
        y, sr = load_wav_mono(f)
        X = extract_frame_features(y, sr)
        if len(X) == 0:
            continue
        s = pipe.decision_function(X)
        rows.append((os.path.basename(f),
                     float(np.min(s)),
                     float(np.percentile(s, 10)),
                     float(np.median(s)),
                     float(np.max(s))))
        print(f"{os.path.basename(f):28s} min={rows[-1][1]:.4f} p10={rows[-1][2]:.4f} median={rows[-1][3]:.4f}")

    # write csv
    with open(SCORES_CSV, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(",".join(map(str, r)) + "\n")
    print(f"[ok] wrote scores -> {SCORES_CSV}")


if __name__ == "__main__":
    main()
