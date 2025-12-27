import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import stft
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

# ---------- config ----------
SR_TARGET = 16000
WIN_SEC = 1.0           # analysis window length
HOP_SEC = 0.5           # hop size
NFFT = 1024
N_BANDS = 32            # log-frequency bands

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    # simple linear resampling (good enough for MVP)
    t_in = np.linspace(0, 1, num=len(x), endpoint=False)
    n_out = int(len(x) * sr_out / sr_in)
    t_out = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)

def load_audio(path: str, sr_target: int = SR_TARGET) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=False)
    x = to_mono(np.asarray(x, dtype=np.float32))
    x = resample_linear(x, sr, sr_target)
    # normalize
    mx = np.max(np.abs(x)) + 1e-9
    x = x / mx
    return x, sr_target

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10**(m / 2595.0) - 1.0)

def make_mel_band_edges(sr: int, nfft: int, n_bands: int):
    f_max = sr / 2.0
    m_min, m_max = hz_to_mel(0.0), hz_to_mel(f_max)
    m = np.linspace(m_min, m_max, n_bands + 1)
    f = mel_to_hz(m)
    bins = np.floor((nfft // 2 + 1) * f / f_max).astype(int)
    bins = np.clip(bins, 0, nfft // 2)
    # ensure strictly increasing
    for i in range(1, len(bins)):
        if bins[i] <= bins[i-1]:
            bins[i] = min(bins[i-1] + 1, nfft // 2)
    return bins

def features_for_signal(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    win = int(WIN_SEC * sr)
    hop = int(HOP_SEC * sr)
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n_frames = 1 + (len(x) - win) // hop

    mel_edges = make_mel_band_edges(sr, NFFT, N_BANDS)

    feats = []
    times = []
    for i in range(n_frames):
        s = i * hop
        seg = x[s:s+win]
        f, t, Z = stft(seg, fs=sr, nperseg=NFFT, noverlap=NFFT//2, nfft=NFFT, padded=False, boundary=None)
        mag = np.abs(Z) + 1e-9  # (freq, time)
        # average over time inside this window
        spec = mag.mean(axis=1)  # (freq_bins,)

        # mel-band energies
        band_e = []
        for b in range(N_BANDS):
            a, c = mel_edges[b], mel_edges[b+1]
            if c <= a:
                c = a + 1
            band_e.append(spec[a:c].mean())
        band_e = np.array(band_e, dtype=np.float32)

        # log-energy + deltas (simple dynamics)
        loge = np.log(band_e + 1e-6)
        d1 = np.diff(loge, prepend=loge[0])
        feat = np.concatenate([loge, d1], axis=0)  # 2*N_BANDS
        feats.append(feat)

        times.append((s + win/2) / sr)

    return np.vstack(feats), np.array(times)

def fit_baseline(normal_paths: list[str]):
    X_all = []
    for p in normal_paths:
        x, sr = load_audio(p)
        X, _ = features_for_signal(x, sr)
        X_all.append(X)
    X_all = np.vstack(X_all)

    # robust-ish covariance (LedoitWolf shrinkage)
    model = LedoitWolf().fit(X_all)
    mu = model.location_
    cov = model.covariance_
    # precompute precision
    prec = np.linalg.inv(cov)
    return mu, prec

def mahalanobis_scores(X: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    D = X - mu
    # score = sqrt(diag(D * Prec * D^T))
    scores = np.sqrt(np.einsum("ij,jk,ik->i", D, prec, D))
    return scores

def main():
    normal_dir = "data/raw/normal"
    test_dir = "data/raw/test"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.wav"))) + sorted(glob.glob(os.path.join(normal_dir, "*.flac"))) + sorted(glob.glob(os.path.join(normal_dir, "*.mp3")))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.wav"))) + sorted(glob.glob(os.path.join(test_dir, "*.flac"))) + sorted(glob.glob(os.path.join(test_dir, "*.mp3")))

    if len(normal_paths) == 0:
        print("No normal audio found in data/raw/normal. Put some .wav/.flac/.mp3 there.")
        return
    if len(test_paths) == 0:
        print("No test audio found in data/raw/test. Put some .wav/.flac/.mp3 there.")
        return

    print(f"[1] Fitting baseline from {len(normal_paths)} normal files...")
    mu, prec = fit_baseline(normal_paths)
    print("[OK] Baseline ready.")

    for tp in test_paths:
        print(f"[2] Scoring test file: {tp}")
        x, sr = load_audio(tp)
        X, times = features_for_signal(x, sr)
        scores = mahalanobis_scores(X, mu, prec)

        # simple threshold: mean + 3*std (MVP)
        thr = scores.mean() + 3.0 * scores.std()
        flags = scores > thr

        base = os.path.splitext(os.path.basename(tp))[0]
        np.save(os.path.join(out_dir, f"{base}_times.npy"), times)
        np.save(os.path.join(out_dir, f"{base}_scores.npy"), scores)

        plt.figure()
        plt.plot(times, scores)
        plt.axhline(thr, linestyle="--")
        plt.title(f"Anomaly score: {base}")
        plt.xlabel("time (s)")
        plt.ylabel("score")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_score.png"), dpi=160)
        plt.close()

        # print top anomalies
        idx = np.where(flags)[0]
        if len(idx) == 0:
            print("  -> No anomalies detected (under MVP threshold).")
        else:
            top = idx[np.argsort(scores[idx])[-10:]]
            top = top[np.argsort(times[top])]
            print("  -> Anomaly windows (approx):")
            for i in top:
                print(f"     t={times[i]:.2f}s score={scores[i]:.3f}")

    print("\nDone. Check outputs/*.png")

if __name__ == "__main__":
    main()
