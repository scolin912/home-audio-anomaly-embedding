# src/detect_events.py
import os
import glob
import joblib
import numpy as np
import soundfile as sf

MODEL_PATH = "models/ocsvm.joblib"
NORMAL_DIR = "data/wav/normal"
TEST_DIR   = "data/wav/test"

# ---- event gating (NEW) ----
# 你原本想做「相對門檻」：每個檔案用自己的背景分數當 base，
# 只要分數比 base 低 DROP_MARGIN 以上才算 event。
# 同時仍保留 global threshold (由 normal percentile 算出) 作為上限。
USE_RELATIVE = True
BASE_MODE = "median"     # "median" or "p50" (同義)
DROP_MARGIN = 0.35       # 分數需要比背景低多少才算 event（可調）
ABS_FLOOR = None         # 也可加絕對底線，例如 -0.5
# ----------------------------

SR_TARGET = 16000
WIN_SEC = 1.0
HOP_SEC = 0.5
N_MFCC = 20

THR_PERCENTILE = 20.0     # 用 normal scores 的 percentile 當門檻
MIN_EVENT_SEC = 0.5       # 最短事件長度


def load_wav_mono(path, sr_target=SR_TARGET):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != sr_target:
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        x_new = np.linspace(0, 1, num=int(len(y) * sr_target / sr), endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = sr_target
    return y.astype(np.float32), sr


def extract_frame_features(y, sr, win_sec=WIN_SEC, hop_sec=HOP_SEC, n_mfcc=N_MFCC):
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)

    feats = []
    times = []
    for i in range(0, max(0, len(y) - win + 1), hop):
        seg = y[i:i + win]
        if len(seg) < win:
            continue

        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))

        spec = np.fft.rfft(seg * np.hanning(len(seg)))
        pow_spec = (np.abs(spec) ** 2).astype(np.float32)

        n_bins = 40
        if len(pow_spec) < n_bins:
            pad = np.zeros(n_bins, dtype=np.float32)
            pad[:len(pow_spec)] = pow_spec
            pow_spec = pad
        else:
            pow_spec = pow_spec[:n_bins]

        loge = np.log(pow_spec + 1e-12)
        mfcc_like = np.real(np.fft.rfft(loge, n=2 * (n_bins - 1)))[:n_mfcc]
        mfcc_like = mfcc_like.astype(np.float32)

        v = np.concatenate([mfcc_like, np.array([rms], dtype=np.float32)], axis=0)
        feats.append(v)
        times.append(i / sr)

    if not feats:
        return np.zeros((0, n_mfcc + 1), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.vstack(feats).astype(np.float32), np.array(times, dtype=np.float32)


def list_wavs(folder):
    files = []
    for e in ("*.wav", "*.WAV"):
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)


def find_events(scores, times, thr, hop_sec=HOP_SEC, min_event_sec=MIN_EVENT_SEC):
    """
    scores < thr  視為 anomaly
    回傳 events: (start_sec, end_sec, min_score)
    """
    is_anom = scores < thr
    events = []
    start = None
    min_score = None

    for i, flag in enumerate(is_anom):
        if flag and start is None:
            start = float(times[i])
            min_score = float(scores[i])
        elif flag and start is not None:
            if float(scores[i]) < min_score:
                min_score = float(scores[i])
        elif (not flag) and start is not None:
            end = float(times[i]) + hop_sec
            if end - start >= min_event_sec:
                events.append((start, end, min_score))
            start = None
            min_score = None

    if start is not None:
        end = float(times[-1]) + hop_sec
        if end - start >= min_event_sec:
            events.append((start, end, min_score))

    return events


def score_stats(scores):
    return {
        "min": float(np.min(scores)),
        "p10": float(np.percentile(scores, 10)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
        "std": float(np.std(scores)),
        "uniq": int(len(np.unique(np.round(scores, 6))))
    }


def compute_base(scores, mode=BASE_MODE):
    mode = (mode or "median").lower()
    if mode in ("p50", "50", "percentile50"):
        return float(np.percentile(scores, 50))
    # default: median
    return float(np.median(scores))


def compute_thr_file(scores, thr_global):
    """回傳單檔使用的 threshold。數值越低越嚴格（更難觸發 event）。"""
    thr_file = float(thr_global)

    if USE_RELATIVE:
        base = compute_base(scores, BASE_MODE)
        # 相對門檻：base - DROP_MARGIN
        thr_rel = base - float(DROP_MARGIN)
        # 同時保留 global threshold 作為上限（避免 base 太低導致 miss）
        thr_file = min(thr_file, thr_rel)

    if ABS_FLOOR is not None:
        thr_file = min(thr_file, float(ABS_FLOOR))

    return thr_file


def main():
    print("[load] model")
    model = joblib.load(MODEL_PATH)

    # build threshold from NORMAL frames
    norm_scores = []
    normal_files = list_wavs(NORMAL_DIR)
    if not normal_files:
        raise FileNotFoundError(f"No wav files found under {NORMAL_DIR}")

    for f in normal_files:
        y, sr = load_wav_mono(f)
        X, _ = extract_frame_features(y, sr)
        if len(X) == 0:
            continue
        s = model.decision_function(X)
        norm_scores.extend(list(s))

    if not norm_scores:
        raise RuntimeError("No normal frames extracted; cannot build threshold.")

    norm_scores = np.array(norm_scores, dtype=np.float64)
    thr_global = float(np.percentile(norm_scores, THR_PERCENTILE))
    print(f"[info] anomaly threshold (global) = {thr_global:.6f}  (normal p{THR_PERCENTILE})")
    ns = score_stats(norm_scores)
    print(f"[debug] normal score stats: min={ns['min']:.6f} p10={ns['p10']:.6f} "
          f"median={ns['median']:.6f} max={ns['max']:.6f} std={ns['std']:.6f} uniq≈{ns['uniq']}")
    print()

    # detect on TEST files
    test_files = list_wavs(TEST_DIR)
    if not test_files:
        raise FileNotFoundError(f"No wav files found under {TEST_DIR}")

    for f in test_files:
        name = os.path.basename(f)
        y, sr = load_wav_mono(f)
        X, times = extract_frame_features(y, sr)

        print(f"[debug] {name} -> X shape {X.shape}")
        if len(X) == 0:
            print(f"=== {name} ===\n  (empty)\n")
            continue

        print(f"[debug] feature mean std ≈ {float(np.std(X, axis=0).mean()):.5f}")

        scores = model.decision_function(X)

        thr_file = compute_thr_file(scores, thr_global)
        base = compute_base(scores, BASE_MODE)

        st = score_stats(scores)
        # ✅ 修正：用 thr_file，而不是誤用 thr_global
        events = find_events(scores, times, thr_file)

        print(f"=== {name} ===")
        print(f"  [thr] global={thr_global:.6f}  base({BASE_MODE})={base:.6f}  thr_file={thr_file:.6f}  "
              f"USE_RELATIVE={USE_RELATIVE}  DROP_MARGIN={DROP_MARGIN}")
        if not events:
            print("  (no anomaly detected)")
        else:
            for s, e, m in events:
                print(f"  EVENT  {s:6.2f}s ~ {e:6.2f}s   min_score={m:.6f}")

        print(f"  [score] min={st['min']:.6f} p10={st['p10']:.6f} median={st['median']:.6f} "
              f"max={st['max']:.6f} std={st['std']:.6f} uniq≈{st['uniq']}")
        print()


if __name__ == "__main__":
    main()
