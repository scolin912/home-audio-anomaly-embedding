import subprocess
from pathlib import Path

SRC_DIRS = [
    ("data/raw/normal", "data/wav/normal"),
    ("data/raw/test", "data/wav/test"),
]

TARGET_SR = 16000

def convert_one(mp3_path: Path, wav_path: Path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp3_path),
        "-ac", "1",               # mono
        "-ar", str(TARGET_SR),    # 16k
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    for src, dst in SRC_DIRS:
        src_dir = Path(src)
        dst_dir = Path(dst)
        if not src_dir.exists():
            continue

        for mp3 in src_dir.glob("*.mp3"):
            wav = dst_dir / (mp3.stem + ".wav")
            print(f"[convert] {mp3.name} -> {wav}")
            convert_one(mp3, wav)

if __name__ == "__main__":
    main()
