# plot_scores.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("outputs/scores.csv")   # 你的 scores.csv 路徑
OUT_PNG  = Path("outputs/anomaly_score_summary.png")

df = pd.read_csv(CSV_PATH)

# 讓圖更好讀：用檔名去掉副檔名（可選）
df["label"] = df["file"].str.replace(".wav", "", regex=False)

# 依「median」排序，讓最異常的排在前面（更直覺）
df = df.sort_values("median", ascending=True).reset_index(drop=True)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

x = range(len(df))
mins = df["min"].to_list()
maxs = df["max"].to_list()
meds = df["median"].to_list()

# 畫 min->max 的垂直線（range）
ax.vlines(x, mins, maxs, linewidth=4)

# median 畫點
ax.scatter(list(x), meds, s=60, marker="o", label="median")

# 輔助線：y=0（方便一般人看正負）
ax.axhline(0, linewidth=1)

# 標籤
ax.set_xticks(list(x))
ax.set_xticklabels(df["label"].to_list(), rotation=20, ha="right")

ax.set_title("Audio Anomaly Score Summary (lower = more anomalous)")
ax.set_ylabel("Anomaly score")
ax.grid(True, axis="y", linewidth=0.5)

# 在每根柱旁邊標出 min（可選：更直觀，但字會多）
for i, (mn, md, mx) in enumerate(zip(mins, meds, maxs)):
    ax.text(i, mn, f"min {mn:.2f}", va="top", ha="center", fontsize=9)
    ax.text(i, md, f"med {md:.2f}", va="bottom", ha="center", fontsize=9)

ax.legend(loc="best")
fig.tight_layout()

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=200)
print(f"[ok] saved -> {OUT_PNG}")
plt.show()
