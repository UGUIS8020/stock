import pandas as pd
import numpy as np

ACCUM_CSV = "out/accumulated_surge.csv"

df = pd.read_csv(ACCUM_CSV, encoding="utf-8-sig")
print(f"=== 📊 急騰銘柄の前日スコア分布分析 ===")
print(f"総サンプル: {len(df)}件（全て急騰した銘柄）\n")

# 各指標の分布
for col, label in [
    ("score", "総合スコア"),
    ("trend", "出来高トレンド（5日間）"),
    ("accel", "出来高加速度（3日前比）"),
    ("ratio", "出来高比率（20日平均比）"),
]:
    s = df[col].dropna()
    print(f"【{label}】")
    print(f"  平均: {s.mean():.2f}  中央値: {s.median():.2f}")
    print(f"  最小: {s.min():.2f}  最大: {s.max():.2f}")

    # 分布をバケツ分け
    if col == "score":
        bins   = [-99, 0, 2, 4, 6, 8, 99]
        labels = ["0未満", "0〜2", "2〜4", "4〜6", "6〜8", "8以上"]
    elif col == "ratio":
        bins   = [0, 1, 2, 3, 5, 10, 9999]
        labels = ["〜1倍", "1〜2倍", "2〜3倍", "3〜5倍", "5〜10倍", "10倍以上"]
    elif col == "accel":
        bins   = [-9999, 0, 50, 100, 200, 500, 9999]
        labels = ["0未満", "0〜50%", "50〜100%", "100〜200%", "200〜500%", "500%以上"]
    else:
        bins   = [-9999, 0, 10, 30, 50, 100, 9999]
        labels = ["0未満", "0〜10", "10〜30", "30〜50", "50〜100", "100以上"]

    counts = pd.cut(s, bins=bins, labels=labels).value_counts().sort_index()
    total  = len(s)
    for lbl, cnt in counts.items():
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {lbl:<12} {cnt:>4}件 {pct:>5.1f}%  {bar}")
    print()

# スコア別の急騰上昇率との相関
print("=== スコア帯別の急騰当日上昇率 ===")
surge_col = "next_surge_%" if "next_surge_%" in df.columns else "actual_surge_%"
if surge_col in df.columns:
    df["score_band"] = pd.cut(df["score"],
        bins=[-99, 0, 2, 4, 6, 8, 99],
        labels=["0未満", "0〜2", "2〜4", "4〜6", "6〜8", "8以上"])
    
    # 急騰当日の上昇率（signal_dateの上昇率はnext_surge_%ではなく元データから）
    # ここではスコア分布だけ表示
    grp = df.groupby("score_band", observed=True)["score"].count()
    print(f"\nスコア分布（急騰した銘柄の前日スコア）:")
    for band, cnt in grp.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {band:<8} {cnt:>4}件 {pct:>5.1f}%  {bar}")

print("\n=== 💡 重要な洞察 ===")
high_score = df[df["score"] >= 6]
low_score  = df[df["score"] < 0]
print(f"スコア6以上の銘柄: {len(high_score)}件 ({len(high_score)/len(df)*100:.1f}%)")
print(f"スコア0未満の銘柄: {len(low_score)}件 ({len(low_score)/len(df)*100:.1f}%)")
print(f"\n→ 急騰した銘柄の{len(low_score)/len(df)*100:.0f}%はスコアが低い（材料型）")
print(f"→ 急騰した銘柄の{len(high_score)/len(df)*100:.0f}%はスコアが高い（出来高型）")
print(f"\n出来高型({len(high_score)}件)が全銘柄スキャンの対象です")
