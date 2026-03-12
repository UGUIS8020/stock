import pandas as pd
import numpy as np

# 蓄積データを読み込む
df = pd.read_csv("out/accumulated_surge.csv", encoding="utf-8-sig")

print(f"=== 📊 失敗30%の内訳分析 ===")
print(f"総サンプル: {len(df)}件\n")

for signal, label in [("day1", "🌱1日目シグナル"), ("day2", "🚀2日目シグナル")]:
    sig_df = df[df[signal] == True].copy()
    if sig_df.empty:
        continue

    r = sig_df["actual_surge_%"]
    total = len(sig_df)

    # 分布を定義
    bins = [
        ("大きく上昇 +15%以上",   r >= 15,          "🚀"),
        ("急騰    +10〜15%",      (r >= 10) & (r < 15), "✅"),
        ("小幅上昇 +5〜10%",      (r >= 5)  & (r < 10), "🔼"),
        ("微上昇  0〜+5%",        (r >= 0)  & (r < 5),  "➡️"),
        ("小幅下落 -5〜0%",       (r >= -5) & (r < 0),  "🔽"),
        ("大きく下落 -5%以下",    r < -5,               "💥"),
    ]

    print(f"【{label}】発火回数: {total}回")
    print(f"{'パターン':<22} {'件数':>5} {'割合':>7} {'平均上昇率':>10}")
    print("-" * 50)

    for name, mask, icon in bins:
        count = mask.sum()
        pct   = count / total * 100
        avg   = r[mask].mean() if count > 0 else 0
        bar   = "█" * int(pct / 3)
        print(f"  {icon} {name:<20} {count:>4}件 {pct:>6.1f}%  {avg:>+8.1f}%  {bar}")

    print(f"\n  翌日平均上昇率: {r.mean():+.2f}%")
    print(f"  翌日中央値    : {r.median():+.2f}%")
    print(f"  最大上昇率    : {r.max():+.2f}%")
    print(f"  最大下落率    : {r.min():+.2f}%")
    print()

print("=== 損切りライン別の影響 ===")
for signal, label in [("day2", "🚀2日目")]:
    sig_df = df[df[signal] == True].copy()
    r = sig_df["actual_surge_%"]

    print(f"\n【{label}】損切りラインを変えた場合：")
    print(f"{'損切りライン':<15} {'救える件数':>10} {'救えない件数':>12}")
    print("-" * 40)
    for cut in [-3, -5, -7, -10]:
        survived = (r >= cut).sum()
        lost     = (r < cut).sum()
        print(f"  {cut}%以下で売却    {survived:>6}件({survived/len(r)*100:.0f}%)   {lost:>6}件({lost/len(r)*100:.0f}%)")