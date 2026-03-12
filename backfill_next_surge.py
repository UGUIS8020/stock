import pandas as pd
import numpy as np
import os

ACCUM_CSV = "out/accumulated_surge.csv"
CACHE_DIR = "out/cache"

df = pd.read_csv(ACCUM_CSV, encoding="utf-8-sig")
print(f"総件数: {len(df)}件")
print(f"next_surge_%がNullの件数: {df['next_surge_%'].isna().sum()}件")

updated = 0
for idx, row in df.iterrows():
    if not pd.isna(row.get("next_surge_%")):
        continue

    code4     = str(row["code"])
    sig_date  = row["signal_date"]
    cache_path = f"{CACHE_DIR}/{code4}.csv"

    if not os.path.exists(cache_path):
        continue

    hist = pd.read_csv(cache_path)
    hist["Date"] = pd.to_datetime(hist["Date"])

    future = hist[hist["Date"] > pd.Timestamp(sig_date)]
    if future.empty:
        continue

    n = future.iloc[0]
    if n["Open"] > 0:
        next_surge = round((n["Close"] - n["Open"]) / n["Open"] * 100, 2)
        df.at[idx, "next_surge_%"] = next_surge
        df.at[idx, "next_date"]    = n["Date"].strftime("%Y-%m-%d")
        updated += 1

print(f"✅ {updated}件のnext_surge_%を補完しました")

# 統計表示
SURGE_THRESHOLD = 10.0
for signal, label in [("day1", "🌱1日目"), ("day2", "🚀2日目")]:
    d = df[(df[signal] == True) & (df["next_surge_%"].notna())]
    if len(d) == 0:
        continue
    hit = (d["next_surge_%"] >= SURGE_THRESHOLD).sum()
    r   = d["next_surge_%"]
    print(f"\n【{label}シグナル】発火回数: {len(d)}回")
    print(f"  急騰成功率  : {hit}/{len(d)}回 ({hit/len(d)*100:.1f}%)")
    print(f"  翌日平均    : {r.mean():+.2f}%")
    print(f"  翌日中央値  : {r.median():+.2f}%")
    print(f"  最大上昇    : {r.max():+.2f}%")
    print(f"  最大下落    : {r.min():+.2f}%")
    print()

    # 分布
    bins = [
        ("大きく上昇 +15%以上",  r >= 15),
        ("急騰    +10〜15%",     (r >= 10) & (r < 15)),
        ("小幅上昇  +5〜10%",    (r >= 5)  & (r < 10)),
        ("微上昇   0〜 +5%",     (r >= 0)  & (r < 5)),
        ("小幅下落 -5〜  0%",    (r >= -5) & (r < 0)),
        ("大きく下落 -5%以下",   r < -5),
    ]
    for name, mask in bins:
        count = mask.sum()
        pct   = count / len(d) * 100
        bar   = "█" * int(pct / 3)
        avg   = r[mask].mean() if count > 0 else 0
        print(f"  {name:<22} {count:>4}件 {pct:>5.1f}%  平均{avg:>+6.1f}%  {bar}")

df.to_csv(ACCUM_CSV, index=False, encoding="utf-8-sig")
print("\n✅ CSVを保存しました")