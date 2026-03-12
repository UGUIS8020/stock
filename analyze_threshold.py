import pandas as pd
import numpy as np

ACCUM_CSV = "out/accumulated_surge.csv"
df = pd.read_csv(ACCUM_CSV, encoding="utf-8-sig")

print("=== 📊 急騰定義別のスコア有効性分析 ===\n")

# TOP20に入るための最低上昇率の分布を確認
print("【TOP20最低上昇率の分布（過去3ヶ月）】")
# signal_dateごとの最小上昇率（TOP20の20位の上昇率）
# next_surge_%ではなくaccumulated dataから確認
# 実際にはTOP20の上昇率は別途必要だが、スコアとの関係を閾値別に分析

# 閾値別のスコア6以上の割合
print(f"{'急騰定義':<12} {'該当銘柄':<10} {'スコア6以上':<12} {'スコア4以上':<12} {'スコア2以上'}")
print("-" * 65)

# next_surge_%がないのでscoreの分布だけで分析
# 実際の急騰率はsignal_dateのTOP20上昇率
# ここではスコア閾値を変えた場合の捕捉率を推定

total = len(df)
for score_threshold in [8, 7, 6, 5, 4, 3, 2, 1, 0]:
    count = (df["score"] >= score_threshold).sum()
    pct   = count / total * 100
    print(f"  スコア{score_threshold:>2}以上  {count:>5}件  {pct:>6.1f}%  {'█' * int(pct/3)}")

print()

# 急騰上昇率別（TOP20に入るボーダー）の分析
# TOP20の20位の上昇率を各日で確認
# accumulated_surge.csvにはsignal_dateと銘柄の上昇率がないので
# next_surge_%の分布から逆算

if "next_surge_%" in df.columns:
    r = df["next_surge_%"].dropna()
    print("【翌日上昇率の分布（全シグナル銘柄）】")
    
    thresholds = [5, 6, 7, 8, 9, 10, 12, 15]
    print(f"{'上昇率閾値':<12} {'達成件数':<10} {'達成率':<10} スコア6以上での達成率")
    print("-" * 60)
    
    high_score_df = df[df["score"] >= 6]
    r_high = high_score_df["next_surge_%"].dropna()
    
    for t in thresholds:
        all_hit   = (r >= t).sum()
        all_pct   = all_hit / len(r) * 100
        high_hit  = (r_high >= t).sum() if len(r_high) > 0 else 0
        high_pct  = high_hit / len(r_high) * 100 if len(r_high) > 0 else 0
        print(f"  +{t}%以上    {all_hit:>5}件  {all_pct:>6.1f}%    スコア6以上: {high_hit}件 ({high_pct:.1f}%)")
else:
    print("next_surge_%データなし → backfill_next_surge.pyを先に実行してください")

