import pandas as pd
import os

CSV_PATH = "out/accumulated_surge.csv"

if not os.path.exists(CSV_PATH):
    print("CSVファイルが見つかりません")
    exit()

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"既存データ: {len(df)}件")
print(f"カラム: {list(df.columns)}")

# すでに移行済みなら何もしない
if "signal_date" in df.columns:
    print("すでに移行済みです")
    exit()

# カラム名を変更
df = df.rename(columns={
    "date":          "signal_date",
    "actual_surge_%": "next_surge_%",
})

# 新しいカラムを追加
df.insert(1, "next_date", None)

# バックアップ
df_backup = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df_backup.to_csv("out/accumulated_surge_backup.csv", index=False, encoding="utf-8-sig")
print("バックアップ: out/accumulated_surge_backup.csv")

# 注意: 既存の「actual_surge_%」は当日上昇率なので
# 翌日上昇率としては不正確 → Noneにリセット
df["next_surge_%"] = None
df["next_date"]    = None

df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 移行完了: {len(df)}件")
print(f"※ next_surge_%はリセットされました。fetch_historical.pyを再実行して正しい翌日データを取得してください。")