# -*- coding: utf-8 -*-
"""
全銘柄の市場コードを JQuants から取得して DB に保存する（一回限りの初期化用）。
scan_daily.py の毎日実行でも同様に更新される。
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

import jquantsapi
import db

db.init_db()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

print("JQuants から市場コード取得中...")
try:
    df = cli.get_list()
    if df.empty:
        print("データなし")
    else:
        df["code4"] = df["Code"].astype(str).str[:4]
        code_mkt = dict(zip(df["code4"], df["Mkt"].astype(str)))
        db.upsert_market_codes(code_mkt)
        print(f"更新完了: {len(code_mkt)}銘柄")

        # サンプル確認
        for code in ["3843", "7203", "4478", "9434"]:
            mkt = db.get_market_code_db(code)
            row = df[df["code4"] == code]
            mkt_name = row["MktNm"].iloc[0] if not row.empty else "?"
            print(f"  {code}: JQuants={row['Mkt'].iloc[0] if not row.empty else '?'} "
                  f"({mkt_name}) → Tachibana={mkt}")
except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()
