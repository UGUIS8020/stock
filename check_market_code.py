# -*- coding: utf-8 -*-
"""JQuants から銘柄の市場区分を取得する"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

import jquantsapi

cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

# get_list で銘柄一覧（市場コード含む）
try:
    df = cli.get_list(code="3843")
    print("3843 columns:", list(df.columns))
    if not df.empty:
        print(df.iloc[0].to_dict())
    else:
        print("データなし")
except Exception as e:
    print(f"get_list error: {e}")

# get_market_segments で市場セグメント一覧
try:
    ms = cli.get_market_segments()
    print("\nMarket segments:", ms)
except Exception as e:
    print(f"get_market_segments error: {e}")
