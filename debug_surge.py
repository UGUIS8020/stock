import jquantsapi
from dotenv import load_dotenv
import os
import time

load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

# 1銘柄だけ確認
code5 = "99800"
df = cli.get_eq_bars_daily(
    code=code5,
    from_yyyymmdd="20250226",
    to_yyyymmdd="20260226",
)

print("列名:", df.columns.tolist())
print("行数:", len(df))
print("最新5日分:")
print(df.tail(5).to_string())
print("\n最古5日分:")
print(df.head(5).to_string())