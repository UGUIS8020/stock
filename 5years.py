import jquantsapi
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

# バルクファイル一覧を確認
bulk_list = cli.get_bulk_list("/equities/bars/daily")
print(bulk_list)