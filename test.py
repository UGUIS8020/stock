import jquantsapi
from dotenv import load_dotenv
import os

load_dotenv()

cli = jquantsapi.ClientV2(
    api_key=os.getenv("JQUANTS_API_KEY")
)

# 銘柄一覧
listed = cli.get_eq_master()
print(listed.head())