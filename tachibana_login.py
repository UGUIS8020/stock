"""
tachibana_login.py - 立花証券API 電話認証 + 再ログイン

使い方:
    python tachibana_login.py

電話認証後にセッションURLを更新します。
"""
import json
import os
import urllib3
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

_BASE_DIR    = Path(__file__).parent
LOGIN_FILE   = _BASE_DIR / "tachibana_login_response.json"
API_BASE     = "https://kabuka.e-shiten.jp/e_api_v4r8/"
USER_ID      = os.getenv("TACHIBANA_LOGIN_ID", "")
PASSWORD     = os.getenv("TACHIBANA_LOGIN_PASS", "")


def _url_encode(s):
    table = {
        ' ': '%20', '!': '%21', '"': '%22', '#': '%23', '$': '%24',
        '%': '%25', '&': '%26', "'": '%27', '(': '%28', ')': '%29',
        '*': '%2A', '+': '%2B', ',': '%2C', '/': '%2F', ':': '%3A',
        ';': '%3B', '<': '%3C', '=': '%3D', '>': '%3E', '?': '%3F',
        '@': '%40', '[': '%5B', ']': '%5D', '^': '%5E', '`': '%60',
        '{': '%7B', '|': '%7C', '}': '%7D', '~': '%7E',
    }
    return ''.join(table[c] if c in table else c for c in s)


def do_login():
    t = datetime.now()
    p_sd = (f"{t.year}.{t.month:02}.{t.day:02}"
            f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
            f".{t.microsecond // 1000:03}")
    params = (
        '{'
        '"p_no":"1",'
        f'"p_sd_date":"{p_sd}",'
        '"sCLMID":"CLMAuthLoginRequest",'
        f'"sUserId":"{USER_ID}",'
        f'"sPassword":"{_url_encode(PASSWORD)}",'
        '"sJsonOfmt":"5"'
        '}'
    )
    url = API_BASE + "auth/?" + params
    http = urllib3.PoolManager()
    resp = http.request("GET", url, timeout=urllib3.Timeout(connect=10, read=15))
    text = resp.data.decode("shift-jis", errors="ignore")
    result = json.loads(text)

    rc = result.get("sResultCode", "")
    if rc != "0":
        err = result.get("sResultText", "不明")
        print(f"❌ ログイン失敗（コード: {rc}、内容: {err}）")
        if rc == "10088":
            print("→ 電話認証がまだ完了していません。もう一度発信してください。")
        return False

    login_data = {
        "sResultCode":        rc,
        "sUrlRequest":        result.get("sUrlRequest", ""),
        "sUrlPrice":          result.get("sUrlPrice", ""),
        "sUrlMaster":         result.get("sUrlMaster", ""),
        "sUrlEvent":          result.get("sUrlEvent", ""),
        "sUrlEventWebSocket": result.get("sUrlEventWebSocket", ""),
        "sZyoutoekiKazeiC":   result.get("sZyoutoekiKazeiC", ""),
        "sLastLoginDate":     result.get("sLastLoginDate", ""),
    }
    with open(LOGIN_FILE, "w", encoding="utf-8") as f:
        json.dump(login_data, f, ensure_ascii=False, indent=2)

    print(f"✅ ログイン成功")
    print(f"   sLastLoginDate: {login_data['sLastLoginDate']}")
    print(f"   sUrlRequest: {login_data['sUrlRequest'][:60]}...")
    return True


if __name__ == "__main__":
    print("=== 立花証券e支店API 再ログイン ===")
    print()

    if not USER_ID or not PASSWORD:
        print("❌ .env に TACHIBANA_LOGIN_ID / TACHIBANA_LOGIN_PASS が設定されていません")
        sys.exit(1)

    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  【電話認証】登録済み電話番号から今すぐ発信:      │")
    print("  │    📞 0120-28-6592（無料）                        │")
    print("  │    📞 050-3102-6575（有料）                       │")
    print("  │  音声が流れて自動的に切れたらOKです               │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    input("  電話が切れたら Enter キーを押してください...")
    print()
    print("  ログイン中...")

    if do_login():
        print()
        print("セッション更新完了。scan_morning.py を起動します...")
        print()
        import subprocess
        from pathlib import Path
        subprocess.run([sys.executable, str(Path(__file__).parent / "scan_morning.py")])
    else:
        sys.exit(1)
