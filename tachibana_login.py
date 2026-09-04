"""
tachibana_login.py - 立花証券API v4r10 ログイン

認証フロー（v4r9 と同一、2026-06-27以降）:
  - 電話認証: 不要（v4r8廃止と同時に廃止）
  - CLMAuthLoginRequest with sAuthId（設定画面で取得した認証ID）
  - レスポンスのURLは公開鍵暗号化済み → 秘密鍵で復号して保存
"""
import json
import os
import base64
import urllib3
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

_BASE_DIR        = Path(__file__).parent
LOGIN_FILE       = _BASE_DIR / "tachibana_login_response.json"
PRIVATE_KEY_FILE = _BASE_DIR / "tatibana_key" / "e_api_private_key.pem"
API_BASE         = "https://kabuka.e-shiten.jp/e_api_v4r10/"
AUTH_ID          = os.getenv("TACHIBANA_AUTH_ID", "")


def _make_psd():
    t = datetime.now()
    return (f"{t.year}.{t.month:02}.{t.day:02}"
            f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
            f".{t.microsecond // 1000:03}")


def _parse_first_json(text):
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text.strip())
    return obj


def _load_private_key():
    if not PRIVATE_KEY_FILE.exists():
        raise FileNotFoundError(f"秘密鍵が見つかりません: {PRIVATE_KEY_FILE}")
    with open(PRIVATE_KEY_FILE, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _decrypt_url(encrypted_b64: str, private_key) -> str:
    if not encrypted_b64:
        return ""
    try:
        data = base64.b64decode(encrypted_b64)
        return private_key.decrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        ).decode("utf-8").strip()
    except Exception:
        try:
            data = base64.b64decode(encrypted_b64)
            return private_key.decrypt(data, padding.PKCS1v15()).decode("utf-8").strip()
        except Exception:
            return encrypted_b64


def do_login() -> bool:
    params = (
        '{'
        '"p_no":"1",'
        f'"p_sd_date":"{_make_psd()}",'
        '"sCLMID":"CLMAuthLoginRequest",'
        f'"sAuthId":"{AUTH_ID}",'
        '"sJsonOfmt":"5"'
        '}'
    )
    url = API_BASE + "auth/?" + params
    http = urllib3.PoolManager()
    resp = http.request("GET", url, timeout=urllib3.Timeout(connect=10, read=15))
    text = resp.data.decode("shift-jis", errors="ignore")

    try:
        result = _parse_first_json(text)
    except Exception as e:
        print(f"❌ JSONパースエラー: {e}")
        print(f"   レスポンス: {text[:200]}")
        return False

    rc = result.get("sResultCode", "")
    p_errno = result.get("p_errno", "")
    if rc != "0":
        err = result.get("sResultText", "") or result.get("p_err", "不明")
        print(f"❌ ログイン失敗（sResultCode:{rc} p_errno:{p_errno} 内容:{err}）")
        return False

    # 秘密鍵でURLを復号
    private_key = _load_private_key()

    login_data = {
        "sResultCode":        rc,
        "sUrlRequest":        _decrypt_url(result.get("sUrlRequest", ""), private_key),
        "sUrlPrice":          _decrypt_url(result.get("sUrlPrice", ""), private_key),
        "sUrlMaster":         _decrypt_url(result.get("sUrlMaster", ""), private_key),
        "sUrlEvent":          _decrypt_url(result.get("sUrlEvent", ""), private_key),
        "sUrlEventWebSocket": _decrypt_url(result.get("sUrlEventWebSocket", ""), private_key),
        "sZyoutoekiKazeiC":   result.get("sZyoutoekiKazeiC", ""),
        "sLastLoginDate":     result.get("sLastLoginDate", ""),
    }
    with open(LOGIN_FILE, "w", encoding="utf-8") as f:
        json.dump(login_data, f, ensure_ascii=False, indent=2)

    print(f"✅ ログイン成功")
    print(f"   sLastLoginDate: {login_data['sLastLoginDate']}")
    print(f"   sUrlRequest: {login_data['sUrlRequest'][:70]}...")
    return True


if __name__ == "__main__":
    print("=== 立花証券e支店API v4r10 ログイン ===")
    print()

    if not AUTH_ID:
        print("❌ .env に TACHIBANA_AUTH_ID が設定されていません")
        sys.exit(1)

    print("  ログイン中（電話認証不要）...")
    if do_login():
        print()
        print("セッション更新完了。scan_morning.py を起動します...")
        print()
        import subprocess
        subprocess.run([sys.executable, str(_BASE_DIR / "scan_morning.py")])
    else:
        sys.exit(1)
