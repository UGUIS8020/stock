# -*- coding: utf-8 -*-
"""
fetch_kisoku_list.py - 増担保規制銘柄コードの取得

取得先: JPX 信用取引に関する規制等（増担保規制実施中の銘柄一覧）
出力: 規制銘柄の4〜5桁コードセット
"""
import sys, re
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# 増担保規制銘柄一覧ページ（実施中）
MARGIN_URL = "https://www.jpx.co.jp/markets/equities/margin-reg/"


def fetch_margin_restricted_codes():
    """増担保規制実施中の4〜5桁銘柄コードを返す。取得失敗時は空set。"""
    try:
        resp = requests.get(MARGIN_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        codes = set()
        tables = soup.find_all("table")
        if not tables:
            return codes

        # 1つ目のテーブル = 実施中の規制銘柄
        table = tables[0]
        for row in table.find_all("tr")[1:]:  # ヘッダー行をスキップ
            cells = row.find_all("td")
            if len(cells) >= 2:
                code = cells[1].get_text(strip=True)
                # 4〜5桁の英数字（例: 6081, 485A）
                if re.fullmatch(r"[0-9A-Z]{4,5}", code):
                    codes.add(code[:4])  # 先頭4桁を使用（通常の銘柄コード）

        return codes

    except Exception as e:
        print(f"  [規制リスト取得エラー] {e}")
        return set()


def main():
    print("=== 増担保規制銘柄 取得テスト ===")
    print(f"URL: {MARGIN_URL}\n")

    codes = fetch_margin_restricted_codes()
    print(f"現在の増担保規制銘柄: {len(codes)}件")
    for c in sorted(codes):
        print(f"  {c}")

    if "6081" in codes:
        print("\n→ 6081（アライドアーキテクツ）は規制中 ✅")
    else:
        print("\n→ 6081 は規制リストに含まれていません（解除済み or ページ変更）")


if __name__ == "__main__":
    main()
