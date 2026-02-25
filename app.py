# pip install pandas openpyxl yfinance tqdm

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

import requests_cache
requests_cache.install_cache("yf_cache", expire_after=60*60*6)


JPX_XLSX_PATH = Path("data/tse_list.xlsx")      # ←ここを自分の保存先に
OUT_CSV_PATH = Path("out/tse_under_2000.csv")   # 出力先
PRICE_THRESHOLD = 2000.0

# yfinanceは一度に大量のティッカーを投げると失敗しやすいので分割
BATCH_SIZE = 10
SLEEP_SEC_BETWEEN_BATCH = 4.0  # 負荷軽減


def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_tse_codes_from_jpx_excel(xlsx_path: Path) -> pd.DataFrame:
    """
    JPXの上場銘柄一覧Excelを読み込み、4桁コードと銘柄名を抜き出す。
    日本語列名(コード/銘柄名)・英語列名(Code/Company Name 等)の両対応。
    """
    df = pd.read_excel(xlsx_path)

    # 列名を柔らかく探索（全角スペース等もあるのでstrip）
    cols = {str(c).strip(): c for c in df.columns}

    # 日本語を優先して候補に入れる
    code_col_candidates = ["コード", "Code", "Local Code", "Securities Code", "Security Code"]
    name_col_candidates = ["銘柄名", "Company Name", "Issue Name", "Name", "Issue"]

    code_col = next((cols[c] for c in code_col_candidates if c in cols), None)
    if code_col is None:
        raise ValueError(
            f"コード列が見つかりませんでした。Excelの列名を確認してください。columns={list(df.columns)}"
        )

    name_col = next((cols[c] for c in name_col_candidates if c in cols), None)

    out = pd.DataFrame()
    out["code4"] = (
        df[code_col]
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
    )

    out["name"] = df[name_col].astype(str) if name_col is not None else ""
    out = out.dropna(subset=["code4"]).drop_duplicates(subset=["code4"]).reset_index(drop=True)
    return out


def fetch_latest_prices_yf(code4_list: List[str]) -> Dict[str, float]:
    """
    4桁コード -> 直近終値(円) を返す
    取得できない銘柄はスキップし、最後にまとめて報告する
    """
    prices: Dict[str, float] = {}
    failed: List[str] = []

    # yfinanceのログ抑制（printは完全には止まらないこともあるが軽減）
    import logging
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    for batch in tqdm(list(chunked(code4_list, BATCH_SIZE)), desc="Downloading prices"):
        tickers = [f"{c}.T" for c in batch]

        try:
            data = yf.download(
                tickers=" ".join(tickers),
                period="10d",
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            # バッチが落ちたら個別フォールバック
            for c in batch:
                t = f"{c}.T"
                try:
                    d1 = yf.download(t, period="10d", interval="1d", progress=False)
                    if not d1.empty and "Close" in d1.columns:
                        close = d1["Close"].dropna()
                        if not close.empty:
                            prices[c] = float(close.iloc[-1])
                        else:
                            failed.append(c)
                    else:
                        failed.append(c)
                except Exception:
                    failed.append(c)
            time.sleep(SLEEP_SEC_BETWEEN_BATCH)
            continue

        for c in batch:
            t = f"{c}.T"
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    # 多銘柄
                    if t in data.columns.get_level_values(0):
                        close_series = data[t]["Close"].dropna()
                    else:
                        close_series = data["Close"][t].dropna()
                else:
                    # 1銘柄
                    close_series = data["Close"].dropna()

                if close_series.empty:
                    failed.append(c)
                    continue

                prices[c] = float(close_series.iloc[-1])
            except Exception:
                failed.append(c)

        time.sleep(SLEEP_SEC_BETWEEN_BATCH)

    # 失敗は最後にまとめて表示（必要ならファイル出力でもOK）
    if failed:
        failed_unique = sorted(set(failed))
        print(f"\n[WARN] price取得できなかった銘柄: {len(failed_unique)}")
        print("例:", ", ".join(failed_unique[:20]), "..." if len(failed_unique) > 20 else "")

    return prices

def download_one_close(ticker: str) -> float | None:
    try:
        d = yf.download(ticker, period="7d", interval="1d", progress=False, threads=False)
        if d is None or getattr(d, "empty", True):
            return None
        s = d.get("Close")
        if s is None:
            return None
        s = s.dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def main():
    if not JPX_XLSX_PATH.exists():
        raise FileNotFoundError(f"JPX Excelが見つかりません: {JPX_XLSX_PATH}")

    base = load_tse_codes_from_jpx_excel(JPX_XLSX_PATH)
    codes = base["code4"].tolist()

    prices = fetch_latest_prices_yf(codes)

    base["last_close_yen"] = base["code4"].map(prices)
    filtered = base.dropna(subset=["last_close_yen"]).copy()
    filtered = filtered[filtered["last_close_yen"] <= PRICE_THRESHOLD].copy()
    filtered = filtered.sort_values("last_close_yen", ascending=True).reset_index(drop=True)

    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"抽出完了: {len(filtered)} 銘柄")
    print(f"CSV出力: {OUT_CSV_PATH.resolve()}")
    print(filtered.head(10))


if __name__ == "__main__":
    main()
