from __future__ import annotations

import math
import os
import time
import logging
from pathlib import Path
from typing import Iterable, List, Dict
from datetime import datetime, timedelta

import requests
import pandas as pd
import yfinance as yf
from tqdm import tqdm

import requests_cache
requests_cache.install_cache("yf_cache", expire_after=60*60*6)


JPX_XLS_PATH = Path("data_j.xls")
DOWNLOAD_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
OUT_CSV_PATH = Path("out/tse_under_2000.csv")
PRICE_THRESHOLD = 2000.0

BATCH_SIZE = 10
SLEEP_SEC_BETWEEN_BATCH = 4.0


def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def download_jpx_if_outdated(days: int = 7) -> None:
    """7日以上古ければ再ダウンロード"""
    should_download = False

    if not JPX_XLS_PATH.exists():
        should_download = True
    else:
        modified_time = datetime.fromtimestamp(JPX_XLS_PATH.stat().st_mtime)
        if datetime.now() - modified_time > timedelta(days=days):
            should_download = True

    if should_download:
        print("JPXファイルをダウンロード中...")
        JPX_XLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(DOWNLOAD_URL, timeout=30)
        response.raise_for_status()
        JPX_XLS_PATH.write_bytes(response.content)
        print("ダウンロード完了")
    else:
        print("JPXファイルは最新です")


def load_tse_codes_from_jpx_excel(xls_path: Path) -> pd.DataFrame:
    """JPXの上場銘柄一覧を読み込み、4桁コードと銘柄名を返す"""
    df = pd.read_excel(xls_path, engine='xlrd')  # .xls形式はxlrd

    cols = {str(c).strip(): c for c in df.columns}

    code_col_candidates = ["コード", "Code", "Local Code", "Securities Code", "Security Code"]
    name_col_candidates = ["銘柄名", "Company Name", "Issue Name", "Name", "Issue"]

    code_col = next((cols[c] for c in code_col_candidates if c in cols), None)
    if code_col is None:
        raise ValueError(f"コード列が見つかりません。columns={list(df.columns)}")

    name_col = next((cols[c] for c in name_col_candidates if c in cols), None)

    out = pd.DataFrame()
    out["code4"] = df[code_col].astype(str).str.extract(r"(\d{4})", expand=False)
    out["name"] = df[name_col].astype(str) if name_col is not None else ""
    out = out.dropna(subset=["code4"]).drop_duplicates(subset=["code4"]).reset_index(drop=True)
    return out


def fetch_latest_prices_yf(code4_list: List[str]) -> Dict[str, float]:
    """4桁コード -> 直近終値(円) を返す"""
    prices: Dict[str, float] = {}
    failed: List[str] = []

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
                    if t in data.columns.get_level_values(0):
                        close_series = data[t]["Close"].dropna()
                    else:
                        close_series = data["Close"][t].dropna()
                else:
                    close_series = data["Close"].dropna()

                if close_series.empty:
                    failed.append(c)
                    continue

                prices[c] = float(close_series.iloc[-1])
            except Exception:
                failed.append(c)

        time.sleep(SLEEP_SEC_BETWEEN_BATCH)

    if failed:
        failed_unique = sorted(set(failed))
        print(f"\n[WARN] price取得できなかった銘柄: {len(failed_unique)}")
        print("例:", ", ".join(failed_unique[:20]), "..." if len(failed_unique) > 20 else "")

    return prices


def main():
    download_jpx_if_outdated(days=7)  # 古ければ自動更新

    base = load_tse_codes_from_jpx_excel(JPX_XLS_PATH)
    codes = base["code4"].tolist()

    prices = fetch_latest_prices_yf(codes)

    base["last_close_yen"] = base["code4"].map(prices)
    filtered = base.dropna(subset=["last_close_yen"]).copy()
    filtered = filtered[filtered["last_close_yen"] <= PRICE_THRESHOLD].copy()
    filtered = filtered.sort_values("last_close_yen", ascending=True).reset_index(drop=True)

    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"\n抽出完了: {len(filtered)} 銘柄")
    print(f"CSV出力: {OUT_CSV_PATH.resolve()}")
    print(filtered.head(10))


if __name__ == "__main__":
    main()