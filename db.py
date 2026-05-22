# -*- coding: utf-8 -*-
"""
db.py - SQLite データベースヘルパー

テーブル構成:
  daily_prices   - 日次株価（out/cache/*.csv の代替）
  intraday_prices- 分足データ（out/intraday/*.csv の代替）
  scan_results   - 戦略A候補（out/scan_results.csv の代替）
  watchlist      - 戦略B候補（out/watchlist.csv の代替）
"""

import os
import sqlite3
import pandas as pd

DB_PATH = "out/stock.db"


# ══════════════════════════════════════════════════════
# 接続・初期化
# ══════════════════════════════════════════════════════

def get_conn():
    os.makedirs("out", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # 読み書き同時アクセスを安全に
    conn.execute("PRAGMA synchronous=NORMAL") # 速度と安全のバランス
    return conn


def init_db():
    """テーブルが存在しない場合のみ作成する（冪等）。"""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            code       TEXT NOT NULL,
            Date       TEXT NOT NULL,
            Open       REAL, High  REAL, Low   REAL, Close REAL,
            UL         REAL, LL    REAL,
            Volume     REAL, Va    REAL,
            AdjFactor  REAL, AdjO  REAL, AdjH  REAL,
            AdjL       REAL, AdjC  REAL, AdjVo REAL,
            prev_close REAL,
            PRIMARY KEY (code, Date)
        );
        CREATE INDEX IF NOT EXISTS idx_dp_code ON daily_prices(code);

        CREATE TABLE IF NOT EXISTS intraday_prices (
            code     TEXT NOT NULL,
            date     TEXT NOT NULL,
            datetime TEXT NOT NULL,
            time     TEXT,
            open     REAL, high REAL, low REAL, close REAL,
            volume   INTEGER,
            PRIMARY KEY (code, datetime)
        );
        CREATE INDEX IF NOT EXISTS idx_ip_code_date ON intraday_prices(code, date);

        CREATE TABLE IF NOT EXISTS scan_results (
            scan_date        TEXT NOT NULL,
            code             TEXT NOT NULL,
            name             TEXT,
            score            REAL, trend REAL, accel REAL, ratio REAL,
            actual_top20     REAL,
            market_condition TEXT,
            today_rise       REAL,
            PRIMARY KEY (scan_date, code)
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            buy_date         TEXT NOT NULL,
            code             TEXT NOT NULL,
            name             TEXT,
            today_rise       REAL, buy_price  REAL,
            score            REAL, ratio      REAL,
            market_condition TEXT,
            next_rise        REAL, next_open  REAL, next_close REAL,
            rebound_score    REAL, rebound_reason  TEXT,
            tdnet_sentiment  TEXT, tdnet_title     REAL,
            PRIMARY KEY (buy_date, code)
        );
    """)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════
# daily_prices（日次株価）
# ══════════════════════════════════════════════════════

def get_stock_history(code, days=None):
    """
    銘柄コードの日次株価履歴を DataFrame で返す。
    既存の pd.read_csv(cache_path) と同じカラム構成を維持。
    """
    conn = get_conn()
    if days:
        df = pd.read_sql(
            "SELECT * FROM daily_prices WHERE code=? ORDER BY Date DESC LIMIT ?",
            conn, params=[str(code), days]
        )
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        df = pd.read_sql(
            "SELECT * FROM daily_prices WHERE code=? ORDER BY Date",
            conn, params=[str(code)]
        )
    conn.close()
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_all_codes():
    """daily_prices に登録済みの全銘柄コードリストを返す。"""
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT code FROM daily_prices").fetchall()
    conn.close()
    return [r[0] for r in rows]


def upsert_daily_price(code, row):
    """
    1日分の価格データを追記する（既存日付はスキップ）。
    row: dict または pandas Series（Open/High/Low/Close/Volume を含む）
    """
    conn = get_conn()
    # 直前の終値を prev_close として使用
    last = conn.execute(
        "SELECT Close FROM daily_prices WHERE code=? ORDER BY Date DESC LIMIT 1",
        [str(code)]
    ).fetchone()
    prev_close = last[0] if last else None

    conn.execute("""
        INSERT OR IGNORE INTO daily_prices
            (code, Date, Open, High, Low, Close, Volume, prev_close)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        str(code),
        str(row["Date"])[:10],  # YYYY-MM-DD
        row.get("Open"),  row.get("High"),
        row.get("Low"),   row.get("Close"),
        row.get("Volume"), prev_close,
    ])
    conn.commit()
    conn.close()


DAILY_PRICE_COLS = {
    "code", "Date", "Open", "High", "Low", "Close",
    "UL", "LL", "Volume", "Va",
    "AdjFactor", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo",
    "prev_close",
}

def bulk_insert_daily_prices(code, df):
    """
    DataFrame 丸ごとを daily_prices に一括挿入（移行用）。
    既存の (code, Date) は INSERT OR IGNORE でスキップ。
    """
    if df.empty:
        return
    df = df.copy()
    df["code"] = str(code)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    # スキーマにない列（Code, stop_high 等）を除外
    known_cols = [c for c in DAILY_PRICE_COLS if c in df.columns]
    df = df[known_cols]
    # NaN → None に変換（sqlite3 parameterized query 用）
    df = df.where(pd.notna(df), other=None)
    placeholders = ", ".join("?" * len(known_cols))
    sql = f"INSERT OR IGNORE INTO daily_prices ({', '.join(known_cols)}) VALUES ({placeholders})"
    rows = list(df.itertuples(index=False, name=None))
    conn = get_conn()
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()


def get_all_latest_prices(n=2):
    """全銘柄の直近n日分の (code, Date, Close, Volume) を返す。騰落比計算に使用。"""
    conn = get_conn()
    df = pd.read_sql("""
        WITH ranked AS (
            SELECT code, Date, Close, Volume,
                   ROW_NUMBER() OVER (PARTITION BY code ORDER BY Date DESC) AS rn
            FROM daily_prices
        )
        SELECT code, Date, Close, Volume FROM ranked WHERE rn <= ?
    """, conn, params=[n])
    conn.close()
    return df


def code_exists(code):
    """daily_prices に指定コードのデータが存在するか確認。"""
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM daily_prices WHERE code=? LIMIT 1", [str(code)]
    ).fetchone()
    conn.close()
    return row is not None


# ══════════════════════════════════════════════════════
# intraday_prices（分足データ）
# ══════════════════════════════════════════════════════

def intraday_exists(code, date):
    """指定銘柄・日付の分足データが存在するか確認。"""
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM intraday_prices WHERE code=? AND date=? LIMIT 1",
        [str(code), str(date)]
    ).fetchone()
    conn.close()
    return row is not None


INTRADAY_COLS = ["code", "date", "datetime", "time", "open", "high", "low", "close", "volume"]

def save_intraday_df(code, date, df):
    """分足 DataFrame を intraday_prices に保存（既存はスキップ）。"""
    if df.empty:
        return
    df = df.copy()
    df["code"] = str(code)
    df["date"] = str(date)
    df["datetime"] = df["datetime"].astype(str)
    known_cols = [c for c in INTRADAY_COLS if c in df.columns]
    df = df[known_cols].where(pd.notna(df[known_cols]), other=None)
    placeholders = ", ".join("?" * len(known_cols))
    sql = f"INSERT OR IGNORE INTO intraday_prices ({', '.join(known_cols)}) VALUES ({placeholders})"
    rows = list(df.itertuples(index=False, name=None))
    conn = get_conn()
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()


def get_intraday(code, date):
    """指定銘柄・日付の分足データを DataFrame で返す。"""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM intraday_prices WHERE code=? AND date=? ORDER BY datetime",
        conn, params=[str(code), str(date)]
    )
    conn.close()
    return df


# ══════════════════════════════════════════════════════
# scan_results（戦略A候補）
# ══════════════════════════════════════════════════════

def get_scan_results(date=None):
    """
    scan_results を DataFrame で返す。
    date 指定なし → 全件、指定あり → その日付のみ。
    既存の pd.read_csv(SCAN_CSV) の代替。
    """
    conn = get_conn()
    if date:
        df = pd.read_sql(
            "SELECT * FROM scan_results WHERE scan_date=?",
            conn, params=[str(date)]
        )
    else:
        df = pd.read_sql("SELECT * FROM scan_results", conn)
    conn.close()
    return df


def save_scan_results(df):
    """
    scan_results テーブルを上書き保存。
    既存の df.to_csv(SCAN_CSV) の代替。
    """
    if df.empty:
        return
    conn = get_conn()
    # 同日のデータを削除してから挿入（累積重複防止）
    dates = df["scan_date"].unique().tolist()
    for d in dates:
        conn.execute("DELETE FROM scan_results WHERE scan_date=?", [d])
    df.to_sql("scan_results", conn, if_exists="append", index=False,
              method="multi", chunksize=500)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════
# watchlist（戦略B候補）
# ══════════════════════════════════════════════════════

def get_watchlist():
    """
    watchlist を DataFrame で返す。
    既存の pd.read_csv(WATCHLIST_CSV) の代替。
    """
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM watchlist", conn)
    conn.close()
    return df


def save_watchlist(df):
    """
    watchlist テーブルを全件置換保存。
    既存の df.to_csv(WATCHLIST_CSV) の代替。
    """
    if df.empty:
        return
    conn = get_conn()
    conn.execute("DELETE FROM watchlist")
    df.to_sql("watchlist", conn, if_exists="append", index=False,
              method="multi", chunksize=500)
    conn.commit()
    conn.close()
