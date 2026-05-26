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
from datetime import datetime
from pathlib import Path

_BASE_DIR = Path(__file__).parent
DB_PATH = str(_BASE_DIR / "out" / "stock.db")


# ══════════════════════════════════════════════════════
# 接続・初期化
# ══════════════════════════════════════════════════════

def get_conn():
    os.makedirs(str(_BASE_DIR / "out"), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # 読み書き同時アクセスを安全に
    conn.execute("PRAGMA synchronous=NORMAL") # 速度と安全のバランス
    return conn


def init_db():
    """テーブルが存在しない場合のみ作成する（冪等）。既存テーブルの列追加も行う。"""
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

        CREATE TABLE IF NOT EXISTS positions (
            date             TEXT NOT NULL,
            code             TEXT NOT NULL,
            name             TEXT,
            shares           INTEGER,
            buy_price        REAL,
            tp_price         REAL,
            sl_price         REAL,
            strategy         TEXT,
            buy_time         TEXT,
            status           TEXT DEFAULT 'open',
            sell_price       REAL,
            sell_time        TEXT,
            pnl_pct          REAL,
            exit_reason      TEXT DEFAULT '',
            entry_change_pct REAL,
            rb_score         REAL,
            condition        TEXT,
            PRIMARY KEY (date, code)
        );

        CREATE TABLE IF NOT EXISTS closing_log (
            date       TEXT NOT NULL,
            code       TEXT NOT NULL,
            name       TEXT,
            change_pct REAL,
            rb_score   REAL,
            price      REAL,
            tp_price   REAL,
            sl_price   REAL,
            PRIMARY KEY (date, code)
        );

        CREATE TABLE IF NOT EXISTS market_log (
            date                    TEXT PRIMARY KEY,
            condition               TEXT,
            ad_ratio                REAL,
            nikkei_change           REAL,
            up_count                INTEGER,
            down_count              INTEGER,
            strategy_a_success_rate REAL
        );

        CREATE TABLE IF NOT EXISTS morning_log (
            date               TEXT PRIMARY KEY,
            condition_forecast TEXT,
            condition_score    REAL,
            us_avg_change      REAL,
            dow_change         REAL,
            nasdaq_change      REAL,
            sp500_change       REAL,
            nikkei_change      REAL,
            usdjpy             REAL,
            strategy_a_thr     REAL,
            stop_loss_pct      REAL
        );

        CREATE TABLE IF NOT EXISTS candidates_log (
            date         TEXT NOT NULL,
            strategy     TEXT NOT NULL,
            code         TEXT NOT NULL,
            condition    TEXT,
            name         TEXT,
            score        REAL,
            ratio        REAL,
            judgment     TEXT,
            reason       TEXT,
            open_price   REAL,
            close_price  REAL,
            high_price   REAL,
            low_price    REAL,
            result_pct   REAL,
            result_grade TEXT,
            PRIMARY KEY (date, strategy, code)
        );

        CREATE TABLE IF NOT EXISTS daytime_signals (
            date               TEXT NOT NULL,
            time               TEXT NOT NULL,
            code               TEXT NOT NULL,
            name               TEXT,
            price              REAL,
            change_pct         REAL,
            prev_high          REAL,
            conditions_met     INTEGER,
            breakout           INTEGER,
            volume_surge       INTEGER,
            momentum           INTEGER,
            volume_pace_ratio  REAL,
            tp_price           REAL,
            sl_price           REAL,
            PRIMARY KEY (date, code)
        );
        CREATE INDEX IF NOT EXISTS idx_ds_date ON daytime_signals(date);
    """)
    conn.commit()
    conn.close()
    migrate_candidates_log_columns()


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


# ══════════════════════════════════════════════════════
# positions（ポジション管理）
# ══════════════════════════════════════════════════════

def save_position_db(code, name, shares, buy_price, strategy, tp_pct=0.03, sl_pct=0.05,
                     entry_change_pct=None, rb_score=None, condition=None):
    """買い約定後にポジションを positions テーブルへ記録する。"""
    tp_price = round(buy_price * (1 + tp_pct))
    sl_price = round(buy_price * (1 - sl_pct))
    today    = datetime.now().strftime("%Y-%m-%d")
    buy_time = datetime.now().strftime("%H:%M:%S")
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO positions
            (date, code, name, shares, buy_price, tp_price, sl_price,
             strategy, buy_time, status, exit_reason,
             entry_change_pct, rb_score, condition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', '', ?, ?, ?)
    """, [
        today, str(code), str(name), int(shares),
        float(buy_price), float(tp_price), float(sl_price),
        str(strategy), buy_time,
        None if entry_change_pct is None else round(float(entry_change_pct), 2),
        None if rb_score         is None else rb_score,
        None if condition        is None else str(condition),
    ])
    conn.commit()
    conn.close()
    print(f"  [DB] ポジション記録: {code} {name} {shares}株 "
          f"買値{buy_price}円 TP:{tp_price}円(+{tp_pct*100:.0f}%) "
          f"SL:{sl_price}円(-{sl_pct*100:.1f}%)")


def load_open_positions():
    """status='open' のポジションを dict リストで返す（position_monitor 互換）。"""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM positions WHERE status='open' ORDER BY date, code",
        conn
    )
    conn.close()
    if df.empty:
        return []
    df = df.where(pd.notna(df), other="")
    return df.to_dict(orient="records")


def update_position(date, code, **fields):
    """指定ポジションのフィールドを更新する（売り約定後の更新用）。"""
    if not fields:
        return
    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [str(date), str(code)]
    conn = get_conn()
    conn.execute(
        f"UPDATE positions SET {set_clause} WHERE date=? AND code=?",
        values
    )
    conn.commit()
    conn.close()


def get_today_ordered_codes(date):
    """当日すでに記録済みの銘柄コードセットを返す（重複発注防止用）。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT code FROM positions WHERE date=?", [str(date)]
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


# ══════════════════════════════════════════════════════
# closing_log（引け前候補ログ）
# ══════════════════════════════════════════════════════

def save_closing_log_db(rows):
    """
    closing_log テーブルに候補銘柄リストを保存する。
    rows: list of dict (date, code, name, change_pct, rb_score, price, tp_price, sl_price)
    """
    if not rows:
        return
    conn = get_conn()
    conn.executemany("""
        INSERT OR REPLACE INTO closing_log
            (date, code, name, change_pct, rb_score, price, tp_price, sl_price)
        VALUES (:date, :code, :name, :change_pct, :rb_score, :price, :tp_price, :sl_price)
    """, rows)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════
# market_log（日次地合い実績）
# ══════════════════════════════════════════════════════

def save_market_log_db(row):
    """
    market_log テーブルに1日分の地合いデータを保存（INSERT OR REPLACE）。
    row: dict (date, condition, ad_ratio, nikkei_change, up_count, down_count,
               strategy_a_success_rate)
    """
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO market_log
            (date, condition, ad_ratio, nikkei_change,
             up_count, down_count, strategy_a_success_rate)
        VALUES (:date, :condition, :ad_ratio, :nikkei_change,
                :up_count, :down_count, :strategy_a_success_rate)
    """, row)
    conn.commit()
    conn.close()


def get_market_log(date=None):
    """market_log を DataFrame で返す。date 指定時はその日のみ。"""
    conn = get_conn()
    if date:
        df = pd.read_sql(
            "SELECT * FROM market_log WHERE date=?", conn, params=[str(date)]
        )
    else:
        df = pd.read_sql("SELECT * FROM market_log ORDER BY date", conn)
    conn.close()
    return df


def get_prev_day_condition_db(today):
    """market_log から today より前の最新の condition を返す。なければ None。"""
    conn = get_conn()
    row = conn.execute(
        "SELECT condition FROM market_log WHERE date < ? ORDER BY date DESC LIMIT 1",
        [str(today)]
    ).fetchone()
    conn.close()
    return row[0] if row else None


def get_today_condition_db(today):
    """当日の地合いを market_log → morning_log の順で探して返す。なければ 'UNKNOWN'。"""
    conn = get_conn()
    row = conn.execute(
        "SELECT condition FROM market_log WHERE date=?", [str(today)]
    ).fetchone()
    if row:
        conn.close()
        return row[0]
    row = conn.execute(
        "SELECT condition_forecast FROM morning_log WHERE date=?", [str(today)]
    ).fetchone()
    conn.close()
    return row[0] if row else "UNKNOWN"


# ══════════════════════════════════════════════════════
# morning_log（朝スキャン結果）
# ══════════════════════════════════════════════════════

def save_morning_log_db(row):
    """
    morning_log テーブルに朝スキャン結果を保存（INSERT OR REPLACE）。
    row: dict (date, condition_forecast, condition_score, us_avg_change, dow_change,
               nasdaq_change, sp500_change, nikkei_change, usdjpy,
               strategy_a_thr, stop_loss_pct)
    """
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO morning_log
            (date, condition_forecast, condition_score, us_avg_change,
             dow_change, nasdaq_change, sp500_change, nikkei_change,
             usdjpy, strategy_a_thr, stop_loss_pct)
        VALUES (:date, :condition_forecast, :condition_score, :us_avg_change,
                :dow_change, :nasdaq_change, :sp500_change, :nikkei_change,
                :usdjpy, :strategy_a_thr, :stop_loss_pct)
    """, row)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════
# candidates_log（朝スキャン候補ログ）
# ══════════════════════════════════════════════════════

def save_candidates_log_db(rows):
    """
    candidates_log テーブルに候補銘柄を保存（当日分を置換）。
    rows: list of dict (date, strategy, code, condition, name, score, ratio, judgment, reason)
    """
    if not rows:
        return
    if rows:
        today = rows[0].get("date", "")
        conn = get_conn()
        if today:
            conn.execute("DELETE FROM candidates_log WHERE date=?", [str(today)])
        conn.executemany("""
            INSERT OR REPLACE INTO candidates_log
                (date, strategy, code, condition, name, score, ratio, judgment, reason)
            VALUES (:date, :strategy, :code, :condition, :name, :score, :ratio, :judgment, :reason)
        """, rows)
        conn.commit()
        conn.close()


def get_candidates_log(date=None):
    """candidates_log を DataFrame で返す。date 指定時はその日のみ。"""
    conn = get_conn()
    if date:
        df = pd.read_sql(
            "SELECT * FROM candidates_log WHERE date=?", conn, params=[str(date)]
        )
    else:
        df = pd.read_sql("SELECT * FROM candidates_log ORDER BY date", conn)
    conn.close()
    return df


def update_candidates_result(date, strategy, code, open_price, close_price,
                              high_price, low_price, result_pct, result_grade):
    """candidates_log の検証結果列を更新する（scan_daily.py から呼び出し）。"""
    conn = get_conn()
    conn.execute("""
        UPDATE candidates_log
        SET open_price=?, close_price=?, high_price=?, low_price=?,
            result_pct=?, result_grade=?
        WHERE date=? AND strategy=? AND code=?
    """, [open_price, close_price, high_price, low_price,
          result_pct, result_grade, str(date), str(strategy), str(code)])
    conn.commit()
    conn.close()


def migrate_candidates_log_columns():
    """既存DBに result 列が未追加の場合のみ ALTER TABLE で追加する（冪等）。"""
    conn = get_conn()
    existing = {row[1] for row in conn.execute("PRAGMA table_info(candidates_log)").fetchall()}
    for col, typedef in [("open_price", "REAL"), ("close_price", "REAL"),
                          ("high_price", "REAL"), ("low_price", "REAL"),
                          ("result_pct", "REAL"), ("result_grade", "TEXT")]:
        if col not in existing:
            conn.execute(f"ALTER TABLE candidates_log ADD COLUMN {col} {typedef}")
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════
# daytime_signals（日中シグナル）
# ══════════════════════════════════════════════════════

def save_daytime_signal(row):
    """daytime_signals テーブルにシグナルを保存（同日同銘柄は上書き）。
    row: dict (date, time, code, name, price, change_pct, prev_high,
               conditions_met, breakout, volume_surge, momentum,
               volume_pace_ratio, tp_price, sl_price)
    """
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO daytime_signals
            (date, time, code, name, price, change_pct, prev_high,
             conditions_met, breakout, volume_surge, momentum,
             volume_pace_ratio, tp_price, sl_price)
        VALUES (:date, :time, :code, :name, :price, :change_pct, :prev_high,
                :conditions_met, :breakout, :volume_surge, :momentum,
                :volume_pace_ratio, :tp_price, :sl_price)
    """, row)
    conn.commit()
    conn.close()


def get_daytime_signals(date=None):
    """daytime_signals を DataFrame で返す。date 指定時はその日のみ。"""
    conn = get_conn()
    if date:
        df = pd.read_sql(
            "SELECT * FROM daytime_signals WHERE date=? ORDER BY time",
            conn, params=[str(date)]
        )
    else:
        df = pd.read_sql(
            "SELECT * FROM daytime_signals ORDER BY date, time", conn
        )
    conn.close()
    return df
