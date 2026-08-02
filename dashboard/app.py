# -*- coding: utf-8 -*-
"""stock監視ダッシュボード（読み取り専用）。

データソースは fetch_db.py が定期的にS3から取得する data/stock.db 。
このアプリ自体はDBを書き換えない。
"""
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, render_template

app = Flask(__name__)

JST = timezone(timedelta(hours=9))
DB_PATH = Path(__file__).parent / "data" / "stock.db"

CONDITION_BADGE = {
    "STRONG": "good",
    "NORMAL": "info",
    "WEAK": "warning",
    "PANIC": "critical",
}
app.jinja_env.globals["CONDITION_BADGE_MAP"] = CONDITION_BADGE


def get_conn():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_health(conn, now):
    row = conn.execute(
        "SELECT date, condition FROM market_log ORDER BY date DESC LIMIT 1"
    ).fetchone()
    latest_market_date = row["date"] if row else None
    latest_condition = row["condition"] if row else None

    candidates_max = conn.execute(
        "SELECT MAX(date) AS d FROM candidates_log"
    ).fetchone()["d"]

    mtime = datetime.fromtimestamp(DB_PATH.stat().st_mtime, JST)
    age_min = int((now - mtime).total_seconds() // 60)
    today_str = now.strftime("%Y-%m-%d")

    return {
        "age_min": age_min,
        "mtime": mtime.strftime("%Y-%m-%d %H:%M"),
        "latest_market_date": latest_market_date,
        "latest_condition": latest_condition,
        "badge": CONDITION_BADGE.get(latest_condition, "info"),
        "is_today_market_log": latest_market_date == today_str,
        "is_today_candidates": candidates_max == today_str,
    }


def get_open_positions(conn):
    rows = conn.execute(
        """SELECT date, code, name, shares, buy_price, tp_price, sl_price,
                  strategy, buy_time, condition
           FROM positions WHERE status = 'open' ORDER BY buy_time"""
    ).fetchall()

    positions = []
    for r in rows:
        ref = conn.execute(
            """SELECT close, time FROM intraday_prices
               WHERE code = ? AND date = ? ORDER BY time DESC LIMIT 1""",
            (r["code"], r["date"]),
        ).fetchone()
        ref_price = ref["close"] if ref else None
        pnl_pct = (
            round((ref_price - r["buy_price"]) / r["buy_price"] * 100, 2)
            if ref_price
            else None
        )
        positions.append(
            {
                **dict(r),
                "ref_price": ref_price,
                "ref_time": ref["time"] if ref else None,
                "pnl_pct": pnl_pct,
            }
        )
    return positions


def get_today_candidates(conn):
    latest = conn.execute("SELECT MAX(date) AS d FROM candidates_log").fetchone()["d"]
    if not latest:
        return None, []
    rows = conn.execute(
        """SELECT strategy, code, name, condition, score, ratio, judgment, reason
           FROM candidates_log WHERE date = ?
           ORDER BY strategy, judgment, score DESC""",
        (latest,),
    ).fetchall()
    return latest, [dict(r) for r in rows]


def get_trade_history(conn):
    rows = conn.execute(
        """SELECT date, code, name, pnl_pct, sell_time, exit_reason, strategy
           FROM positions WHERE status = 'closed'
           ORDER BY date, sell_time"""
    ).fetchall()

    trades = [dict(r) for r in rows]
    cumulative = 0.0
    wins = 0
    points = []
    for t in trades:
        pnl = t["pnl_pct"] or 0.0
        cumulative += pnl
        t["cumulative_pct"] = round(cumulative, 2)
        points.append(t["cumulative_pct"])
        if pnl > 0:
            wins += 1

    win_rate = round(wins / len(trades) * 100, 1) if trades else None
    return {
        "trades": list(reversed(trades)),
        "points": points,
        "win_rate": win_rate,
        "total_pct": round(cumulative, 2),
        "count": len(trades),
    }


def build_sparkline(points, width=640, height=180, pad=28):
    """累積損益%の推移を表す折れ線SVGを組み立てる（外部JSライブラリ不使用）。"""
    if not points:
        return None

    lo = min(points + [0.0])
    hi = max(points + [0.0])
    span = (hi - lo) or 1.0

    def x_at(i):
        if len(points) == 1:
            return pad
        return pad + (width - 2 * pad) * i / (len(points) - 1)

    def y_at(v):
        return height - pad - (height - 2 * pad) * (v - lo) / span

    coords = [(x_at(i), y_at(v)) for i, v in enumerate(points)]
    path_d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in coords)
    zero_y = y_at(0.0)
    last_x, last_y = coords[-1]

    dots = [
        {"x": round(x, 1), "y": round(y, 1), "value": v}
        for (x, y), v in zip(coords, points)
    ]

    return {
        "width": width,
        "height": height,
        "path_d": path_d,
        "zero_y": round(zero_y, 1),
        "last_x": round(last_x, 1),
        "last_y": round(last_y, 1),
        "last_value": points[-1],
        "dots": dots,
        "hi": round(hi, 2),
        "lo": round(lo, 2),
    }


@app.route("/")
def index():
    now = datetime.now(JST)

    if not DB_PATH.exists():
        return render_template("index.html", db_missing=True, generated_at=now.strftime("%Y-%m-%d %H:%M:%S"))

    conn = get_conn()
    try:
        health = get_health(conn, now)
        positions = get_open_positions(conn)
        candidates_date, candidates = get_today_candidates(conn)
        history = get_trade_history(conn)
    finally:
        conn.close()

    chart = build_sparkline(history["points"])

    return render_template(
        "index.html",
        db_missing=False,
        health=health,
        positions=positions,
        candidates_date=candidates_date,
        candidates=candidates,
        history=history,
        chart=chart,
        generated_at=now.strftime("%Y-%m-%d %H:%M:%S"),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8002, debug=False)
