"""
scan_morning.py - 朝スキャナー（毎朝8:30〜8:50頃に実行）

前日のscan_daily.pyが生成した候補リストに対して
・米国株指数（ダウ・ナスダック・S&P500）
・ドル円
・日経先物（^N225で代用）
をもとに多層スコアリングで「買い推奨 / 見送り / 要注意」を判定して出力する。

使い方:
    python scan_morning.py
"""

import sys
import os
import json
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


class _Tee:
    """標準出力を画面とログファイルの両方へ同時出力する。"""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _run_with_log(fn, log_path):
    """fn()実行中の標準出力を画面とlog_pathの両方に書き出す。
    market_watch.py / closing_watch.py はscan_morning.pyの中で直接関数呼び出しされ
    （daytime.py/position_monitor.pyのようなsubprocess経由ではないため）、
    それらの出力が今までファイルに残らず後から確認できない問題があった。
    """
    orig_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(orig_stdout, f)
        try:
            fn()
        finally:
            sys.stdout = orig_stdout

import db as _db
_db.init_db()

from scan_morning_market_data import (
    fetch_us_market, fetch_nikkei, fetch_sgx, fetch_usdjpy,
    fetch_macro_indicators, get_prev_day_condition, fetch_prev_day_breadth,
    MACRO_SECTORS,
)
from scan_morning_tachibana_session import (
    load_tachibana_url, ensure_tachibana_session, fetch_tachibana_prices,
    _tprice, _tquote, _quote_summary, fetch_stock_hist,
    TACHIBANA_LOGIN_FILE,
)
from scan_morning_strategy_a import (
    SURGE_SCORE_MIN, SURGE_RATIO_MIN, is_surge_flag, judge_entry_a,
)
from scan_morning_strategy_b import judge_entry_b
from scan_morning_macro_scan import (
    scan_strategy_d, calc_stop_loss, calc_score_d,
)

_BASE_DIR          = Path(__file__).parent
SCAN_CSV           = str(_BASE_DIR / "out" / "scan_results.csv")
WATCHLIST_CSV      = str(_BASE_DIR / "out" / "watchlist.csv")
MORNING_LOG_CSV    = str(_BASE_DIR / "out" / "morning_log.csv")
CANDIDATES_LOG_CSV = str(_BASE_DIR / "out" / "candidates_log.csv")

TODAY = datetime.now().strftime("%Y-%m-%d")

# ══════════════════════════════════════════════════════
# 地合い予測（多層スコアリング）
# ══════════════════════════════════════════════════════
def predict_market(us_data, nikkei, usdjpy, sgx=None, prev_condition=None, prev_breadth=None):
    """
    6つの独立したシグナルをスコア化して地合いを判定する。

    満点構成（最大13点）
    ─────────────────────────────────────────
    レイヤー1: 米3指数平均        0〜3点  (weight: 大)
    レイヤー2: ドル円             0〜2点  (weight: 中)
    レイヤー3: 日経先物           0〜3点  (SGX優先・CMEフォールバック)
    レイヤー4: 米指数のばらつき    0〜2点  (spread bonus)
    レイヤー5: 前日WEAK補正        0〜1点  (リバウンドパターン対応)
    レイヤー6: 前日日本市場騰落    0〜2点  ★新規追加
    ─────────────────────────────────────────
    判定ロジック（評価順）
    ─────────────────────────────────────────
    ① PANIC : score<=3 かつ 日経先物<=-1.0%（急落局面を優先捕捉）
    ② STRONG: score>=8 かつ 米指数2つ以上↑
    ③ NORMAL: score>=4
    ④ WEAK  : score>=2
    ⑤ PANIC : それ以外（score<=1 で日経は急落していない）

    [シミュレーション根拠]
    2025-04〜2026-03の1年間・4432銘柄で検証。
    前日の出来高加重平均変化(vw_change)が翌日パフォーマンスと最も相関が高く、
    複合スコア>=8の日は勝率54.5%・日平均+0.170%・Sharpe3.21を達成。
    旧STRONGは勝率51.4%・日平均-0.057%と非採算だった。
    """
    score     = 0
    details   = {}
    breakdown = []   # 表示用の内訳

    # ── Layer 1: 米3指数平均 ──────────────────────── (0〜3点)
    us_changes = [d["change"] for d in us_data.values() if d]
    if us_changes:
        us_avg = sum(us_changes) / len(us_changes)
        if us_avg >= 1.0:
            pts = 3; label = "強い上昇"
        elif us_avg >= 0.0:
            pts = 2; label = "小幅上昇"
        elif us_avg >= -1.0:
            pts = 1; label = "小幅下落"
        elif us_avg >= -2.0:
            pts = 0; label = "下落"
        else:
            pts = -1; label = "急落"
        score += pts
        details["us_avg"] = round(us_avg, 2)
        breakdown.append(f"  米指数平均   : {us_avg:>+.2f}%  → {pts:>+d}点 ({label})")
    else:
        us_avg = None
        breakdown.append("  米指数平均   : 取得失敗  → 0点")

    # ── Layer 2: ドル円 ───────────────────────────── (0〜2点)
    if usdjpy:
        if usdjpy >= 155:
            pts = 2; label = "強い円安（輸出株に追い風）"
        elif usdjpy >= 148:
            pts = 1; label = "緩やかな円安"
        elif usdjpy >= 140:
            pts = 0; label = "中立圏"
        else:
            pts = -1; label = "円高（輸出株に逆風）"
        score += pts
        details["usdjpy"] = usdjpy
        breakdown.append(f"  ドル円       : {usdjpy:.2f}円  → {pts:>+d}点 ({label})")
    else:
        breakdown.append("  ドル円       : 取得失敗  → 0点")

    # ── Layer 3: 日経先物（SGX優先・CMEフォールバック）──── (0〜3点)
    # SGX日経先物は東京開場直前まで動くため、CMEより直近の動きを反映する。
    # 3/27誤判定の教訓: CME -2.08%（8:30時点）が東京開場前に急反転 → SGXで検知可能
    nk_chg = 0.0  # default（取得失敗時はPANIC判定に使わない）
    nk_data   = sgx if sgx is not None else nikkei
    nk_source = "CME円建先物" if sgx is not None else "CME先物(USD)"
    if nk_data:
        nk_chg = nk_data["change"]
        if nk_chg >= 1.0:
            pts = 3; label = "強い上昇"
        elif nk_chg >= 0.0:
            pts = 2; label = "小幅上昇"
        elif nk_chg >= -0.5:
            pts = 1; label = "小幅下落"
        elif nk_chg >= -1.5:
            pts = 0; label = "下落"
        else:
            pts = -1; label = "急落"
        score += pts
        details["nikkei_change"] = nk_chg
        breakdown.append(f"  {nk_source:<10}: {nk_chg:>+.2f}%  → {pts:>+d}点 ({label})")
        # CMEとSGXが両方取得できた場合、乖離があれば参考表示
        if sgx is not None and nikkei is not None:
            cme_usd_chg = nikkei["change"]
            if abs(sgx["change"] - cme_usd_chg) >= 0.5:
                breakdown.append(f"  ※CME(USD)   : {cme_usd_chg:>+.2f}%（円建と{sgx['change']-cme_usd_chg:>+.2f}%乖離）")
    else:
        breakdown.append("  日経先物     : 取得失敗  → 0点")

    # ── Layer 4: 米指数のばらつきボーナス ────────── (0〜2点)
    up_count = 0  # default（STRONG判定に使用）
    if len(us_changes) == 3:
        up_count = sum(1 for c in us_changes if c > 0)
        if up_count == 3:
            pts = 2; label = "3指数一致上昇"
        elif up_count == 2:
            pts = 1; label = "2指数上昇"
        elif up_count == 1:
            pts = 0; label = "2指数下落"
        else:
            pts = -1; label = "3指数一致下落"
        score += pts
        breakdown.append(f"  指数一致度   : {up_count}/3 上昇  → {pts:>+d}点 ({label})")
    else:
        breakdown.append("  指数一致度   : データ不足  → 0点")

    # ── Layer 5: 前日WEAK/PANICリバウンドボーナス ─── (0〜1点)
    # [検証] WEAK翌日はリバウンドが起きやすいパターンが確認されている
    # 3/12 WEAK→3/13 NORMAL(61.3%)、3/17 WEAK→3/18 NORMAL(75%)、3/26 WEAK→3/27 NORMAL(65.9%)
    # 売られすぎの翌朝は反発需要が入りやすく、CME先物の下落が過大評価されやすい
    if prev_condition in ("WEAK", "PANIC"):
        pts = 1
        score += pts
        breakdown.append(f"  前日地合い補正: {prev_condition}→リバウンド期待  → {pts:>+d}点")
    else:
        breakdown.append(f"  前日地合い補正: {prev_condition or '取得失敗'}  → +0点")

    # ── Layer 6: 前日日本市場騰落（キャッシュから計算）─── (0〜2点)
    # [シミュレーション根拠] vw_change(出来高加重平均騰落)が翌日パフォーマンスと
    # 最も相関が高い指標（corr=+0.071）。ad_ratio>=0.60 & vw>=1.0% の組み合わせで
    # 複合スコア>=8 → 勝率54.5%・日平均+0.170%・Sharpe3.21（検証期間2025-04〜2026-03）
    if prev_breadth is not None:
        pts = prev_breadth["score"]
        score += pts
        breakdown.append(f"  前日市場騰落  : {prev_breadth['summary']}  → {pts:>+d}点")
    else:
        breakdown.append("  前日市場騰落  : キャッシュ未取得  → +0点")

    # ── 総合判定 ──────────────────────────────────────
    # 評価順序: PANIC → STRONG → NORMAL → WEAK → PANIC(残余)
    #
    # PANIC: 日経先物が急落(-2.0%以下)かつスコアが極めて低い(2以下)
    #   closing_watchのPANIC基準(PANIC_NIKKEI=-2.0)に統一（旧:-1.0%は誤判定多発）
    #   2026-06-24修正: -1.0%&score<=3 → -2.0%&score<=2（実績: 朝PANIC8件中PANIC確認1件のみ）
    if score <= 2 and nk_chg <= -2.0:
        condition         = "PANIC"
        strategy_a_thr    = 99.0
        stop_loss_pct     = -3.0
    # STRONG: 高スコア + 米3指数のうち2つ以上上昇（過剰判定防止）
    # 閾値を7→8に引き上げ（シミュレーションで旧7は日平均-0.057%、新8は+0.170%）
    elif score >= 8 and up_count >= 2:
        condition         = "STRONG"
        strategy_a_thr    = 7.5
        stop_loss_pct     = -5.0
    elif score >= 4:
        condition         = "NORMAL"
        strategy_a_thr    = 7.5
        stop_loss_pct     = -5.0
    elif score >= 2:
        condition         = "WEAK"
        strategy_a_thr    = 8.0
        stop_loss_pct     = -4.0
    else:  # score <= 1（海外指標がほぼ全滅、日経先物によらずPANIC相当）
        condition         = "PANIC"
        strategy_a_thr    = 99.0
        stop_loss_pct     = -3.0

    details["score"]          = score
    details["strategy_a_thr"] = strategy_a_thr
    details["stop_loss_pct"]  = stop_loss_pct

    return condition, score, breakdown, details


def judge_icon(judgment):
    return {"BUY": "✅ 買い", "CAUTION": "⚠️ 要注意", "PASS": "❌ 見送り"}.get(judgment, "  ")

def condition_icon(cond):
    return {"STRONG": "🚀", "NORMAL": "✅", "WEAK": "⚠️ ", "PANIC": "🚨", "UNKNOWN": "❓"}.get(cond, "")


# ══════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════
def main():
    print(f"=== 🌅 朝スキャナー（{TODAY} 寄り付き前）===\n")

    candidate_rows = []

    # ── 0. DBの鮮度確認 → 古ければS3から取得、今日更新済みならスキップ ──
    import time as _time
    _db_path = Path(__file__).parent / "out" / "stock.db"
    _db_age_hours = (_time.time() - _db_path.stat().st_mtime) / 3600 if _db_path.exists() else 999
    if _db_age_hours > 20:  # 20時間以上古い（=昨日以前）→ S3から取得
        print(f"  DBが{_db_age_hours:.0f}時間前のものです。S3から最新版を取得します...")
        try:
            import download_db
            download_db.run(db_only=True)
            _db.init_db()
        except SystemExit:
            print("  ⚠️  S3ダウンロード失敗 - ローカルDBで続行します")
        except Exception as e:
            print(f"  ⚠️  S3ダウンロードエラー: {e} - ローカルDBで続行します")
    else:
        print(f"  ✅ ローカルDB使用（{_db_age_hours:.1f}時間前に更新済み）")

    # ── Tachibana API セッション確保（期限切れなら電話認証フローを案内）──
    print("  Tachibana API セッション確認中...")
    tachibana_url = ensure_tachibana_session()

    # ── 1. データ取得 ──
    print("  海外市場データ取得中...")
    us_data        = fetch_us_market()
    nikkei         = fetch_nikkei()
    sgx            = fetch_sgx()
    usdjpy         = fetch_usdjpy()
    prev_condition = get_prev_day_condition()
    print("  前日市場騰落データ計算中...")
    prev_breadth   = fetch_prev_day_breadth()

    # ── Tachibana から候補銘柄の現在値を一括取得 ──
    tachibana_prices = {}
    if tachibana_url:
        all_codes = set()
        try:
            df_tmp = _db.get_scan_results()
            if not df_tmp.empty:
                latest = df_tmp["scan_date"].max()
                for c in df_tmp[df_tmp["scan_date"] == latest]["code"].astype(str):
                    all_codes.add(c)
        except Exception:
            pass
        try:
            df_tmp2 = _db.get_watchlist()
            if not df_tmp2.empty:
                for c in df_tmp2["code"].astype(str):
                    all_codes.add(c)
        except Exception:
            pass
        for sector in MACRO_SECTORS:
            for code, _ in sector.get("stocks", []):
                all_codes.add(str(code))
        if all_codes:
            print(f"  Tachibana API: {len(all_codes)}銘柄の現在値取得中...")
            tachibana_prices = fetch_tachibana_prices(tachibana_url, list(all_codes))
            ok = len([v for v in tachibana_prices.values() if v and v.get("price")])
            print(f"  → {ok}/{len(all_codes)}銘柄 取得成功")
    else:
        print("  ⚠️  Tachibana API未接続（tachibana_login_response.json なし）- キャッシュ使用")

    print(f"\n{'='*60}")
    print(f"【海外市場】（前日終値ベース）")
    print(f"{'='*60}")
    for name, d in us_data.items():
        if d:
            icon = "📈" if d["change"] >= 0 else "📉"
            print(f"  {icon} {name:<12}: {d['close']:>10,.2f}  ({d['change']:>+.2f}%)")
        else:
            print(f"  ❓ {name:<12}: 取得失敗")

    if sgx:
        print(f"  🌐 CME円建先物(NIY): {sgx['close']:>10,.0f}  ({sgx['change']:>+.2f}%)  ← 直近値")
    else:
        print(f"  ❓ CME円建先物(NIY): 取得失敗")
    if nikkei:
        print(f"  🌐 CME先物(NKD/USD): {nikkei['close']:>10,.0f}  ({nikkei['change']:>+.2f}%)")
    else:
        print(f"  ❓ CME先物(NKD/USD): 取得失敗")

    if usdjpy:
        print(f"  💴 ドル円      : {usdjpy:>10.2f} 円")

    if prev_condition:
        print(f"  📅 前日地合い  : {prev_condition}")
    else:
        print(f"  📅 前日地合い  : 取得失敗（market_log.csvなし）")

    # ── 2. 地合い予測（多層スコアリング）──
    condition, score, breakdown, market_details = predict_market(
        us_data, nikkei, usdjpy, sgx=sgx, prev_condition=prev_condition,
        prev_breadth=prev_breadth
    )
    strategy_a_thr = market_details["strategy_a_thr"]
    stop_loss_pct  = market_details["stop_loss_pct"]
    cond_icon      = condition_icon(condition)

    print(f"\n{'='*60}")
    print(f"【今日の地合い予測】（多層スコアリング）")
    print(f"{'='*60}")
    print(f"\n  📊 スコア内訳:")
    for line in breakdown:
        print(line)

    max_score = 10
    bar_filled = max(0, min(score, max_score))
    bar = "█" * bar_filled + "░" * (max_score - bar_filled)
    print(f"\n  総合スコア   : {score:>+d} / {max_score}点  [{bar}]")
    print(f"  判定         : {cond_icon} {condition}")
    print()

    if condition == "STRONG":
        print(f"  戦略A      : 積極エントリー可")
        print(f"  推奨条件   : score5〜8 / ratio<5倍 / 前日比<+2% / 上位3件")
        print(f"  TP/SL      : +3%利確 / -1%損切り")
        print(f"  ※スイープ検証: Sharpe4.21 / 年率+10.8% / DD-1.6%（アウトオブサンプル確認済み）")
    elif condition == "NORMAL":
        print(f"  戦略A      : ✅ 有望条件のみ（前日微下落 × スコア3〜6 × ratio<3倍）")
        print(f"  closing_watch: ✅ NORMAL日も戦略B対象（14:30〜自動起動）")
    elif condition == "WEAK":
        print(f"  戦略A      : ⚠️  スコア{strategy_a_thr}以上のみ検討")
        print(f"  closing_watch: ⚠️  WEAK日は見送り")
    elif condition == "PANIC":
        print(f"  戦略A      : ❌ 全見送り推奨")
        print(f"  closing_watch: ❌ 全見送り推奨")
        print(f"  → ノーポジが最強戦略です")
    else:
        print(f"  予測       : ❓ データ取得失敗（慎重に判断してください）")

    print(f"{'='*60}")

    # ── 3. 戦略A候補の判定 ──
    print(f"\n{'='*60}")
    print(f"【戦略A】順張り候補 - 本日エントリー判定")
    print(f"{'='*60}")

    df_a = _db.get_scan_results()
    if df_a.empty:
        print("  ⚠️  戦略Aデータが見つかりません")
        print("  → 前日にscan_daily.pyを実行してください")
    else:
        latest_date = df_a["scan_date"].max()
        candidates_a = df_a[df_a["scan_date"] == latest_date].copy()
        candidates_a = candidates_a.sort_values(["score", "ratio"], ascending=[False, False])
        print(f"  スキャン日: {latest_date}  候補数: {len(candidates_a)}件\n")

        buy_a = caution_a = pass_a = 0
        print(f"  {'判定':<10} {'コード':<6} {'銘柄名':<16} {'現在値':>8} {'売気配':>7} {'買気配':>7} {'需給':>5} {'スコア':>6} {'比率':>5}  理由")
        print("  " + "─" * 95)

        surge_candidates = []
        for _, row in candidates_a.iterrows():
            judgment, reason = judge_entry_a(row, condition, strategy_a_thr)
            surge = is_surge_flag(row)
            candidate_rows.append({
                "date": TODAY, "strategy": "A",
                "condition": condition,
                "code": row["code"], "name": row["name"],
                "score": row["score"], "ratio": row["ratio"],
                "judgment": judgment, "reason": reason,
            })
            icon = judge_icon(judgment)
            surge_mark = " ⚡" if surge else ""
            if judgment == "BUY":      buy_a    += 1
            elif judgment == "CAUTION": caution_a += 1
            else:                       pass_a    += 1
            q = _tquote(tachibana_prices, row["code"])
            rt = q["price"] if q else None
            price_str = f"{rt:>8,.0f}円" if rt else f"{'---':>8}  "
            ask_s, bid_s, ratio_s = _quote_summary(q)
            print(f"  {icon:<10} {str(row['code']):<6} {str(row['name'])[:14]:<16} "
                  f"{price_str} {ask_s} {bid_s} {ratio_s} "
                  f"{float(row['score']):>6.2f} {float(row['ratio']):>5.1f}倍{surge_mark}  {reason}")
            if surge:
                surge_candidates.append(row)

        print(f"\n  【集計】 ✅買い:{buy_a}件  ⚠️要注意:{caution_a}件  ❌見送り:{pass_a}件")

        # 急騰フラグ銘柄の専用表示
        if surge_candidates:
            print(f"\n  {'='*56}")
            print(f"  ⚡【急騰候補】score>={SURGE_SCORE_MIN} & ratio>={SURGE_RATIO_MIN:.0f}倍以上  {len(surge_candidates)}件")
            print(f"  {'='*56}")
            print(f"  過去実績: 高値+10%以上到達率10.6% / TP+8%設定で平均+1.02%")
            print(f"  推奨: TP+8% / SL-1%（引け決済は不可 → 高値で反落するケース多）")
            print(f"  {'─'*56}")
            for row in surge_candidates:
                jdg, _ = judge_entry_a(row, condition, strategy_a_thr)
                icon = judge_icon(jdg)
                note = "→ 地合い不問で注目" if jdg != "BUY" else "→ BUY推奨"
                print(f"  {icon} [{row['code']}]{str(row['name'])[:16]}  "
                      f"score:{float(row['score']):.1f}  ratio:{float(row['ratio']):.1f}倍  {note}")
                print(f"    推奨注文: 寄付き成行 / TP+8% / SL-1%")

    # ── 4. 前日closing_watch 仕込み済み確認 ──
    # 戦略B（逆張り）はclosing_watchに統合済み。
    # 前日14:58に引け前で仕込んだポジションの現在値を参考表示する。
    print(f"\n{'='*60}")
    print(f"【前日closing_watch 仕込み済みポジション確認】")
    print(f"  ※ 逆張り買いは前日引け前（closing_watch）で実施済み")
    print(f"  ※ 本日の決済目安: TP+1.9% / SL-5.6%（GA最適化結果 2026-05-26）")
    print(f"{'='*60}")

    df_b = _db.get_watchlist()
    if df_b.empty:
        print("  前日のclosing_watchデータが見つかりません")
    else:
        latest_date_b = df_b["buy_date"].max()
        candidates_b  = df_b[
            (df_b["buy_date"] == latest_date_b) & df_b["next_rise"].isna()
        ].copy()

        if candidates_b.empty:
            print(f"  前日（{latest_date_b}）の仕込み銘柄なし（STRONG地合い以外 or 条件該当なし）")
        else:
            print(f"  仕込み日: {latest_date_b}  銘柄数: {len(candidates_b)}件\n")
            print(f"  {'コード':<6} {'銘柄名':<16} {'現在値':>8} {'売気配':>7} {'買気配':>7} {'需給':>5} {'仕込み騰落':>10} {'TP目安':>8} {'SL目安':>8} {'RB':>4}")
            print("  " + "─" * 100)

            for _, row in candidates_b.iterrows():
                buy_p    = float(row["buy_price"])
                tp_price = round(buy_p * 1.019)
                sl_price = round(buy_p * 0.944)
                drop_str = f"{float(row['today_rise']):>+.1f}%"
                rb_score = int(row.get("rebound_score", 0))
                q = _tquote(tachibana_prices, row["code"])
                rt = q["price"] if q else None
                price_str = f"{rt:>8,.0f}円" if rt else f"{'---':>8}  "
                ask_s, bid_s, ratio_s = _quote_summary(q)
                print(f"  {str(row['code']):<6} {str(row['name'])[:14]:<16} "
                      f"{price_str} {ask_s} {bid_s} {ratio_s} {drop_str:>10} "
                      f"{tp_price:>8,}円 {sl_price:>8,}円 {rb_score:>4}点")
                
    # ── 4.5 戦略D: 大型株マクロ連動スキャン ──
    macro = fetch_macro_indicators(usdjpy)
    scan_strategy_d(macro, condition, tachibana_prices=tachibana_prices)

    # ── 5. 候補銘柄ログ保存 ──
    if candidate_rows:
        _db.save_candidates_log_db(candidate_rows)

     # ── 6. 本日のアクションプラン ──
    print(f"\n{'='*60}")
    print(f"【本日のアクションプラン】")
    print(f"{'='*60}")

    if condition == "PANIC":
        print(f"  🚨 全ポジション見送り。ノーポジが最強戦略です。")
        print(f"\n  ⏰ 本日のチェックリスト")
        print(f"  □ 新規エントリーは一切しない")
        print(f"  □ 保有中のポジションがあれば損切りを検討")
        print(f"  □ 米国先物・ニュースの動向を引き続き監視")
        print(f"  □ 地合い回復のシグナル（騰落比40%超）を待つ")
    elif condition == "WEAK":
        print(f"  ⚠️  スコア{strategy_a_thr}以上の戦略A候補のみ、少額で検討。")
        print(f"  ⚠️  損切りを{stop_loss_pct:.0f}%に設定（通常より引き締め）。")
        print(f"  ❌ closing_watch本日は見送り（WEAK地合いは逆張り非推奨）")
        print(f"\n  ⏰ チェックリスト（8:50まで）")
        print(f"  □ 候補銘柄のチャートを確認（前日夜〜今朝のPTS動向）")
        print(f"  □ 候補銘柄の最新ニュース・IR確認")
        print(f"  □ 損切りライン: 戦略A=寄付きから-1%")
        print(f"  □ 前日closing_watch仕込み済みポジションは早めの損切り検討")
        print(f"  □ 1銘柄あたりの投資額を確認（リスク管理）")
    elif condition == "STRONG":
        print(f"  🚀 地合い良好。戦略Aの優良銘柄を優先。")
        print(f"  ✅ closing_watch 14:30〜 自動起動（scan_morningから連続実行）")
        print(f"\n  ⏰ チェックリスト（8:50まで）")
        print(f"  □ 候補銘柄のチャートを確認（前日夜〜今朝のPTS動向）")
        print(f"  □ 候補銘柄の最新ニュース・IR確認")
        print(f"  □ 損切りライン: 戦略A=寄付きから-1%")
        print(f"  □ 1銘柄あたりの投資額を確認（リスク管理）")
    else:  # NORMAL
        print(f"  ✅ 通常通りスキャン結果に従ってエントリー。")
        print(f"  📌 9:00直後の値動きで方向感を確認してから判断もOK。")
        print(f"  ℹ️  closing_watchはNORMAL・STRONG地合いで本日起動可（WEAK・PANICは見送り）")
        print(f"\n  ⏰ チェックリスト（8:50まで）")
        print(f"  □ 候補銘柄のチャートを確認（前日夜〜今朝のPTS動向）")
        print(f"  □ 候補銘柄の最新ニュース・IR確認")
        print(f"  □ 損切りライン: 戦略A=寄付きから-1%")
        print(f"  □ 1銘柄あたりの投資額を確認（リスク管理）")

    # ── 7. 朝判定ログ保存 ──
    log_row = {
        "date":               TODAY,
        "condition_forecast": condition,
        "condition_score":    score,
        "us_avg_change":      market_details.get("us_avg"),
        "dow_change":         us_data.get("ダウ", {}).get("change")     if us_data.get("ダウ")      else None,
        "nasdaq_change":      us_data.get("ナスダック", {}).get("change") if us_data.get("ナスダック") else None,
        "sp500_change":       us_data.get("S&P500", {}).get("change")   if us_data.get("S&P500")   else None,
        "nikkei_change":      nikkei["change"] if nikkei else None,
        "usdjpy":             usdjpy,
        "strategy_a_thr":     strategy_a_thr,
        "stop_loss_pct":      stop_loss_pct,
    }
    _db.save_morning_log_db(log_row)

    print(f"\n✅ 朝スキャン完了（DB記録）")
    print(f"   候補銘柄ログ: {len(candidate_rows)}件記録")
    print(f"{'='*60}\n")

    import subprocess, sys as _sys

    _OUT_DIR = _BASE_DIR / "out"

    # ── position_monitor: 地合いに関係なく常に起動 ──
    # 既存ポジションのTP/SL監視・前日ポジションの強制決済が役目なので、
    # 新規発注をしないPANIC日こそ、保有中ポジションの監視を止めてはいけない。
    # market_watch（ブロッキング）より前に起動することで9:00から並行稼働させる
    try:
        subprocess.Popen(
            [_sys.executable, "-u", str(_BASE_DIR / "position_monitor.py")],
            stdout=open(str(_OUT_DIR / "position_monitor.log"), "w", encoding="utf-8"),
            stderr=open(str(_OUT_DIR / "position_monitor_err.log"), "w", encoding="utf-8"),
        )
        print("  📊 position_monitor をバックグラウンドで起動しました（15:28まで監視）")
        print(f"     ログ: out/position_monitor.log")
    except Exception as e:
        print(f"⚠️  position_monitor の起動に失敗しました: {e}")
        print("   → python position_monitor.py を手動で実行してください")

    # ── daytime: PANIC以外のみ起動（新規シグナル検知が役目のため）──
    if condition != "PANIC":
        try:
            subprocess.Popen(
                [_sys.executable, "-u", str(_BASE_DIR / "daytime.py")],
                stdout=open(str(_OUT_DIR / "dt_live.txt"), "w", encoding="utf-8"),
                stderr=open(str(_OUT_DIR / "dt_err.txt"), "w", encoding="utf-8"),
            )
            print("  📈 daytime をバックグラウンドで起動しました（9:00〜14:30 自動売買）")
            print(f"     ログ: out/dt_live.txt")
        except Exception as e:
            print(f"⚠️  daytime の起動に失敗しました: {e}")

    # ── 戦略A：BUY/CAUTION 候補がある場合のみ寄り付き監視・発注 ──
    has_buy = any(r["judgment"] in ("BUY", "CAUTION") for r in candidate_rows)
    if has_buy and condition != "PANIC":
        try:
            import market_watch
            print(f"  ログ: out/market_watch_live.txt")
            _run_with_log(market_watch.main, str(_OUT_DIR / "market_watch_live.txt"))
        except Exception as e:
            print(f"⚠️  リアルタイム監視でエラーが発生しました: {e}")
            print("   → python market_watch.py を手動で実行してください")

    # ── closing_watch（ブロッキング）──
    if condition != "PANIC":
        try:
            import closing_watch
            print(f"  ログ: out/closing_watch_live.txt")
            _run_with_log(closing_watch.main, str(_OUT_DIR / "closing_watch_live.txt"))
        except Exception as e:
            print(f"⚠️  引け前スキャンでエラーが発生しました: {e}")
            print("   → python closing_watch.py を手動で実行してください")

    # ── closing_watch 終了後、16:45 に scan_daily.py を自動実行（データ未取得時は15分おきにリトライ）──
    _run_scan_daily_at_1640()


def _run_scan_daily_at_1640():
    """closing_watch 終了後、16:45 まで待機して scan_daily.py を自動実行する。
    データ未取得の場合は 15 分おきに 20:00 まで最大 3 回リトライする。"""
    import subprocess as _subprocess
    import sys as _sys2
    import time as _time
    import sqlite3 as _sqlite3
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    _JST    = _tz(_td(hours=9))
    now     = _dt.now(_JST)
    target  = now.replace(hour=16, minute=45, second=0, microsecond=0)
    today   = now.strftime("%Y-%m-%d")

    if now < target:
        wait = int((target - now).total_seconds())
        print(f"\n  ⏳ scan_daily.py を 16:45 に自動実行します（あと {wait//60}分{wait%60}秒）")
        _time.sleep(wait)

    def _has_today_data():
        try:
            conn = _sqlite3.connect(str(_BASE_DIR / "out" / "stock.db"))
            cnt = conn.execute(
                "SELECT COUNT(*) FROM daily_prices WHERE Date=?", (today,)
            ).fetchone()[0]
            conn.close()
            return cnt > 0
        except Exception:
            return False

    max_retries = 3
    retry_interval = 15 * 60  # 15分

    for attempt in range(max_retries + 1):
        now_str = _dt.now(_JST).strftime("%H:%M")
        print(f"\n{'='*60}")
        if attempt == 0:
            print(f"  📊 scan_daily.py を自動起動します（{now_str}）")
        else:
            print(f"  🔄 scan_daily.py リトライ {attempt}/{max_retries}（{now_str}）")
        print(f"{'='*60}")

        try:
            _subprocess.run([_sys2.executable, str(_BASE_DIR / "scan_daily.py")])
        except Exception as e:
            print(f"⚠️  scan_daily.py の実行に失敗しました: {e}")

        if _has_today_data():
            print(f"  ✅ {today} のデータ取得完了")
            return

        if attempt < max_retries:
            next_time = (_dt.now(_JST) + _td(seconds=retry_interval)).strftime("%H:%M")
            if _dt.now(_JST).hour >= 20:
                print(f"  ⚠️  20:00 を過ぎたためリトライ中止")
                break
            print(f"  ⚠️  データ未取得 → {next_time} に再試行します")
            _time.sleep(retry_interval)

    print(f"  ❌ データ取得できませんでした → python scan_daily.py を手動で実行してください")


if __name__ == "__main__":
    main()


