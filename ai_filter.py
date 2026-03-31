"""
ai_filter.py - AI補助判断（scan_morning.py の後に実行）

scan_morning.py が出力した本日のBUY/CAUTION候補を Claude API に渡し、
・Google News RSSで銘柄ニュースを自動取得（Step 1）
・チャート詳細分析（MA・RSI・出来高推移・価格レンジ）をAIへ送信（Step 2）
・決算発表チェック（Yahoo Finance Japanから次回決算日を取得）（Step 3）
・懸念点・推奨度・コメントをAIが分析して表示する。

実行方法:
    python ai_filter.py

前提:
    - scan_morning.py を先に実行済みであること
    - .env に ANTHROPIC_API_KEY が設定されていること
    - pip install anthropic feedparser beautifulsoup4
"""

import os
import sys
import json
import time
import re
import urllib.parse
import numpy as np
import pandas as pd
import anthropic
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

CANDIDATES_LOG_CSV = "out/candidates_log.csv"
MORNING_LOG_CSV    = "out/morning_log.csv"
CACHE_DIR          = "out/cache"
TODAY              = datetime.now().strftime("%Y-%m-%d")

# 使用するClaudeモデル
# Haiku: 高速・低コスト（朝のスキャン補助に最適）
# Opus:  高精度・低速（より詳細な分析が必要な場合）
MODEL = "claude-haiku-4-5-20251001"

# ニュース取得件数（銘柄ごと）
NEWS_MAX = 3


# ══════════════════════════════════════════════
# 銘柄ニュース取得（Google News RSS）
# ══════════════════════════════════════════════
def fetch_news(code, name):
    """Google News RSSで銘柄の直近ニュースを取得する。
    コード番号と銘柄名の両方で検索し、最新NEWS_MAX件を返す。
    """
    results = []
    # 銘柄名で検索（例: 「トヨタ自動車 株」）日本語をURLエンコード
    query = urllib.parse.quote(f"{name} 株")
    url   = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"

    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:NEWS_MAX]:
            title     = entry.get("title", "")
            published = entry.get("published", "")
            # 日付を短縮表示（例: "Fri, 28 Mar 2026 02:00:00 GMT" → "03/28"）
            try:
                dt  = datetime(*entry.published_parsed[:3])
                pub = dt.strftime("%m/%d")
            except Exception:
                pub = published[:10] if published else "日付不明"
            results.append({"title": title, "date": pub})
    except Exception:
        pass

    # 取得できなかった場合
    if not results:
        results.append({"title": "ニュース取得失敗（ネットワーク確認）", "date": ""})

    return results


# ══════════════════════════════════════════════
# チャート詳細分析（Step 2: MA・RSI・出来高推移）
# ══════════════════════════════════════════════
def calc_rsi(closes, period=14):
    """RSI(14)を計算する"""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs  = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def fetch_chart_detail(code):
    """キャッシュデータから詳細なチャート分析情報を取得してdictで返す。
    MA5・MA25・RSI・出来高推移・価格レンジ・直近10日の日足サマリーを含む。
    AIがより精度の高い判断をするために使用する（Step 2）。
    """
    cache_path = f"{CACHE_DIR}/{code}.csv"
    if not os.path.exists(cache_path):
        return {"エラー": "キャッシュデータなし"}

    try:
        df = pd.read_csv(cache_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        if len(df) < 26:
            return {"エラー": f"データ不足({len(df)}日分)"}

        closes  = df["Close"].astype(float).values
        volumes = df["Volume"].astype(float).values

        # ── 基本指標 ──
        latest   = df.iloc[-1]
        prev     = df.iloc[-2]
        day_chg  = round((float(latest["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100, 2)

        # ── 移動平均 ──
        ma5   = round(float(np.mean(closes[-5:])), 1)
        ma25  = round(float(np.mean(closes[-25:])), 1)
        ma_diff_pct = round((ma5 / ma25 - 1) * 100, 2)  # MA5がMA25より何%上か

        # ── RSI(14) ──
        rsi = calc_rsi(closes)

        # ── 出来高分析 ──
        vol_avg20  = round(float(np.mean(volumes[-20:])), 0)
        vol_latest = float(latest["Volume"])
        vol_ratio  = round(vol_latest / vol_avg20, 1) if vol_avg20 > 0 else 0

        # ── 価格レンジ（20日）──
        high20 = round(float(np.max(df["High"].tail(20))), 0)
        low20  = round(float(np.min(df["Low"].tail(20))), 0)
        # 現在値が20日レンジの何%の位置か（0%=最安値、100%=最高値）
        range_pos = round((float(latest["Close"]) - low20) / (high20 - low20) * 100, 0) if high20 != low20 else 50

        # ── 直近5日間の日足サマリー ──
        # 前日比は close-to-close で統一（open-to-close ではなく）
        tail5 = df.tail(6).reset_index(drop=True)  # 1行余分に取って前日比計算に使う
        recent_days = []
        for i in range(max(1, len(tail5) - 5), len(tail5)):
            r       = tail5.iloc[i]
            r_prev  = tail5.iloc[i - 1]
            chg_cc  = round((float(r["Close"]) - float(r_prev["Close"])) / float(r_prev["Close"]) * 100, 1)
            recent_days.append({
                "日付":   str(r["Date"])[:10],
                "終値":   int(r["Close"]),
                "前日比": f"{chg_cc:+.1f}%",
                "出来高": f"{int(r['Volume']):,}",
            })

        # ── トレンド判定 ──
        close5  = closes[-5:]
        up_days = sum(1 for i in range(1, 5) if close5[i] > close5[i-1])
        if up_days >= 4:
            trend = "強い上昇トレンド"
        elif up_days == 3:
            trend = "上昇傾向"
        elif up_days == 2:
            trend = "横ばい"
        elif up_days == 1:
            trend = "下落傾向"
        else:
            trend = "強い下落トレンド"

        return {
            "直近終値":       int(latest["Close"]),
            "前日比":         f"{day_chg:+.1f}%",
            "MA5":            ma5,
            "MA25":           ma25,
            "MA5_MA25乖離":   f"{ma_diff_pct:+.2f}%（{'ゴールデンクロス圏' if ma_diff_pct > 0 else 'デッドクロス圏'}）",
            "RSI14":          f"{rsi}（{'買われすぎ' if rsi and rsi > 70 else '売られすぎ' if rsi and rsi < 30 else '中立'}）" if rsi else "計算不可",
            "出来高比率":      f"{vol_ratio}倍（20日平均比）",
            "20日高値安値":   f"高値{int(high20)}円 / 安値{int(low20)}円",
            "レンジ位置":      f"{int(range_pos)}%（0%=安値圏 / 100%=高値圏）",
            "5日トレンド":    trend,
            "直近5日足":      recent_days,
        }

    except Exception as e:
        return {"エラー": f"解析失敗: {e}"}


# ══════════════════════════════════════════════
# 決算発表チェック（Step 3: Yahoo Finance Japanから取得）
# ══════════════════════════════════════════════
_YF_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def _parse_earnings_text(text):
    """「次回の決算発表日は2026年5月8日です。」等のテキストから日付を解析。
    具体的な日付が取れた場合はdatetime、取れない場合は説明テキストを返す。
    """
    # 具体的な日付パターン（例: 2026年5月8日）
    m = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # 上旬・中旬・下旬パターン（例: 2026年5月上旬）
    m2 = re.search(r'(\d{4})年(\d{1,2})月(上旬|中旬|下旬)', text)
    if m2:
        year, month = int(m2.group(1)), int(m2.group(2))
        period = m2.group(3)
        # 期間の中央値で近似
        day = {"上旬": 5, "中旬": 15, "下旬": 25}[period]
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    return None


def fetch_earnings_info(code):
    """Yahoo Finance Japanから次回決算発表日を取得する（Step 3）。
    決算またぎは予測不可能なリスクのため、本日〜3日以内の決算はPASS推奨。
    戻り値: {
        "date_text": "2026年5月上旬頃",
        "date":      datetime or None,
        "risk":      "高" / "中" / "低" / "不明",
        "risk_reason": "本日決算" / "3日以内" / "今月決算" / "問題なし" / "取得失敗"
    }
    """
    url = f"https://finance.yahoo.co.jp/quote/{code}.T"
    try:
        resp = requests.get(url, headers=_YF_HEADERS, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")

        date_text = None
        for tag in soup.find_all(string=True):
            t = str(tag).strip()
            if "次回の決算発表日" in t:
                date_text = t
                break

        if not date_text:
            return {"date_text": "取得失敗", "date": None, "risk": "不明", "risk_reason": "取得失敗"}

        earnings_dt = _parse_earnings_text(date_text)
        today       = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        if earnings_dt is None:
            return {"date_text": date_text, "date": None, "risk": "不明", "risk_reason": "日付解析失敗"}

        days_until = (earnings_dt - today).days

        if days_until == 0:
            risk, reason = "高", "本日決算（絶対見送り）"
        elif days_until <= 3:
            risk, reason = "高", f"{days_until}日後に決算（見送り推奨）"
        elif days_until <= 14:
            risk, reason = "中", f"{days_until}日後に決算（要注意）"
        else:
            risk, reason = "低", f"{days_until}日後（問題なし）"

        return {
            "date_text":   date_text,
            "date":        earnings_dt,
            "risk":        risk,
            "risk_reason": reason,
        }

    except Exception as e:
        return {"date_text": "取得失敗", "date": None, "risk": "不明", "risk_reason": f"エラー: {e}"}


# ══════════════════════════════════════════════
# 本日の候補銘柄を取得
# ══════════════════════════════════════════════
def load_today_candidates():
    """本日のBUY/CAUTION候補をcandidates_log.csvから取得"""
    if not os.path.exists(CANDIDATES_LOG_CSV):
        print("❌ out/candidates_log.csv が見つかりません")
        print("   → まず scan_morning.py を実行してください")
        return None, None

    df = pd.read_csv(CANDIDATES_LOG_CSV, encoding="utf-8-sig")
    today_df = df[df["date"] == TODAY].copy()

    if today_df.empty:
        print(f"❌ 本日（{TODAY}）の候補データがありません")
        print("   → まず scan_morning.py を実行してください")
        return None, None

    # BUY/CAUTIONのみ（PASSは除外）
    targets   = today_df[today_df["judgment"].isin(["BUY", "CAUTION"])].copy()
    condition = today_df["condition"].iloc[0] if not today_df.empty else "UNKNOWN"

    return targets, condition


def load_today_market():
    """本日の地合いスコア詳細をmorning_log.csvから取得"""
    if not os.path.exists(MORNING_LOG_CSV):
        return {}
    try:
        df    = pd.read_csv(MORNING_LOG_CSV, encoding="utf-8-sig")
        today = df[df["date"] == TODAY]
        if not today.empty:
            return today.iloc[0].to_dict()
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════
# Claude APIを呼び出す
# ══════════════════════════════════════════════
def ask_claude(candidates_df, condition, market_info, news_data, chart_data, earnings_data=None):
    """候補銘柄・ニュース・チャートをClaudeに渡して補助判断を求める"""

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # 候補銘柄をJSON形式に整形（ニュース・チャート詳細を含む）
    candidates_list = []
    for _, row in candidates_df.iterrows():
        code = str(row["code"])
        ei = (earnings_data or {}).get(code, {})
        candidates_list.append({
            "コード":        code,
            "銘柄名":        str(row["name"]),
            "戦略":          str(row["strategy"]),
            "スコア":        float(row["score"]),
            "判定":          str(row["judgment"]),
            "判定理由":      str(row["reason"]),
            "チャート詳細":  chart_data.get(code, {"エラー": "データなし"}),
            "直近ニュース":  news_data.get(code, []),
            "決算リスク":    ei.get("risk", "不明"),
            "決算情報":      ei.get("risk_reason", "取得失敗"),
        })

    # 市場情報を整形
    market_summary = (
        f"地合い: {condition}\n"
        f"日経先物変化率: {market_info.get('nikkei_change', '不明')}%\n"
        f"ドル円: {market_info.get('usdjpy', '不明')}円\n"
        f"米指数平均変化率: {market_info.get('us_avg_change', '不明')}%"
    )

    # 地合いによって分析方針を切り替える
    if condition == "STRONG":
        strategy_guide = """## 地合い：STRONG（強気相場）
【パラメータスイープ検証済みルール（sweep.py・2025-04〜2026-03・アウトオブサンプル確認済み）】

★ STRONG日の採用条件（3つ全て満たす銘柄のみ推奨）：
  1. スコア5.0以上7.0未満（5未満は不安定・7以上は出尽くし）
  2. 出来高倍率3倍未満（ratio<3倍）※3倍超は過熱サイン・除外
  3. 前日比+2%未満（すでに動いた銘柄は出尽くし・除外）

理由：高スコア＝前日に出来高が爆発的急増＝「出尽くし」で翌日売られやすい
      ratio3倍超＝過熱しすぎ・翌日失速リスク高
      前日急騰済み＝材料が既に織り込まれている

推奨売買ルール（戦略A）：
  - 上位3件に絞る（top3）
  - TP+3%指値 / SL-1%逆指値
  - ギャップアップ+5%超の銘柄は除外
  - 検証結果：Sharpe4.21 / 年率+10.8% / DD-1.6%（アウトオブサンプル検証済み）

戦略B（逆張り）が混在する場合（前日-4%以下の銘柄）：
  - RB4以上のみ推奨（RB3はCAUTION止まり）
  - 【最重要】寄り付きがギャップアップ（前日比プラス）の場合は見送り
    理由：前日下落の修正が寄り付きで完了→当日の上昇余地なし
    STRONG×ギャップなし×引け決済: Out avg+1.0〜1.3%・勝率65%（最優秀）
  - 出口：引け決済 or TP+5%/SL-5%（TP+3%/SL-3%は利確しすぎ）"""

    elif condition == "WEAK":
        strategy_guide = """## 地合い：WEAK（軟調相場）
【シミュレーション検証済みルール（simulate_b.py・2025-04〜2026-03・3216件）】

★ WEAK日は戦略A・戦略B ともに見送りが基本
  - 戦略A（順張り）: 全見送り（WEAK日は下落継続しやすい）
  - 戦略B（逆張り）: WEAK日avg -0.82%・勝率31% ← 最も悪い地合い
    理由：市場全体が弱い日に下がった銘柄は翌日もさらに下がりやすい

★ 例外的に検討できる条件
  - リバウンドスコア3〜6 かつ 個別材料（好決算・業績上方修正）がある場合のみ「様子見」
  - スコア7以上は「落ちナイフ」→ 絶対見送り
  - 確信が持てない場合は「見送り推奨」を積極的に使うこと"""

    elif condition == "PANIC":
        strategy_guide = """## 地合い：PANIC（大暴落）
【シミュレーション検証済みルール】

全銘柄を「見送り推奨」にすること。例外なし。
理由：PANIC日の翌日平均リターン -0.83%・勝率32.5%（211件）
ノーポジが最強戦略。"""

    else:  # NORMAL
        strategy_guide = """## 地合い：NORMAL（通常相場）
【シミュレーション検証済みルール（2025-04〜2026-03・16865件）】

★ NORMAL日の基本姿勢：厳選が最重要。安易なBUYは避ける。
  - NORMAL全体: 引け勝率46.2% / 平均-0.099%（全銘柄を買うと負け）

■ 有望な条件（AIがニュース・チャートで上乗せ判断すべきゾーン）
  1. 前日比マイナス(-1〜0%)の銘柄 × スコア0〜7
     → 勝率50.8〜51.9% / 平均+0.13〜0.20%（わずかにプラス）
     理由：「今日少し下げた→明日リバウンド期待」の銘柄が優秀
  2. 窓下げスタート（ギャップ-2〜-5%）
     → 勝率50.4% / 平均+0.198%（下げすぎ修正狙い）

■ 避けるべき条件
  - ギャップアップ+2%超: 勝率40.7% / 平均-0.601%（高値掴み）
  - 出来高比率5倍超: 勝率41.2% / 平均-0.407%（出尽くし）
  - 当日+5%超の急騰銘柄: 勝率41.4% / 平均-0.427%（追いかけ禁止）

■ 戦略B（逆張り・リバウンド狙い）が混在する場合
  - 戦略Bは「前日-4%以下下落 × RBスコア3以上」の銘柄
  - 【最重要】寄り付きがプラス（ギャップアップ）の場合は必ず「見送り推奨」
    理由: ギャップアップ排除だけで勝率41%→67%に改善（STRONG日）
  - NORMAL日 × ギャップアップなし × 引け決済: 勝率50.5% / avg+0.219%★
  - 出口戦略: 引け決済 or TP+5%/SL-5% → TP+3%/SL-3%は利確しすぎ
  - RBスコア7〜9でも有効（落ちナイフ扱い不要）

■ AIに期待すること
  - 上記「有望条件」を満たした銘柄の中から、ニュース・チャートで本物を見極める
  - 「前日比微マイナス + 良いニュース + RSI中立圏」→「買い推奨」候補
  - 「前日比微マイナス + 悪いニュース or 下落トレンド」→「見送り推奨」
  - 確信が持てない場合は「様子見」を積極的に使うこと"""

    prompt = f"""あなたは日本株のデイトレード補助AIです。
1年間・4432銘柄のバックテストで検証済みのルールに基づいて、各候補銘柄を分析してください。

## 本日の市場状況
{market_summary}

{strategy_guide}

## 候補銘柄（スキャンシステムがBUY/CAUTIONと判定・ニュース・チャート付き）
{json.dumps(candidates_list, ensure_ascii=False, indent=2)}

## チャート判断の基準
- MA5 > MA25（乖離プラス）→ 短期上昇トレンド
- RSI > 70 → 買われすぎ（高値圏・利確売り圧力）
- RSI < 30 → 売られすぎ（リバウンド期待）
- 出来高比率 > 2倍 → 本格的な動きの可能性
- レンジ位置 > 80% → 高値圏（慎重に）、< 30% → 安値圏（リバウンド期待）
- ギャップアップ（寄付き価格が前日終値より+5%超）→ STRONG日でも除外推奨

## 決算リスクの判断（最優先）
- 決算リスク「高」（本日 or 3日以内）→ 必ず「見送り推奨」
- 決算リスク「中」（14日以内）→ 特段の理由がない限り「様子見」
- 決算リスク「低」→ 他の指標で判断

## お願いしたいこと
各銘柄について以下を日本語で答えてください：
1. 「買い推奨」「様子見」「見送り推奨」のどれか
2. チャート評価（MA・RSI・出来高・トレンドから読み取れること）
3. ニュースから読み取れる追い風・逆風
4. 総合コメント（2〜3文。地合いルールとの整合性にも言及すること）

回答はJSON形式で返してください：
{{
  "銘柄コード": {{
    "推奨": "買い推奨 or 様子見 or 見送り推奨",
    "チャート評価": "強気 or 中立 or 弱気",
    "ニュース評価": "追い風 or 逆風 or 中立 or 情報なし",
    "コメント": "..."
  }},
  ...
}}

注意: 株価の具体的な予測は行わず、テクニカル・ニュース・市場環境の観点からコメントしてください。"""

    # 混雑時(529)は最大3回リトライ
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < 2:
                print(f"  ⚠️ APIが混雑中。{10}秒後にリトライ（{attempt+1}/3）...")
                time.sleep(10)
            else:
                raise


# ══════════════════════════════════════════════
# 結果を表示
# ══════════════════════════════════════════════
def parse_and_display(ai_response, candidates_df, condition, news_data, chart_data=None, earnings_data=None):
    """AIの回答をパースして銘柄ごとに詳細表示"""

    print(f"\n{'='*60}")
    print(f"【AI総合レポート】Claude {MODEL}")
    print(f"  本日の地合い: {condition}  対象: {len(candidates_df)}件")
    print(f"{'='*60}")

    # JSONを抽出（```json ... ``` 形式にも対応）
    text = ai_response.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        ai_data = json.loads(text)
    except json.JSONDecodeError:
        print("\n  ⚠️ JSON解析に失敗しました。生の回答を表示します：")
        print(f"\n{ai_response}")
        return

    buy_count = caution_count = pass_count = 0

    for _, row in candidates_df.sort_values("score", ascending=False).iterrows():
        code     = str(row["code"])
        name     = str(row["name"])
        judgment = str(row["judgment"])
        score    = float(row["score"])
        strategy = str(row["strategy"])

        ai_info    = ai_data.get(code, {})
        ai_rec     = ai_info.get("推奨", "情報なし")
        ai_chart   = ai_info.get("チャート評価", "—")
        ai_news    = ai_info.get("ニュース評価", "—")
        ai_cmt     = ai_info.get("コメント", "—")

        jdg_icon   = {"BUY": "✅ 買い", "CAUTION": "⚠️ 注意"}.get(judgment, judgment)
        ai_icon    = {"買い推奨": "🟢", "様子見": "🟡", "見送り推奨": "🔴"}.get(ai_rec, "⚪")
        chart_icon = {"強気": "📊↑", "中立": "📊→", "弱気": "📊↓"}.get(ai_chart, "📊")
        news_icon  = {"追い風": "📰↑", "逆風": "📰↓", "中立": "📰→", "情報なし": "📰?"}.get(ai_news, "📰?")

        if ai_rec == "買い推奨":     buy_count     += 1
        elif ai_rec == "様子見":     caution_count += 1
        elif ai_rec == "見送り推奨": pass_count    += 1

        # candidates_log.csv に ai_recommendation を書き戻す
        try:
            log_df = pd.read_csv(CANDIDATES_LOG_CSV, encoding="utf-8-sig", dtype=str)
            mask   = (log_df["date"] == TODAY) & (log_df["code"] == code)
            if mask.any():
                if "ai_recommendation" not in log_df.columns:
                    log_df["ai_recommendation"] = ""
                log_df.loc[mask, "ai_recommendation"] = ai_rec
                log_df.to_csv(CANDIDATES_LOG_CSV, index=False, encoding="utf-8-sig")
        except Exception:
            pass

        # 銘柄ごとに区切り表示
        print(f"\n  ━━ {code} {name}（戦略{strategy} / スコア{score:.1f}）━━")
        print(f"  スキャン: {jdg_icon}  AI推奨: {ai_icon} {ai_rec}")
        print(f"  チャート: {chart_icon} {ai_chart}  ニュース: {news_icon} {ai_news}")

        # 決算リスク表示
        ei = (earnings_data or {}).get(code, {})
        if ei:
            risk        = ei.get("risk", "不明")
            risk_reason = ei.get("risk_reason", "")
            date_text   = ei.get("date_text", "")
            risk_icon   = {"高": "🚨", "中": "⚠️ ", "低": "✅", "不明": "❓"}.get(risk, "❓")
            print(f"  📅 決算: {risk_icon} リスク{risk}  {risk_reason}  ({date_text})")

        # チャート詳細（主要指標のみ表示）
        chart = (chart_data or {}).get(code, {})
        if isinstance(chart, dict) and "エラー" not in chart:
            print(f"  📊 MA5:{chart.get('MA5','—')}円 / MA25:{chart.get('MA25','—')}円"
                  f"  乖離:{chart.get('MA5_MA25乖離','—')}")
            print(f"     RSI:{chart.get('RSI14','—')}  出来高:{chart.get('出来高比率','—')}"
                  f"  レンジ:{chart.get('レンジ位置','—')}")
            print(f"     トレンド:{chart.get('5日トレンド','—')}")

        # ニュース表示
        news_list = news_data.get(code, [])
        if news_list:
            print(f"  📰 直近ニュース:")
            for n in news_list:
                date_str = f"（{n['date']}）" if n["date"] else ""
                print(f"     ・{n['title'][:60]}{date_str}")

        # AIコメント
        print(f"  🤖 {ai_cmt}")

    print(f"\n{'─'*60}")
    print(f"  【AIサマリー】 🟢買い推奨:{buy_count}件  🟡様子見:{caution_count}件  🔴見送り:{pass_count}件")
    print(f"  ⚠️  AIの判断は参考情報です。最終判断はご自身で行ってください。")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    print(f"=== 🤖 AI総合レポート（{TODAY}）===\n")

    # APIキー確認
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY が .env に設定されていません")
        return

    # 候補銘柄読み込み
    print("  候補銘柄を読み込み中...")
    candidates, condition = load_today_candidates()
    if candidates is None:
        return

    if candidates.empty:
        print(f"  本日のBUY/CAUTION候補がありません（地合い: {condition}）")
        print("  → AIによる判断は不要です")
        return

    market_info = load_today_market()
    print(f"  地合い: {condition}  対象銘柄: {len(candidates)}件\n")

    # ニュース・チャート・決算情報を銘柄ごとに取得
    news_data     = {}
    chart_data    = {}
    earnings_data = {}
    for _, row in candidates.iterrows():
        code = str(row["code"])
        name = str(row["name"])
        print(f"  📰 データ取得中: {code} {name}...")
        news_data[code]     = fetch_news(code, name)
        chart_data[code]    = fetch_chart_detail(code)     # Step 2: MA・RSI・出来高詳細
        earnings_data[code] = fetch_earnings_info(code)    # Step 3: 決算発表チェック
        time.sleep(0.5)  # 連続アクセスを避ける

    # 決算リスクが高い銘柄を事前に警告
    high_risk = [
        f"{code}({earnings_data[code]['risk_reason']})"
        for code in earnings_data
        if earnings_data[code].get("risk") == "高"
    ]
    if high_risk:
        print(f"\n  🚨 決算リスク高（見送り推奨）: {', '.join(high_risk)}")

    # Claude APIを呼び出す
    print(f"\n  Claude API ({MODEL}) に問い合わせ中...")
    try:
        ai_response = ask_claude(
            candidates, condition, market_info,
            news_data, chart_data, earnings_data
        )
    except anthropic.AuthenticationError:
        print("❌ APIキーが無効です。.env の ANTHROPIC_API_KEY を確認してください")
        return
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")
        return

    # 結果表示
    parse_and_display(ai_response, candidates, condition, news_data, chart_data, earnings_data)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
