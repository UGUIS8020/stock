---
name: コード改善履歴
description: scan_morning.py / scan_daily.py への主な修正内容と理由
type: project
---

## scan_morning.py 修正履歴

### 地合い判定（judge_market_score）
- STRONGにus_up_count>=2を追加（過剰判定防止）
- PANICにnk_chg<=-1.0を追加
- WEAK閾値: score>=3 → score>=2

### 戦略A判定（judge_entry_a）
- score>=9.0の一律CAUTION廃止
- ratio>=12.0 かつ STRONGの場合のみBUY（その他はCAUTION）
- スコア9.0〜が利確率46%（最高）をコメントに記録
- スコア8.0〜8.5が利確率25%（最低）を観察コメントに記録（閾値変更は1ヶ月後に再評価）

### 戦略B判定（judge_entry_b）
- **RBスコア7以上 → PASS**（落ちナイフ：RB7が0%/17件、RB8が0%/12件）
- **RBスコア3〜6 × WEAK日 → BUY**（avg +2.51%、プラス率66%）
- RBスコア3〜6 × NORMAL日 → CAUTION（avg +0.48%）
- RSI変数を削除（使用されていなかった）
- 各判定にデータ根拠をコメントとして明記

### その他
- NIKKEI_TICKER二重定義を削除（37行目）
- candidates_log.csvにconditionカラムを追加

---

## scan_daily.py 修正履歴

### 地合い判定（judge_market_condition）
- **STRONGを追加**（日経+0.5%以上 かつ 騰落比60%以上）
- 閾値定数を追加: STRONG_NIKKEI_THRESHOLD=0.5、STRONG_AD_THRESHOLD=0.60

### print_market_condition
- STRONGのメッセージを追加（🚀アイコン、戦略A積極推奨）

### show_market_log_summary
- 集計対象にSTRONGを追加

### インデントバグ修正
- 843行目の余分な2スペースインデントを修正

---

## 新規作成ファイル
- **analyze_multiday.py**: 複数日保有バックテスト分析スクリプト
  - backtest_log.csv × キャッシュデータで1〜5日後のリターンを追跡
  - +3%利確/損切り-5%の複数日シミュレーション
  - 出力: out/multiday_log.csv
  - 実行: `python analyze_multiday.py`

## 削除済みファイル（不要と判断）
- analyze_b.py
- Analyze_results.py
- backtest_a_v2.py
- backtest_scanner_today.py
- backtest_scanner_tomorrow.py
- Backtest_score_band.py（verify_results.pyに統合）
- extend_cache.py
- fetch_historical.py
- migrate_csv.py
- pattern_analysis.py
- run_daily.bat
- data_j.xls
- data/フォルダ
- out/backtest_result.csv、backtest_results_v6.csv、backtest_results_v7.csv
