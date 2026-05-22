---
name: 株スキャンシステム概要
description: scan_morning.py / scan_daily.py の役割・ワークフロー・改善履歴
type: project
---

## システム構成

| ファイル | 実行時刻 | 役割 |
|---|---|---|
| scan_daily.py | 16:30以降（引け後） | 全銘柄スコアリング → 翌日候補を選出・前日結果を検証 |
| scan_morning.py | 9:00前（寄り付き前） | 海外市場から地合い予測 → 候補に BUY/CAUTION/PASS 判定 |
| verify_results.py | 15:30以降 | 過去の判定を事後検証（yfinance使用） |

## ワークフロー
```
前日16:30 scan_daily.py → 候補リスト生成
当日9:00前 scan_morning.py → BUY/CAUTION/PASS 判定
当日引け後 scan_daily.py → 結果検証 + 翌日候補
```

## 地合い判定（4段階）
- **STRONG**: スコア7以上 + 米3指数のうち2つ以上上昇 + 騰落比60%以上 + 日経+0.5%以上
- **NORMAL**: 上記以外の通常相場
- **WEAK**: 日経-1%以下 または 騰落比35%以下
- **PANIC**: 日経-2%以下 または 騰落比20%以下

## 主な出力ファイル
- `out/candidates_log.csv` — 朝判定ログ
- `out/backtest_log.csv` — 戦略A検証結果
- `out/watchlist.csv` — 戦略B監視リスト
- `out/market_log.csv` — 地合い日次ログ
- `out/scan_results.csv` — 戦略Aスコアリング結果
