# out/ フォルダ — ファイル説明

このフォルダは全スクリプトの出力先です。
S3（s3://shibuya8020/）と同期して複数端末で共有します。

---

## データベース（S3共有あり）

| ファイル | 生成元 | 更新頻度 | 内容 |
|---|---|---|---|
| `stock.db` | scan_daily.py | 毎営業日 16:30以降 | 全銘柄の日次株価・分足・スキャン結果を格納するSQLiteDB（約200MB） |

### stock.db のテーブル構成
- `daily_prices` — 全銘柄の日次OHLCV（2024-05-23〜、約180万件）
- `intraday_prices` — 分足データ（scan_daily.py が収集した銘柄のみ）
- `scan_results` — 戦略A候補（翌日の順張り候補リスト）
- `watchlist` — 戦略B候補（前日-4%以下の逆張り候補リスト）
- `positions` — 保有ポジション（買い後のTP/SL管理用）

---

## 最適化データ（S3共有あり）

optimize_b.py / evolve_b.py を実行すると生成されます。
**他端末への反映**: `python download_db.py` を実行してください。

| ファイル | 生成元 | 更新頻度 | 内容 |
|---|---|---|---|
| `opt_b_trades.csv` | optimize_b.py | 月1回 | 戦略B全トレードデータ（-3%以下の銘柄×翌日リターン） |
| `opt_b_stage1.csv` | optimize_b.py | 月1回 | グリッドサーチ結果（drop帯×RB×gap×TP/SLの全組み合わせ） |
| `opt_b_stage2.csv` | optimize_b.py | 月1回 | ウォークフォワード検証結果（前60%学習→後40%検証） |
| `opt_b_report.txt` | optimize_b.py | 月1回 | 最適化サマリーレポート（推奨パラメータ一覧） |
| `evolve_b_best.csv` | evolve_b.py | 月1回 | GA最良戦略一覧（In/Out両方プラスの戦略） |
| `evolve_b_history.csv` | evolve_b.py | 月1回 | GA世代別Sharpe推移（進化の記録） |
| `evolve_b_report.txt` | evolve_b.py | 月1回 | GAサマリーレポート |

### opt_b_trades.csv の列説明
| 列 | 内容 |
|---|---|
| code | 銘柄コード |
| scan_date | スキャン日（候補として検出された日） |
| trade_date | 翌営業日（実際に買う日） |
| scan_cond | スキャン日の地合い（STRONG/NORMAL/WEAK/PANIC） |
| trade_cond | 翌日の地合い |
| today_rise | スキャン日の株価変化率（%）← -4%以下が対象 |
| rb_score | リバウンドスコア（高いほど反発しやすい銘柄） |
| gap_pct | 翌朝の窓開け率（始値 / 前日終値 - 1） |
| ret_oc | 翌日の始値→終値リターン（%） |
| ret_max | 翌日の始値→高値リターン（%） |
| ret_low | 翌日の始値→安値リターン（%） |
| lower_shadow_ratio | 下ヒゲ比率（長いほど買い支えあり） |
| vol_decay_3d | 3日間出来高減少フラグ（1=売り枯れ） |
| consec_drop | 連続下落日数 |

---

## 日次ログ（ローカルのみ）

| ファイル | 生成元 | 内容 |
|---|---|---|
| `candidates_log.csv` | scan_morning.py | 朝判定ログ（BUY/CAUTION/PASSと理由） |
| `backtest_log.csv` | scan_daily.py | 戦略A検証ログ（翌日実績） |
| `market_log.csv` | scan_daily.py | 地合い日次ログ（STRONG/NORMAL/WEAK/PANIC） |
| `observe_log.csv` | market_watch.py | OBSERVEモード記録（WEAK+CAUTION銘柄の価格推移） |
| `positions.csv` | tachibana_order.py | 保有ポジション一覧（buy→TP/SL管理） |

---

## S3 同期方法

```bash
# 最新データをダウンロード（他端末で作業開始時）
python download_db.py

# DBのみ高速ダウンロード（朝スキャン前 ← scan_morning.py が自動実行）
python download_db.py --db-only
```

S3パス一覧:
- `s3://shibuya8020/stock-db/stock.db` — 日次DB
- `s3://shibuya8020/stock-optimize/` — 最適化結果
