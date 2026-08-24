# データ所在リスト

各種データがどこに（どのテーブル・どのログファイルに）記録されているかの索引。
「このデータはあるか？」と聞かれたら、まずここを確認する。
各項目は目的別に別々に作られてきたため、関連するものが複数箇所に分散していることに注意。

## 地合い（市場条件）関連

| データ | 保存先 | 記録タイミング | 蓄積期間 | 備考 |
|---|---|---|---|---|
| 朝の予測(CME/SGX先物ベース) | `morning_log`テーブル (`condition_forecast`, `condition_score`) | 朝＋9:03補正 | 2026-03頃〜 | scan_morning.py。A/AN/AS発注の実際の判断基準 |
| 16:45実測(AD比率+日経、その日の終値まで反映) | `market_log`テーブル (`condition`, `ad_ratio`, `nikkei_change`) | 16:45(1日1回) | 2026-03頃〜 | scan_daily.py。「一日を通した地合い」の代理指標として使える(2026-08-24に発見: 朝STRONG→終日STRONG維持は29日中1日=3.4%のみ) |
| 正午観測(AD比率、記録のみ・発注判断には未使用) | `logs/noon_ad_observation.log` | 12:40, 13:00 | 2026-08-14〜 | daytime.py（戦略D向け）。CSV形式: date,time,ad_ratio,nikkei_change,scanned_cond,condition |
| 15:00実測(戦略B発注前の地合い再判定) | `logs/condition_1500_observation.log` | 15:00頃 | 2026-08-24〜 | closing_watch.py。CSV形式: date,morning_cond,scanned_cond,ad_ratio,nikkei_change |
| 前日の地合い(Layer5リバウンド補正用) | `db.get_prev_day_condition_db()` | — | — | 2026-08-23修正: market_logではなくmorning_log.condition_forecastを参照(実測比較で朝の値の方が翌日回復予測精度が高いと確認済み) |

## 取引・損益関連

| データ | 保存先 | 備考 |
|---|---|---|
| 発注・決済の詳細(時刻・価格・損益・entry_change_pct等) | `positions`テーブル | 全戦略共通。buy_price/sell_priceは過去に精度問題があり2026-08-15 Fix②③で改善([[bug_buy_price_inaccuracy]]参照) |
| TP/SL検知タイミング(発注前) | `logs/exit_detection.log` | 2026-08-24追加。sell_time(実約定)と突き合わせてスリッページの原因(検知遅れ or 執行ラグ)を分析可能 |

## 銘柄選定・スコア関連

| データ | 保存先 | 備考 |
|---|---|---|
| 銘柄選定の判定理由・結果検証 | `candidates_log`テーブル | condition列は上書きされず当日の値を保持。result_pct/gradeは市場評価専用でどこからも参照されない(2026-08-23確認) |
| 前日16:45計算のscore/ratio/trend/accel | `scan_results`テーブル | scan_daily.pyが前日16:45に計算、翌朝scan_morning.pyが読み込んで候補判定に使用。`today_rise`は変数名に反して「前日の値動き」を意味する |
| 日足OHLCV | `daily_prices`テーブル | 2024-05-23〜（2年超）。当日分はscan_daily.pyの16:45実行時のみ書き込み(INSERT OR IGNOREのため上書きされない、16:45時点で既に確定値) |
| 分足 | `intraday_prices`テーブル | 2026-05-22〜。収集対象はscore上位30件+下位30件+戦略D200件に限定(scan_daily.py:803)。全候補を対象にすると誤解しやすいので注意 |

## 毎日上書き（翌日には消える）ログ

- `out/dt_live.txt`（daytime.py）
- `out/closing_watch_live.txt`（closing_watch.py）
- `out/market_watch_live.txt`（market_watch.py）

## 日付ごとに残るログ（削除処理なし、2026-08-24時点で15日分確認）

- `logs/run_daily_YYYYMMDD.log`
