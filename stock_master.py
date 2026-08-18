# -*- coding: utf-8 -*-
"""
stock_master.py - 銘柄コード→会社名のキャッシュ管理（2026-08-18追加）

closing_watch.py・scan_morning_strategy_an_candidates.py などが銘柄名解決の
たびにJ-Quantsの全銘柄マスタ(get_eq_master())をライブ取得すると、ネットワーク
遅延で処理が数分〜十数分ブレる（2026-08-18、戦略Bの引け前スキャンで実測、
15:05スキャン開始から発注まで約10分かかり15:20締切に迫った）。
銘柄名は頻繁に変わらないため、1日1回だけ取得してDB(stock_masterテーブル)に
キャッシュし、時間に余裕のない処理（closing_watch.py の15:00〜15:20枠など）は
このキャッシュを読むだけにする。
"""
import os

import db


def refresh(force=False, max_age_hours=20):
    """キャッシュが古い（or force=True）ならJ-Quantsから再取得してDBに保存する。
    戻り値: 更新後のcode→name辞書。取得失敗時は既存キャッシュ（あれば）をそのまま返す。"""
    if not force:
        age = db.get_stock_master_age_hours()
        if age is not None and age < max_age_hours:
            return db.get_stock_master()

    try:
        import jquantsapi
        from dotenv import load_dotenv
        load_dotenv()
        cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))
        master = cli.get_eq_master()
        master["code4"] = master["Code"].astype(str).str[:4]
        name_dict = dict(zip(master["code4"], master["CoName"]))
        db.save_stock_master(name_dict)
        print(f"  ✅ 銘柄名マスタキャッシュ更新: {len(name_dict)}件")
        return name_dict
    except Exception as e:
        print(f"  ⚠️  銘柄名マスタ取得失敗（既存キャッシュを使用）: {e}")
        return db.get_stock_master()


def get_names(warn_after_hours=48):
    """キャッシュから code→name 辞書を返す（通常はネットワーク呼び出しなし）。
    キャッシュが空の場合（初回起動時など）のみ、その場でrefresh()する。

    scan_morning.py の朝の自動更新が何日も静かに失敗し続けると、キャッシュが
    「空ではないが古い」状態のまま気づかれずに使われ続ける恐れがある
    （2026-08-18指摘）。ここでは処理時間を守るためネットワーク呼び出しは
    追加せず、代わりにwarn_after_hours超過時にログへ警告だけ出す。"""
    name_dict = db.get_stock_master()
    if not name_dict:
        return refresh(force=True)

    age = db.get_stock_master_age_hours()
    if age is not None and age > warn_after_hours:
        print(f"  ⚠️  銘柄名キャッシュが{age:.0f}時間前のまま更新されていません"
              f"（scan_morning.pyの朝の更新を確認してください）")
    return name_dict


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    result = refresh(force=True)
    print(f"銘柄名マスタ: {len(result)}件")
