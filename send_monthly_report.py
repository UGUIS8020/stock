# -*- coding: utf-8 -*-
"""
send_monthly_report.py - check_monthly.py の結果をメールで送る

毎月1日 cron から実行する想定。check_monthly.py を実行して出力を丸ごとメール本文にし、
hoero_world と同じ Amazon SES SMTP（.env の MAIL_SERVER/MAIL_PORT/MAIL_USERNAME/
MAIL_PASSWORD/MAIL_DEFAULT_SENDER、同一EC2で稼働実績のある経路）で送信する。
2026-09-11: 同時に用意した GMAIL_APP_PASSWORD 直SMTPは認証エラー（535、hoero_world
コード側でも実際には未使用の値だったため）で失敗したので、こちらに切替。
送信のみ・パラメータ変更なし。

使い方:
    python send_monthly_report.py            # 実行して送信
    python send_monthly_report.py --dry-run  # 送信せず本文を標準出力に表示
"""
import os
import sys
import smtplib
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from email.header import Header
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

JST = timezone(timedelta(hours=9))

MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_SENDER = os.getenv("MAIL_DEFAULT_SENDER") or MAIL_USERNAME
RECIPIENT = os.getenv("MONTHLY_REPORT_RECIPIENT")


def run_check_monthly():
    result = subprocess.run(
        [sys.executable, "check_monthly.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    body = result.stdout
    if result.returncode != 0:
        body += f"\n\n[⚠️ check_monthly.py が異常終了 exit={result.returncode}]\n{result.stderr}"
    return body


def send_mail(subject, body):
    if not MAIL_SERVER or not MAIL_USERNAME or not MAIL_PASSWORD:
        print("❌ MAIL_SERVER / MAIL_USERNAME / MAIL_PASSWORD が .env に無いため送信できません")
        return False
    if not RECIPIENT:
        print("❌ MONTHLY_REPORT_RECIPIENT が .env に無いため送信できません")
        return False
    # MAIL_DEFAULT_SENDER は "表示名 <addr>" 形式（hoero_world用の表示名が入っている）。
    # 2026-09-11: SMTPエンベロープ送信者(MAIL FROM)にこの文字列をそのまま渡すと、表示名が
    # RFC2047エンコードされて SES に「ユーザー名が無い」(554) と拒否された。エンベロープは
    # 必ずベアアドレスにし、ヘッダー側だけこのレポート用の表示名を付け直す。
    _, bare_addr = parseaddr(MAIL_SENDER)
    from_header = formataddr((str(Header("stock 月次レビュー", "utf-8")), bare_addr))

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = from_header
    msg["To"] = RECIPIENT
    with smtplib.SMTP(MAIL_SERVER, MAIL_PORT, timeout=20) as s:
        s.starttls()
        s.login(MAIL_USERNAME, MAIL_PASSWORD)
        s.sendmail(bare_addr, [RECIPIENT], msg.as_string())
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="送信せず本文を表示するだけ")
    args = ap.parse_args()

    today = datetime.now(JST)
    subject = f"stock 月次レビュー {today.strftime('%Y-%m')}"
    body = run_check_monthly()

    if args.dry_run:
        print(f"[DRY RUN] 件名: {subject}\n\n{body}")
        return

    ok = send_mail(subject, body)
    print("✅ 送信完了" if ok else "❌ 送信失敗（上記メッセージ参照）")


if __name__ == "__main__":
    main()
