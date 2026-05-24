import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.email_notifier import load_email_config, send_report_email


def build_test_email(email_config: dict):
    sender = (email_config.get("sender_email") or "").strip()
    recipient = (email_config.get("default_recipient") or "").strip()
    if not recipient:
        raise ValueError("email_config.json 中未配置 default_recipient")

    subject = "QQ SMTP 测试邮件"
    body = "\n".join([
        "这是一封用于验证 SMTP 配置的测试邮件。",
        f"发件人: {sender}",
        f"收件人: {recipient}",
        f"发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])
    return subject, body, recipient


def main() -> None:
    email_config = load_email_config()
    subject, body, recipient = build_test_email(email_config)
    send_report_email(subject, body, email_config, recipient)
    print(f"测试邮件已发送到: {recipient}")


if __name__ == "__main__":
    main()
