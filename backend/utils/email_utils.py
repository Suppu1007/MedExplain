# app/utils/email_utils.py

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import datetime

from app.core.config import EMAIL_SENDER, EMAIL_PASSWORD


# ============================================================
# INTERNAL EMAIL SENDER (PLAIN TEXT)
# ============================================================
def _send_email(to_email: str, subject: str, body: str):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("⚠ Email credentials missing — email skipped.")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"📧 Email sent → {to_email}")

    except Exception as e:
        print(f"❌ Email failed → {to_email}: {e}")


# ============================================================
# ACCOUNT CREATION EMAIL
# ============================================================
def send_account_created_email(
    to_email: str,
    username: str,
    temporary_password: str
):
    subject = "MedExplain – Your Account Has Been Created"

    body = f"""
Hello {username},

Welcome to MedExplain.

Your account has been successfully created on the MedExplain
Medical AI Assistance Platform.

Login Credentials:
------------------
Email: {to_email}
Temporary Password: {temporary_password}

For security reasons:
• Please change your password after first login
• Do not share your credentials

IMPORTANT:
MedExplain is an AI-assisted educational and decision-support system.
It does NOT provide medical diagnosis or treatment.

If you did not request this account, please contact support immediately.

Regards,
MedExplain Administration Team
"""
    _send_email(to_email, subject, body)


# ============================================================
# PASSWORD RESET EMAIL
# ============================================================
def send_reset_password_email(
    to_email: str,
    username: str,
    reset_link: str
):
    subject = "MedExplain – Password Reset Request"

    body = f"""
Hello {username},

We received a request to reset your MedExplain account password.

Reset Link (valid for 15 minutes):
---------------------------------
{reset_link}

If you did not initiate this request, please ignore this email.
Your account remains secure.

Regards,
MedExplain Security Team
"""
    _send_email(to_email, subject, body)


# ============================================================
# ROLE CHANGE / ACCESS UPDATE EMAIL
# ============================================================
def send_role_change_email(
    to_email: str,
    username: str,
    old_role: str,
    new_role: str,
    changed_by: str
):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    subject = "MedExplain – Access Role Update"

    body = f"""
Hello {username},

Your access permissions on MedExplain have been updated.

Previous Role: {old_role}
New Role: {new_role}
Updated By: {changed_by}
Date & Time: {timestamp}

If you believe this change is incorrect,
please contact the system administrator.

Regards,
MedExplain Administration Team
"""
    _send_email(to_email, subject, body)


# ============================================================
# MEDICAL REPORT PROCESSED NOTIFICATION
# ============================================================
def send_report_processed_email(
    to_email: str,
    username: str,
    report_id: str
):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    subject = "MedExplain – Medical Report Processed"

    body = f"""
Hello {username},

Your uploaded medical report has been successfully processed.

Report ID: {report_id}
Processed At: {timestamp}

You can now:
• View structured results
• Review explainability insights (SHAP / Grad-CAM)
• Ask questions using the Medical Assistant

IMPORTANT DISCLAIMER:
MedExplain provides AI-generated insights for educational
and decision-support purposes only.
It does NOT replace professional medical advice.

Regards,
MedExplain System Notification
"""
    _send_email(to_email, subject, body)
