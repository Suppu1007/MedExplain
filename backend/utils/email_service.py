from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import datetime

from core.config import EMAIL_SENDER, EMAIL_PASSWORD


# =====================================================
# INTERNAL EMAIL SENDER (PLAIN TEXT)
# =====================================================
def _send_email(to_email: str, subject: str, body: str):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("Email credentials not configured. Skipping email.")
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

        print(f"Email sent -> {to_email}")

    except Exception as e:
        print(f"Email send failed -> {to_email}: {e}")


# =====================================================
# MEDICAL REPORT UPLOAD CONFIRMATION
# =====================================================
def send_report_uploaded_email(report: dict, uploaded_by: str):
    report_id = report.get("report_id", "Unknown")
    patient_name = report.get("patient_name", "User")
    report_type = report.get("report_type", "Medical Report")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    body = f"""
Medical Report Uploaded Successfully

Report ID: {report_id}
Report Type: {report_type}
Patient Name: {patient_name}
Uploaded By: {uploaded_by}
Uploaded At: {ts}

Your report is now queued for:
• OCR & parsing
• AI-based analysis
• Explainability generation (SHAP / Grad-CAM)

IMPORTANT:
MedExplain provides AI-assisted insights only.
It does NOT provide medical diagnosis or treatment.

Regards,
MedExplain System
"""

    if report.get("user_email"):
        _send_email(
            report["user_email"],
            f"MedExplain – Report {report_id} Uploaded",
            body
        )


# =====================================================
# MEDICAL REPORT STATUS UPDATE
# =====================================================
def notify_report_status_change(
    report: dict,
    old_status: str,
    new_status: str,
    analysis_metadata: dict | None = None
):
    report_id = report.get("report_id", "Unknown")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    metadata_text = ""
    if analysis_metadata:
        metadata_text = "\nAnalysis Summary:\n"
        for key, val in analysis_metadata.items():
            metadata_text += f" - {key}: {val}\n"

    body = f"""
Medical Report Status Update

Report ID: {report_id}
Previous Status: {old_status}
Current Status: {new_status}
Updated At: {ts}
{metadata_text}

You can now view:
• Structured results
• Risk indicators
• AI explainability insights

DISCLAIMER:
All outputs are for educational and
clinical decision-support purposes only.

Regards,
MedExplain System
"""

    subject = f"MedExplain – Report {report_id} Status: {new_status}"

    if report.get("user_email"):
        _send_email(report["user_email"], subject, body)
