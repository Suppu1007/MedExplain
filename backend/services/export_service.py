from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import re

class ExportService:
    @staticmethod
    def generate_medical_pdf(data: dict, user_email: str) -> bytes:
        """
        Generates a premium clinical PDF report for Lab Analysis.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        story = []
        styles = getSampleStyleSheet()

        # --- Custom Styles ---
        primary_color = colors.HexColor("#0d6efd") # Professional Blue
        dark_bg = colors.HexColor("#1a1a1a")
        light_bg = colors.HexColor("#f8f9fa")

        # Reuse or add common styles
        styles.add(ParagraphStyle(name='BannerTitle', parent=styles['Heading1'], fontSize=22, textColor=colors.white, alignment=TA_CENTER, spaceAfter=10))
        styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading3'], fontSize=12, textColor=primary_color, spaceBefore=15, spaceAfter=8))
        styles.add(ParagraphStyle(name='MetricsLabel', parent=styles['BodyText'], fontSize=9, textColor=colors.gray, alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='MetricsValue', parent=styles['BodyText'], fontSize=13, textColor=dark_bg, alignment=TA_CENTER, fontName="Helvetica-Bold"))
        styles.add(ParagraphStyle(name='NarrativeBody', parent=styles['BodyText'], fontSize=10, leading=15, spaceAfter=10))
        styles.add(ParagraphStyle(name='TableText', parent=styles['BodyText'], fontSize=9))
        styles.add(ParagraphStyle(name='TableTextBold', parent=styles['TableText'], fontName="Helvetica-Bold"))

        # --- 1. Header Banner ---
        report_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        header_content = [
            [Paragraph("MEDIEXPLAIN INTELLIGENCE", styles['BannerTitle'])],
            [Paragraph(f"LABORATORY ANALYSIS REPORT • {report_date}", 
                       ParagraphStyle(name='BannerSub', parent=styles['Normal'], textColor=colors.whitesmoke, alignment=TA_CENTER))]
        ]
        banner_table = Table(header_content, colWidths=[6.5*inch])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), dark_bg),
            ('TOPPADDING', (0,0), (-1,-1), 15),
            ('BOTTOMPADDING', (0,0), (-1,-1), 15),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 15))

        # --- 1b. Patient Information ---
        story.append(Paragraph(f"<b>PATIENT ID:</b> {user_email}", styles['Normal']))
        story.append(Spacer(1, 10))

        # --- 2. Key Metrics Grid ---
        global_conf = data.get('global_confidence') or data.get('confidence') or "85.0%"
        structured = data.get('structured', {})
        triage_data = structured.get('triage', {})
        urgency = triage_data.get('urgency', 'NORMAL').upper()
        specialist = triage_data.get('specialist', 'Primary Care Physician')

        triage_color = "#198754" # Success Green
        if urgency in ["URGENT", "HIGH"]: triage_color = "#fd7e14" # Warning Orange
        if urgency == "EMERGENCY": triage_color = "#dc3545" # Danger Red

        metrics_data = [
            [Paragraph("ANALYSIS CONFIDENCE", styles['MetricsLabel']), 
             Paragraph("TRIAGE STATUS", styles['MetricsLabel']),
             Paragraph("SPECIALIST CONSULT", styles['MetricsLabel'])],
            
            [Paragraph(global_conf, styles['MetricsValue']), 
             Paragraph(f"<font color='{triage_color}'>{urgency}</font>", styles['MetricsValue']),
             Paragraph(specialist, styles['MetricsValue'])]
        ]

        metrics_table = Table(metrics_data, colWidths=[2.1*inch, 2.1*inch, 2.3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), light_bg),
            ('GRID', (0,0), (-1,-1), 0.5, colors.Color(0.9, 0.9, 0.9)),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 15))

        # --- 3. CLINICAL NARRATIVE ---
        narrative_html = data.get('narrative') or data.get('human_narrative') or ''
        if narrative_html:
            story.append(Paragraph("Expert Clinical Narrative", styles['SectionHeader']))
            
            # Standardize and clean narrative
            clean_html = narrative_html.replace('<div class="clinical-narrative">', '').replace('</div>', '')
            clean_html = clean_html.replace('<br>', '<br/>').replace('<br >', '<br/>')
            
            # Extract sections: <p><b>Header</b><br>Content</p>
            sections = re.findall(r'<p>(.*?)</p>', clean_html, re.DOTALL)
            
            if not sections: # Fallback for flat text
                story.append(Paragraph(clean_html.replace('<p>', '').replace('</p>', '\n'), styles['NarrativeBody']))
            else:
                for section in sections:
                    header_match = re.search(r'<b>(.*?)</b>', section)
                    if header_match:
                        h_text = header_match.group(1).strip()
                        body_text = re.sub(r'<b>.*?</b>', '', section).strip()
                        body_text = body_text.lstrip(':').strip()
                        body_text = body_text.replace('<br/>', '\n').strip()
                        
                        story.append(Paragraph(h_text.upper(), ParagraphStyle(name='NarrHeader', parent=styles['Normal'], fontSize=9, fontName="Helvetica-Bold", textColor=primary_color, spaceBefore=5)))
                        story.append(Paragraph(body_text, styles['NarrativeBody']))
                    else:
                        story.append(Paragraph(section, styles['NarrativeBody']))

        story.append(Spacer(1, 10))

        # --- 4. STRUCTURED BIOMARKER ANALYSIS ---
        markers = structured.get('markers', [])
        if markers:
            story.append(Paragraph("Detailed Biomarker Analysis", styles['SectionHeader']))
            
            table_data = [
                [Paragraph("BIOMARKER", styles['TableTextBold']), 
                 Paragraph("RESULT", styles['TableTextBold']), 
                 Paragraph("CONF.", styles['TableTextBold']), 
                 Paragraph("STATUS", styles['TableTextBold']), 
                 Paragraph("REFERENCE", styles['TableTextBold'])]
            ]
            
            for m in markers:
                status_raw = m.get('status', 'NORMAL').upper()
                status_color = "#198754" # Normal
                if status_raw == 'CRITICAL': status_color = "#dc3545"
                elif status_raw in ['ABNORMAL', 'WARNING', 'HIGH', 'LOW']: status_color = "#fd7e14"
                
                row = [
                    Paragraph(m.get('name', 'N/A'), styles['TableText']),
                    Paragraph(f"<b>{m.get('value', '-')}</b> <font size=7>{m.get('unit', '')}</font>", styles['TableText']),
                    Paragraph(m.get('confidence', '95%'), styles['TableText']),
                    Paragraph(f"<font color='{status_color}'><b>{status_raw}</b></font>", styles['TableText']),
                    Paragraph(f"{m.get('ref_min', '?')} - {m.get('ref_max', '?')}", styles['TableText'])
                ]
                table_data.append(row)

            table = Table(table_data, colWidths=[2.2*inch, 1.2*inch, 0.8*inch, 1.1*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.95, 0.95, 0.95)),
                ('TEXTCOLOR', (0, 0), (-1, 0), dark_bg),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LINEBELOW', (0, 0), (-1, 0), 1, colors.Color(0.8, 0.8, 0.8)),
                ('GRID', (0, 1), (-1, -1), 0.2, colors.Color(0.9, 0.9, 0.9)),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 15))

        # --- DISCLAIMER ---
        story.append(Spacer(1, 0.5 * inch))
        normal_style = styles['Normal']
        disclaimer_text = (
            "<b>DISCLAIMER:</b> This report is generated by an AI system for educational and "
            "informational purposes only. It is NOT a medical diagnosis. "
            "Please consult a qualified healthcare provider for interpretation and medical advice."
        )
        story.append(Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=normal_style, fontSize=8, textColor=colors.gray, alignment=TA_CENTER)))

        # Build PDF
        try:
            doc.build(story)
            return buffer.getvalue()
        except Exception as e:
            print(f"ReportLab Build Error: {e}")
            return None