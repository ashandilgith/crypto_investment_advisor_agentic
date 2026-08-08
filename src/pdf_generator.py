import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def create_pdf_report(target_coin: str, weights: dict, gpt_55_output: str, gpt_4_translation: str, history: list) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#1A365D")   # Deep Navy
    SECONDARY = colors.HexColor("#2B6CB0") # Slate Blue
    ACCENT = colors.HexColor("#D69E2E")    # Amber Gold
    BG_LIGHT = colors.HexColor("#F7FAFC")  # Off White
    TEXT_DARK = colors.HexColor("#2D3748") # Charcoal
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=22, textColor=PRIMARY, spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#718096"), spaceAfter=15)
    h2_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=14, textColor=SECONDARY, spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontSize=9.5, textColor=TEXT_DARK, leading=14, spaceAfter=8)
    bold_body = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')

    story = []

    # Header
    story.append(Paragraph("Locusts - Multi-Agent ", title_style))
    story.append(Paragraph(f"Autonomous Crypto Market Intelligence Report • Generated {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=15))

    # Executive Box
    rec_text = f"<b>RECOMMENDED ALLOCATION TODAY:</b> <font color='{PRIMARY.hexval()}'><b>{target_coin}</b></font>"
    rec_table = Table([[Paragraph(rec_text, ParagraphStyle('Rec', parent=body_style, fontSize=12, leading=16))]], colWidths=[540])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1.5, PRIMARY),
        ('PADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 15))

    # Applied Weights Table
    story.append(Paragraph("Model Feature Weight Distribution", h2_style))
    weight_data = [["Macro / Indicator Domain", "Applied Weight"]]
    for k, v in weights.items():
        clean_key = k.replace("_", " ").title()
        weight_data.append([clean_key, f"{v * 100:.1f}%"])

    w_table = Table(weight_data, colWidths=[380, 160])
    w_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('PADDING', (0,0), (-1,-1), 6)
    ]))
    story.append(w_table)
    story.append(Spacer(1, 15))

    # Plain-English Summary (GPT-4o-mini)
    story.append(Paragraph("1. Executive Summary (Plain English)", h2_style))
    for line in gpt_4_translation.split('\n'):
        if line.strip():
            story.append(Paragraph(line.strip().replace('#', ''), body_style))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceAfter=10))

    # Technical Core Analysis (GPT-5.5)
    story.append(Paragraph("2. Deep Technical Analysis (GPT-5.5 Engine)", h2_style))
    story.append(Paragraph(gpt_55_output.replace('\n', '<br/>'), body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()