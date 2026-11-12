import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Preformatted, Image, PageBreak, KeepTogether)

# ---------- Yahan apni details check kar lo ----------
NAME = 'Shaikh Naushad'
ROLL = '48'
DIVISION = 'A'
CLASS = 'BSc FYDS'
TITLE = 'Facial Emotion Recognition'
SUBTITLE = 'Computer Vision - Unit 2 Mini Project'
ACCURACY = '64.78%'
# -----------------------------------------------------

SUB = 'submission'
CODE_FILES = [
    ('train_model.py', 'Model training (Google Colab, T4 GPU)'),
    ('app.py', 'Live webcam emotion recognition app'),
    ('report.py', 'Session timeline graph and summary'),
]
SHOTS = [
    ('emotion_20261010_174739.png', 'Neutral'),
    ('emotion_20261010_174757.png', 'Happy'),
    ('emotion_20261010_175600.png', 'Surprise'),
    ('emotion_20261010_175653.png', 'Angry'),
]
GRAPH = ('session_20261010_175455_report.png',
         'Emotion timeline and time share by dominant emotion (one live session)')

INFO = [
    f'Name: {NAME}',
    f'Class: {CLASS}    Division: {DIVISION}    Roll No: {ROLL}',
    'Dataset: FER2013 (7 emotions, 48x48 grayscale faces)',
    f'Model: custom CNN trained from scratch, test accuracy {ACCURACY}',
]


def read_code(name):
    path = os.path.join(SUB, 'code', name)
    with open(path, encoding='utf-8') as f:
        return f.read().replace('\t', '    ').rstrip()


# Pehle check: saari files maujood hain ya nahi
needed = [os.path.join(SUB, 'code', n) for n, _ in CODE_FILES]
needed += [os.path.join(SUB, 'output', n) for n, _ in SHOTS + [GRAPH]]
missing = [p for p in needed if not os.path.exists(p)]
if missing:
    raise SystemExit('ERROR: ye files nahi mili:\n' + '\n'.join(missing))


# ---------------- WORD ----------------
def build_docx(path):
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.7)
    sec.top_margin = sec.bottom_margin = Inches(0.7)

    doc.add_heading(TITLE, 0)
    doc.add_paragraph(SUBTITLE)
    for line in INFO:
        doc.add_paragraph(line)

    doc.add_heading('1. Code', 1)
    for name, desc in CODE_FILES:
        doc.add_heading(f'{name} - {desc}', 2)
        p = doc.add_paragraph()
        run = p.add_run(read_code(name))
        run.font.name = 'Consolas'
        run.font.size = Pt(7.5)
        p.paragraph_format.space_after = Pt(8)

    doc.add_page_break()
    doc.add_heading('2. Output', 1)
    for fname, label in SHOTS + [GRAPH]:
        cap = doc.add_paragraph()
        cap.add_run(label).bold = True
        doc.add_picture(os.path.join(SUB, 'output', fname), width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.save(path)


# ---------------- PDF ----------------
def pdf_image(path, max_w=480):
    w, h = ImageReader(path).getSize()
    scale = max_w / w
    return Image(path, width=max_w, height=h * scale)


def build_pdf(path):
    styles = getSampleStyleSheet()
    code_style = ParagraphStyle('code', fontName='Courier', fontSize=6.8, leading=8.2)
    cap_style = ParagraphStyle('cap', parent=styles['Heading4'], alignment=TA_CENTER)

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=36, rightMargin=36,
                            topMargin=36, bottomMargin=36)
    story = [Paragraph(TITLE, styles['Title']),
             Paragraph(SUBTITLE, styles['Heading3'])]
    for line in INFO:
        story.append(Paragraph(line, styles['Normal']))
    story.append(Spacer(1, 10))

    story.append(Paragraph('1. Code', styles['Heading2']))
    for name, desc in CODE_FILES:
        story.append(Paragraph(f'{name} - {desc}', styles['Heading4']))
        story.append(Preformatted(read_code(name), code_style))
        story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph('2. Output', styles['Heading2']))
    for fname, label in SHOTS + [GRAPH]:
        story.append(KeepTogether([
            Paragraph(label, cap_style),
            pdf_image(os.path.join(SUB, 'output', fname)),
            Spacer(1, 12),
        ]))
    doc.build(story)


docx_path = os.path.join(SUB, 'CV_Unit2_Emotion_Recognition.docx')
pdf_path = os.path.join(SUB, 'CV_Unit2_Emotion_Recognition.pdf')
build_docx(docx_path)
build_pdf(pdf_path)
print('Word saved:', docx_path)
print('PDF saved :', pdf_path)