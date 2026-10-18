"""
Generate News Bias Detector Interview & Architecture Walkthrough PDF
Uses ReportLab to create a clean, comprehensive, professional document.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_OUTPUT = "News_Bias_Detector_Walkthrough.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages after cover)
        if self._pageNumber > 1:
            self.drawString(54, 750, "News Bias Detector · Architecture & Interview Walkthrough")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)
            
        # Footer (all pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 35, page_text)
        self.drawString(54, 35, "Confidential · Prepared for Technical & System Design Interviews")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 47, 612 - 54, 47)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Color Palette
    PRIMARY = colors.HexColor("#1e293b")     # Slate 800
    SECONDARY = colors.HexColor("#0f766e")   # Teal 700
    ACCENT = colors.HexColor("#2563eb")      # Blue 600
    DARK_TEXT = colors.HexColor("#0f172a")   # Slate 900
    MUTED_TEXT = colors.HexColor("#475569")  # Slate 600
    BG_LIGHT = colors.HexColor("#f8fafc")    # Slate 50
    BORDER_COL = colors.HexColor("#cbd5e1")  # Slate 300
    CODE_BG = colors.HexColor("#f1f5f9")     # Slate 100

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=body_style,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e3a8a"),
    )

    qa_q_style = ParagraphStyle(
        'QAQuestion',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    qa_a_style = ParagraphStyle(
        'QAAnswer',
        parent=body_style,
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
        spaceAfter=8
    )

    story = []

    # ─────────────────────────────────────────────────────────────
    # COVER / HEADER
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("News Bias & Framing Detector", title_style))
    story.append(Paragraph("System Architecture, ML/NLP Pipeline & Technical Interview Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceAfter=12))

    # Executive Summary Card
    summary_html = (
        "<b>Executive Summary:</b> News Bias Detector is a portfolio-grade, automated data pipeline "
        "and analytical dashboard that ingests news articles from 15+ global and Indian outlets, "
        "classifies political bias, sentiment, and entity portrayal using Google Gemini LLM, "
        "clusters articles reporting on the same real-world event using Sentence Embeddings & DBSCAN, "
        "and visualizes coverage gaps and narrative polarization on a live Streamlit dashboard."
    )
    t = Table([[Paragraph(summary_html, callout_style)]], colWidths=[504])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#bfdbfe")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ─────────────────────────────────────────────────────────────
    # SECTION 1: SYSTEM ARCHITECTURE & 4-LAYER PIPELINE
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("1. System Architecture & Pipeline Breakdown", h1_style))
    story.append(Paragraph(
        "The project follows a decoupled, sequential ETL (Extract, Transform, Load) architecture divided into 4 modular layers:",
        body_style
    ))

    pipeline_table_data = [
        [Paragraph("<b>Layer</b>", body_style), Paragraph("<b>Component & Technology</b>", body_style), Paragraph("<b>Function & Core Output</b>", body_style)],
        [
            Paragraph("<b>Layer 1: Ingestion</b>", body_style),
            Paragraph("<code>scraper.py</code><br/>feedparser · newspaper4k", body_style),
            Paragraph("Polls RSS feeds from 15+ outlets. Extracts clean article text, removes paywalled/short stubs (&lt;100 chars), deduplicates by URL, enforces a 10-day rolling window. Output: <code>articles.json</code>.", body_style)
        ],
        [
            Paragraph("<b>Layer 2: Analysis</b>", body_style),
            Paragraph("<code>analyzer.py</code><br/>Google Gemini 3.8 Flash", body_style),
            Paragraph("Zero-shot LLM reasoning with strict JSON Schema. Extracts: <b>Bias Label</b> (Left→Right), <b>Sentiment</b> (-1.0 to +1.0), <b>Framing</b> (Alarmist, Neutral, etc.), and <b>Entity Portrayals</b>. Incremental processing saves API calls. Output: <code>analyzed.json</code>.", body_style)
        ],
        [
            Paragraph("<b>Layer 3: Clustering</b>", body_style),
            Paragraph("<code>clusterer.py</code><br/>all-MiniLM-L6-v2 · DBSCAN", body_style),
            Paragraph("Embeds Title + Body text into dense 384-d vectors. Uses cosine distance matrix + DBSCAN to identify cross-outlet coverage of identical events. Generates headline-overlap cluster labels. Output: <code>clustered.json</code>.", body_style)
        ],
        [
            Paragraph("<b>Layer 4: Dashboard</b>", body_style),
            Paragraph("<code>app.py</code><br/>Streamlit · Plotly · Pandas", body_style),
            Paragraph("Interactive visualization with 4 analytical views: <b>Coverage Gaps Matrix</b>, <b>Entity Sentiment Heatmap</b>, <b>Bias Radar</b>, and <b>Article Browser</b>. Non-blocking UI.", body_style)
        ]
    ]

    ptable = Table(pipeline_table_data, colWidths=[100, 140, 264])
    ptable.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COL),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(ptable)
    story.append(Spacer(1, 14))

    # ─────────────────────────────────────────────────────────────
    # SECTION 2: CLOUD AUTOMATION & PRODUCTION DEPLOYMENT
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("2. Automation & Cloud Infrastructure", h1_style))
    story.append(Paragraph(
        "A key design highlight is the <b>complete decoupling of heavy compute from web serving</b>:",
        body_style
    ))
    story.append(Paragraph("• <b>Automated 3-Day ETL via GitHub Actions</b>: A scheduled cron workflow (<code>0 0 */3 * *</code>) runs in GitHub's cloud environment. It executes scraping, LLM classification (with rate-limit throttling of 14 RPM to respect Google's free tier), and event clustering. It automatically commits updated data files back to the repository.", bullet_style))
    story.append(Paragraph("• <b>Lightweight Web Hosting on Render</b>: Render hosts only the Streamlit visualization layer. Because Render free tier has a 512MB RAM cap, PyTorch and Sentence-Transformers are excluded from Render's build. Render builds in &lt;20 seconds using a lean <code>requirements.txt</code>.", bullet_style))
    story.append(Paragraph("• <b>Continuous Deployment (CD)</b>: Whenever GitHub Actions commits fresh data to <code>main</code>, Render automatically detects the push and redeploys the dashboard without any manual intervention.", bullet_style))
    story.append(Spacer(1, 14))

    # ─────────────────────────────────────────────────────────────
    # SECTION 3: DEEP DIVE INTO ALGORITHMIC CHOICES
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Technical Design Decisions: Why These Tools?", h1_style))
    
    decisions = [
        ("Why DBSCAN over K-Means for Event Clustering?", 
         "K-Means requires specifying <i>k</i> (the number of clusters) upfront, which is impossible in real-world news where the number of breaking stories fluctuates hourly. DBSCAN is density-based, discovers arbitrary numbers of clusters automatically, and marks isolated/unique stories as noise points (outliers) instead of forcing them into unrelated clusters."),
        
        ("Why Dense Sentence-Transformers over TF-IDF / BM25?",
         "News outlets use radically different vocabulary to describe the same event (e.g., <i>'President signed executive decree'</i> vs <i>'White House enacts controversial policy order'</i>). TF-IDF fails because word overlap is zero. Sentence-Transformers (<code>all-MiniLM-L6-v2</code>) captures latent semantic meaning in 384-dimensional dense space."),
         
        ("Why Structured JSON Output with Gemini 3.8 Flash?",
         "Gemini 3.8 Flash provides sub-second inference, low cost, and native JSON schema enforcement via <code>response_json_schema</code>. This eliminates common LLM hallucination and parsing issues, guaranteeing strict typing for downstream Pandas pivoting."),
         
        ("Why Incremental Processing & URL Hashing?",
         "Analyzing 200+ articles from scratch on every run would burn API rate limits. <code>analyzer.py</code> loads existing <code>analyzed.json</code>, cross-references URLs, and only sends net-new articles to the LLM, preserving rate limits and execution speed.")
    ]

    for title, desc in decisions:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        story.append(Paragraph(desc, body_style))

    story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────
    # SECTION 4: INTERVIEW CHEATSHEET — TOP QUESTIONS & ANSWERS
    # ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Technical Interview Cheatsheet (Questions &amp; Answers)", h1_style))
    story.append(Paragraph("Here are the top questions an interviewer will ask about this project, along with optimal, high-scoring answers:", body_style))
    story.append(Spacer(1, 6))

    qa_list = [
        (
            "Q1: Walk me through the high-level architecture of your project.",
            "<b>Answer:</b> 'The project is an automated end-to-end news intelligence system. It operates in 4 stages: First, a scraping engine collects RSS feeds from 15 global outlets using Feedparser and Newspaper4k. Second, an LLM pipeline uses Google Gemini 3.8 Flash with strict JSON schema to evaluate political bias, sentiment, framing, and entity portrayal. Third, an NLP clustering pipeline embeds headlines with MiniLM sentence transformers and clusters identical events using DBSCAN. Finally, an interactive Streamlit app visualizes media bias polarization, entity heatmaps, and coverage gaps. The entire pipeline is automated using GitHub Actions on a 3-day cron and deployed on Render.'"
        ),
        (
            "Q2: How do you prevent the LLM from producing invalid or hallucinated data?",
            "<b>Answer:</b> 'I used Google Gemini's native structured outputs feature with a strict JSON Schema definition. The schema enforces enum constraints on categorical fields like <code>bias_label</code> (Left, Center, Right), restricts <code>sentiment_score</code> to a float between -1.0 and 1.0, and bounds entity portrayals. Additionally, the code includes a fallback JSON parser with exponential retry logic to handle transient API hiccups.'"
        ),
        (
            "Q3: How did you handle API rate limits during batch processing?",
            "<b>Answer:</b> 'Google Gemini's free tier has a 15 Requests-Per-Minute (RPM) limit. To avoid 429 Resource Exhausted errors, I implemented deterministic client-side pacing with a 4.2-second sleep interval (max ~14 RPM), coupled with exponential backoff on HTTP 429 responses. Furthermore, the pipeline is incremental: it checks already-analyzed URLs in our persistent cache and processes batches of 30 articles per run.'"
        ),
        (
            "Q4: Why did you pick DBSCAN for clustering and how did you tune it?",
            "<b>Answer:</b> 'In news data, we do not know the number of breaking news stories in advance, which rules out K-Means. DBSCAN is ideal because it finds clusters based on spatial density and naturally isolates outlier articles as noise (-1). I precomputed cosine distances between L2-normalized sentence embeddings, set <code>min_samples=2</code> so an event requires at least two outlets to form a cluster, and tuned <code>eps=0.35–0.40</code> to balance cluster tightness with recall.'"
        ),
        (
            "Q5: How does the application maintain freshness in production?",
            "<b>Answer:</b> 'We use an automated CI/CD schedule in GitHub Actions that triggers every 3 days. It executes the ETL pipeline, computes new clusters, and commits the updated JSON artifacts back to Git with <code>[skip ci]</code>. Render detects the repository commit and triggers a zero-downtime redeployment of the web UI. On the dashboard itself, we maintain a rolling 10-day retention window to prune stale news.'"
        ),
        (
            "Q6: How did you address cloud memory constraints on Render's free tier?",
            "<b>Answer:</b> 'Render free tier has a 512MB RAM ceiling. Heavy ML libraries like PyTorch and HuggingFace Sentence-Transformers easily exceed 1GB during initialization. I decoupled compute from presentation: all model inference and clustering run on GitHub Actions runners (which provide 7GB RAM for free), while Render only installs lightweight visualization libraries (<code>streamlit</code>, <code>plotly</code>, <code>pandas</code>), reducing build time to under 20 seconds and memory to ~80MB.'"
        ),
        (
            "Q7: What is the Coverage Gap analysis and how is Polarization scored?",
            "<b>Answer:</b> 'Coverage Gaps identify which outlets reported on a specific story and which outlets chose to ignore it—a key indicator of editorial agenda. For each event cluster, we calculate the standard deviation of sentiment across all reporting outlets as a <b>Polarization Score</b>. A high standard deviation highlights polarizing controversies where different media houses framed the exact same event in opposite emotional tones.'"
        )
    ]

    for q, a in qa_list:
        card_content = [
            Paragraph(q, qa_q_style),
            Paragraph(a, qa_a_style)
        ]
        card_table = Table([[card_content]], colWidths=[504])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
            ('BOX', (0,0), (-1,-1), 0.5, BORDER_COL),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(card_table)
        story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────
    # SECTION 5: FUTURE ROADMAP / HOW YOU WOULD SCALE IT
    # ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(Paragraph("5. System Scalability & Production Roadmap", h1_style))
    story.append(Paragraph("If asked <i>'How would you take this to production at enterprise scale?'</i>, highlight these 4 architectural enhancements:", body_style))
    
    scale_points = [
        ("Vector Database Integration", "Replace in-memory DBSCAN distance matrix with a managed vector database (e.g. Qdrant, Pinecone, or Milvus with HNSW indexing) to scale from hundreds to millions of article embeddings with sub-millisecond approximate nearest neighbor searches."),
        ("Distributed Message Queue", "Decouple scraper workers from LLM workers using Celery / Redis or Apache Kafka to process articles asynchronously across distributed worker nodes."),
        ("Relational / Time-Series DB", "Migrate from flat JSON files to PostgreSQL + pgvector, enabling SQL queries on historical bias shifts, outlet trends over 12 months, and complex aggregation queries."),
        ("Multi-Modal & Video Bias", "Expand ingestion to YouTube news transcripts and podcast audio using Whisper for automated transcription prior to LLM analysis.")
    ]

    for title, desc in scale_points:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated {PDF_OUTPUT}")

if __name__ == "__main__":
    build_pdf()
