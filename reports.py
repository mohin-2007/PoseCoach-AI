"""
PoseCoach AI - Report Generation Module
Generates downloadable CSV and professional PDF reports using ReportLab.
"""

import io
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def format_duration(seconds: float) -> str:
    """Format seconds into readable Xh Ym Zs format."""
    total_sec = int(round(seconds))
    if total_sec < 60:
        return f"{total_sec}s"
    minutes, sec = divmod(total_sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"


# ==================================================
# CSV REPORT GENERATORS
# ==================================================

def generate_session_csv(session: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
    """Generate CSV text for a single posture monitoring session."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Session Summary Section
    writer.writerow(["POSECOACH AI - SESSION REPORT"])
    writer.writerow(["Session ID", session.get("id", "N/A")])
    writer.writerow(["Session UUID", session.get("session_uuid", "N/A")])
    writer.writerow(["Start Time", session.get("start_time", "N/A")])
    writer.writerow(["End Time", session.get("end_time", "N/A")])
    writer.writerow(["Status", session.get("status", "N/A")])
    writer.writerow([])

    total_s = float(session.get("total_duration_seconds", 0.0))
    stand_s = float(session.get("standing_seconds", 0.0))
    sit_s = float(session.get("sitting_seconds", 0.0))
    squat_s = float(session.get("squatting_seconds", 0.0))
    bend_s = float(session.get("bending_seconds", 0.0))

    writer.writerow(["METRIC", "VALUE (SECONDS)", "FORMATTED", "PERCENTAGE"])
    writer.writerow(["Total Monitoring Time", f"{total_s:.1f}", format_duration(total_s), "100.0%"])
    writer.writerow(["Standing Time", f"{stand_s:.1f}", format_duration(stand_s), f"{(stand_s/total_s*100) if total_s else 0:.1f}%"])
    writer.writerow(["Sitting Time", f"{sit_s:.1f}", format_duration(sit_s), f"{(sit_s/total_s*100) if total_s else 0:.1f}%"])
    writer.writerow(["Squatting Time", f"{squat_s:.1f}", format_duration(squat_s), f"{(squat_s/total_s*100) if total_s else 0:.1f}%"])
    writer.writerow(["Bending Time", f"{bend_s:.1f}", format_duration(bend_s), f"{(bend_s/total_s*100) if total_s else 0:.1f}%"])
    writer.writerow(["Total Posture Transitions", session.get("total_transitions", 0), "-", "-"])
    writer.writerow(["Longest Sitting Streak", f"{float(session.get('longest_sitting_seconds', 0.0)):.1f}", format_duration(float(session.get("longest_sitting_seconds", 0.0))), "-"])
    writer.writerow(["Longest Bending Streak", f"{float(session.get('longest_bending_seconds', 0.0)):.1f}", format_duration(float(session.get("longest_bending_seconds", 0.0))), "-"])
    writer.writerow([])

    # Timeline Events Section
    writer.writerow(["POSTURE TRANSITIONS TIMELINE"])
    writer.writerow(["Event #", "Posture", "Start Time", "End Time", "Duration (s)", "Formatted Duration", "Avg Confidence (%)"])
    for idx, ev in enumerate(events, 1):
        ev_dur = float(ev.get("duration_seconds", 0.0))
        writer.writerow([
            idx,
            ev.get("posture", "Unknown"),
            ev.get("start_time", ""),
            ev.get("end_time", ""),
            f"{ev_dur:.1f}",
            format_duration(ev_dur),
            f"{float(ev.get('avg_confidence', 0.0)):.1f}%"
        ])

    return output.getvalue()


def generate_all_sessions_csv(sessions: List[Dict[str, Any]]) -> str:
    """Generate CSV text containing all recorded monitoring sessions."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Session ID", "Session UUID", "Start Time", "End Time", "Status",
        "Total Duration (s)", "Total Duration Formatted",
        "Standing (s)", "Sitting (s)", "Squatting (s)", "Bending (s)",
        "Standing %", "Sitting %", "Squatting %", "Bending %",
        "Transitions", "Longest Sitting (s)", "Longest Bending (s)"
    ])

    for s in sessions:
        tot = float(s.get("total_duration_seconds", 0.0))
        st = float(s.get("standing_seconds", 0.0))
        si = float(s.get("sitting_seconds", 0.0))
        sq = float(s.get("squatting_seconds", 0.0))
        be = float(s.get("bending_seconds", 0.0))

        writer.writerow([
            s.get("id"),
            s.get("session_uuid"),
            s.get("start_time"),
            s.get("end_time"),
            s.get("status"),
            f"{tot:.1f}",
            format_duration(tot),
            f"{st:.1f}",
            f"{si:.1f}",
            f"{sq:.1f}",
            f"{be:.1f}",
            f"{(st/tot*100) if tot else 0:.1f}%",
            f"{(si/tot*100) if tot else 0:.1f}%",
            f"{(sq/tot*100) if tot else 0:.1f}%",
            f"{(be/tot*100) if tot else 0:.1f}%",
            s.get("total_transitions", 0),
            f"{float(s.get('longest_sitting_seconds', 0.0)):.1f}",
            f"{float(s.get('longest_bending_seconds', 0.0)):.1f}"
        ])

    return output.getvalue()


# ==================================================
# PDF REPORT GENERATOR (REPORTLAB)
# ==================================================

def generate_session_pdf(session: Dict[str, Any], events: List[Dict[str, Any]]) -> bytes:
    """
    Generate an executive-style PDF posture monitoring report using ReportLab.
    Returns the binary PDF bytes.
    """
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

    # Custom styles
    primary_color = colors.HexColor("#0f172a")  # Dark Navy
    accent_color = colors.HexColor("#2563eb")   # Blue Accent
    muted_color = colors.HexColor("#64748b")    # Slate Muted
    card_bg = colors.HexColor("#f8fafc")        # Light Gray Card
    border_color = colors.HexColor("#e2e8f0")   # Soft Border

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=muted_color,
        spaceAfter=14
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=accent_color,
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=primary_color
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=muted_color
    )

    story = []

    # Title Banner
    story.append(Paragraph("PoseCoach AI – Posture Monitoring Session Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} &bull; "
        f"Session ID: #{session.get('id', 'N/A')} &bull; Status: {session.get('status', 'Completed').capitalize()}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=14))

    total_s = float(session.get("total_duration_seconds", 0.0))
    stand_s = float(session.get("standing_seconds", 0.0))
    sit_s = float(session.get("sitting_seconds", 0.0))
    squat_s = float(session.get("squatting_seconds", 0.0))
    bend_s = float(session.get("bending_seconds", 0.0))
    transitions = session.get("total_transitions", 0)
    longest_sit = float(session.get("longest_sitting_seconds", 0.0))
    longest_bend = float(session.get("longest_bending_seconds", 0.0))

    # Executive Summary Cards Table (2x2 Grid)
    summary_data = [
        [
            Paragraph(f"<b>Total Monitoring Time</b><br/><font size=13 color='{accent_color.hexval()}'><b>{format_duration(total_s)}</b></font>", body_style),
            Paragraph(f"<b>Posture Transitions</b><br/><font size=13 color='{accent_color.hexval()}'><b>{transitions}</b></font>", body_style)
        ],
        [
            Paragraph(f"<b>Longest Continuous Sitting</b><br/><font size=13 color='{accent_color.hexval()}'><b>{format_duration(longest_sit)}</b></font>", body_style),
            Paragraph(f"<b>Longest Continuous Bending</b><br/><font size=13 color='{accent_color.hexval()}'><b>{format_duration(longest_bend)}</b></font>", body_style)
        ]
    ]

    summary_table = Table(summary_data, colWidths=[270, 270])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), card_bg),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 1, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Posture Breakdown Table
    story.append(Paragraph("Posture Distribution & Time Allocation", section_heading))
    breakdown_data = [
        [
            Paragraph("Posture Category", table_header_style),
            Paragraph("Duration (Formatted)", table_header_style),
            Paragraph("Seconds", table_header_style),
            Paragraph("Percentage", table_header_style),
        ],
        [
            Paragraph("Standing", body_style),
            Paragraph(format_duration(stand_s), body_style),
            Paragraph(f"{stand_s:.1f}s", body_style),
            Paragraph(f"{(stand_s/total_s*100) if total_s else 0:.1f}%", body_style),
        ],
        [
            Paragraph("Sitting", body_style),
            Paragraph(format_duration(sit_s), body_style),
            Paragraph(f"{sit_s:.1f}s", body_style),
            Paragraph(f"{(sit_s/total_s*100) if total_s else 0:.1f}%", body_style),
        ],
        [
            Paragraph("Squatting", body_style),
            Paragraph(format_duration(squat_s), body_style),
            Paragraph(f"{squat_s:.1f}s", body_style),
            Paragraph(f"{(squat_s/total_s*100) if total_s else 0:.1f}%", body_style),
        ],
        [
            Paragraph("Bending", body_style),
            Paragraph(format_duration(bend_s), body_style),
            Paragraph(f"{bend_s:.1f}s", body_style),
            Paragraph(f"{(bend_s/total_s*100) if total_s else 0:.1f}%", body_style),
        ],
    ]

    breakdown_table = Table(breakdown_data, colWidths=[140, 140, 130, 130])
    breakdown_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, card_bg]),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 14))

    # Event Timeline Section
    story.append(Paragraph(f"Recorded Posture Timeline ({len(events)} events)", section_heading))
    if events:
        events_data = [
            [
                Paragraph("#", table_header_style),
                Paragraph("Posture", table_header_style),
                Paragraph("Start Time", table_header_style),
                Paragraph("End Time", table_header_style),
                Paragraph("Duration", table_header_style),
                Paragraph("Confidence", table_header_style),
            ]
        ]
        # Show up to 25 events on PDF to prevent unbounded document length
        shown_events = events[:25]
        for idx, ev in enumerate(shown_events, 1):
            dur = float(ev.get("duration_seconds", 0.0))
            st_str = ev.get("start_time", "")
            # Clean timestamp for display (HH:MM:SS)
            if "T" in st_str:
                st_str = st_str.split("T")[1][:8]
            et_str = ev.get("end_time", "")
            if "T" in et_str:
                et_str = et_str.split("T")[1][:8]

            events_data.append([
                Paragraph(str(idx), body_style),
                Paragraph(str(ev.get("posture", "Unknown")), body_style),
                Paragraph(st_str, body_style),
                Paragraph(et_str, body_style),
                Paragraph(format_duration(dur), body_style),
                Paragraph(f"{float(ev.get('avg_confidence', 0.0)):.1f}%", body_style),
            ])

        events_table = Table(events_data, colWidths=[30, 100, 105, 105, 100, 100])
        events_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_color),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, card_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(events_table)

        if len(events) > 25:
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<i>...and {len(events) - 25} more transitions. Download the CSV report for complete raw event logs.</i>", disclaimer_style))
    else:
        story.append(Paragraph("<i>No discrete posture transitions were logged for this session.</i>", disclaimer_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=10))

    # Health & Disclaimer Notice
    disclaimer_text = (
        "<b>Notice:</b> PoseCoach AI provides ergonomic awareness and automated posture classification "
        "via computer vision and machine learning. This report is for fitness and wellness tracking purposes only "
        "and is not intended to serve as medical diagnosis or clinical advice. Consult a licensed healthcare "
        "professional or ergonomist for posture-related health conditions."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    return buffer.getvalue()
