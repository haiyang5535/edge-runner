#!/usr/bin/env python3
"""
Safety Audit Report Generator
=============================

Generates comprehensive safety audit reports in PDF and JSON formats.
Aggregates event data, calculates statistics, and produces actionable insights.

Usage:
    python -m tools.safety_audit_report --start 2026-01-01 --end 2026-01-31
    python -m tools.safety_audit_report --start 2026-01-01 --end 2026-01-31 --format pdf
    python -m tools.safety_audit_report --start 2026-01-01 --end 2026-01-31 --format json
    python -m tools.safety_audit_report --last 7  # Last 7 days
    python -m tools.safety_audit_report --last 30 --output monthly_report.pdf
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event_store import EventStore, SafetyEvent, EventType, Severity


# ============================================================
# Report Data Structures
# ============================================================

@dataclass
class EventStats:
    """Statistics for a category of events."""
    total: int = 0
    acknowledged: int = 0
    unacknowledged: int = 0
    avg_duration_sec: float = 0.0
    max_duration_sec: float = 0.0
    by_zone: Dict[str, int] = None
    by_severity: Dict[str, int] = None
    
    def __post_init__(self):
        if self.by_zone is None:
            self.by_zone = {}
        if self.by_severity is None:
            self.by_severity = {}


@dataclass 
class AuditReport:
    """Complete Safety Audit Report structure."""
    # Metadata
    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    system_id: str
    
    # Executive Summary
    executive_summary: Dict[str, Any]
    
    # Event Analysis
    total_events: int
    events_by_type: Dict[str, EventStats]
    events_by_severity: Dict[str, int]
    events_by_zone: Dict[str, int]
    
    # Time Analysis
    daily_breakdown: Dict[str, int]
    hourly_distribution: Dict[int, int]
    peak_hours: List[int]
    
    # Acknowledgement Analysis
    ack_rate: float
    avg_ack_time_sec: Optional[float]
    
    # Trend Analysis
    trend_direction: str  # "improving", "stable", "worsening"
    week_over_week_change: Optional[float]
    
    # System Health
    calibration_status: str
    calibration_score: Optional[int]
    uptime_estimate: Optional[float]
    
    # Recommendations
    recommendations: List[str]
    
    # Raw data (for JSON export)
    event_sample: List[Dict]


# ============================================================
# Report Generator
# ============================================================

class SafetyAuditReportGenerator:
    """Generates comprehensive safety audit reports."""
    
    def __init__(self, db_path: str = "events.db"):
        self.db_path = db_path
        self.store = EventStore(db_path=db_path)
    
    def generate(
        self,
        start_date: str,
        end_date: str,
        system_id: str = "node_01"
    ) -> AuditReport:
        """
        Generate a complete audit report for the specified period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            system_id: System identifier
            
        Returns:
            AuditReport object
        """
        # Query all events in range
        events = self.store.query_events(
            start=f"{start_date}T00:00:00",
            end=f"{end_date}T23:59:59",
            limit=50000
        )
        
        # Generate report ID
        report_id = f"AUDIT-{start_date.replace('-', '')}-{end_date.replace('-', '')}"
        
        # Calculate statistics
        events_by_type = self._analyze_by_type(events)
        events_by_severity = self._count_by_severity(events)
        events_by_zone = self._count_by_zone(events)
        daily_breakdown = self._daily_breakdown(events)
        hourly_dist = self._hourly_distribution(events)
        ack_rate, avg_ack_time = self._acknowledgement_analysis(events)
        trend_dir, wow_change = self._trend_analysis(daily_breakdown)
        calib_status, calib_score = self._check_calibration()
        recommendations = self._generate_recommendations(
            events, events_by_type, ack_rate, trend_dir
        )
        
        # Create executive summary
        executive_summary = {
            "total_safety_events": len(events),
            "high_severity_events": events_by_severity.get("HIGH", 0),
            "acknowledgement_rate_pct": round(ack_rate * 100, 1),
            "most_common_violation": max(events_by_type.keys(), 
                                         key=lambda k: events_by_type[k].total) if events_by_type else "N/A",
            "trend": trend_dir,
            "calibration_status": calib_status,
            "recommendation_count": len(recommendations)
        }
        
        # Sample events for JSON export
        event_sample = [e.to_dict() for e in events[:20]]
        
        return AuditReport(
            report_id=report_id,
            generated_at=datetime.now().isoformat(),
            period_start=start_date,
            period_end=end_date,
            system_id=system_id,
            executive_summary=executive_summary,
            total_events=len(events),
            events_by_type={k: asdict(v) for k, v in events_by_type.items()},
            events_by_severity=events_by_severity,
            events_by_zone=events_by_zone,
            daily_breakdown=daily_breakdown,
            hourly_distribution=hourly_dist,
            peak_hours=self._find_peak_hours(hourly_dist),
            ack_rate=round(ack_rate, 3),
            avg_ack_time_sec=round(avg_ack_time, 1) if avg_ack_time else None,
            trend_direction=trend_dir,
            week_over_week_change=round(wow_change, 1) if wow_change else None,
            calibration_status=calib_status,
            calibration_score=calib_score,
            uptime_estimate=None,  # TODO: Integrate with stability reports
            recommendations=recommendations,
            event_sample=event_sample
        )
    
    def _analyze_by_type(self, events: List[SafetyEvent]) -> Dict[str, EventStats]:
        """Analyze events grouped by type."""
        by_type = defaultdict(list)
        for e in events:
            type_key = e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type)
            by_type[type_key].append(e)
        
        result = {}
        for event_type, type_events in by_type.items():
            durations = [e.duration_seconds for e in type_events if e.duration_seconds]
            
            stats = EventStats(
                total=len(type_events),
                acknowledged=sum(1 for e in type_events if e.acknowledged),
                unacknowledged=sum(1 for e in type_events if not e.acknowledged),
                avg_duration_sec=sum(durations) / len(durations) if durations else 0,
                max_duration_sec=max(durations) if durations else 0,
                by_zone=self._count_by_zone(type_events),
                by_severity=self._count_by_severity(type_events)
            )
            result[event_type] = stats
        
        return result
    
    def _count_by_severity(self, events: List[SafetyEvent]) -> Dict[str, int]:
        """Count events by severity level."""
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for e in events:
            sev = e.severity.value if hasattr(e.severity, 'value') else str(e.severity)
            if sev in counts:
                counts[sev] += 1
        return counts
    
    def _count_by_zone(self, events: List[SafetyEvent]) -> Dict[str, int]:
        """Count events by zone."""
        counts = defaultdict(int)
        for e in events:
            counts[e.zone_id] += 1
        return dict(counts)
    
    def _daily_breakdown(self, events: List[SafetyEvent]) -> Dict[str, int]:
        """Break down events by day."""
        daily = defaultdict(int)
        for e in events:
            date = e.timestamp[:10]
            daily[date] += 1
        return dict(sorted(daily.items()))
    
    def _hourly_distribution(self, events: List[SafetyEvent]) -> Dict[int, int]:
        """Distribution of events by hour of day."""
        hourly = {h: 0 for h in range(24)}
        for e in events:
            try:
                hour = int(e.timestamp[11:13])
                hourly[hour] += 1
            except (ValueError, IndexError):
                pass
        return hourly
    
    def _find_peak_hours(self, hourly: Dict[int, int]) -> List[int]:
        """Find the top 3 peak hours."""
        if not hourly:
            return []
        sorted_hours = sorted(hourly.items(), key=lambda x: x[1], reverse=True)
        return [h for h, c in sorted_hours[:3] if c > 0]
    
    def _acknowledgement_analysis(self, events: List[SafetyEvent]) -> tuple:
        """Analyze acknowledgement rates and times."""
        if not events:
            return 0.0, None
        
        acked = [e for e in events if e.acknowledged]
        ack_rate = len(acked) / len(events) if events else 0
        
        # Calculate average acknowledgement time
        ack_times = []
        for e in acked:
            if e.acknowledge_time and e.timestamp:
                try:
                    event_time = datetime.fromisoformat(e.timestamp)
                    ack_time = datetime.fromisoformat(e.acknowledge_time)
                    delta = (ack_time - event_time).total_seconds()
                    if 0 < delta < 86400:  # Less than 24 hours
                        ack_times.append(delta)
                except (ValueError, TypeError):
                    pass
        
        avg_ack_time = sum(ack_times) / len(ack_times) if ack_times else None
        
        return ack_rate, avg_ack_time
    
    def _trend_analysis(self, daily: Dict[str, int]) -> tuple:
        """Analyze trends over the period."""
        if len(daily) < 7:
            return "insufficient_data", None
        
        dates = sorted(daily.keys())
        
        # Compare last week vs previous week
        if len(dates) >= 14:
            last_week = sum(daily.get(d, 0) for d in dates[-7:])
            prev_week = sum(daily.get(d, 0) for d in dates[-14:-7])
            
            if prev_week > 0:
                change = ((last_week - prev_week) / prev_week) * 100
                
                if change < -10:
                    return "improving", change
                elif change > 10:
                    return "worsening", change
                else:
                    return "stable", change
        
        # Simple trend based on first half vs second half
        mid = len(dates) // 2
        first_half = sum(daily.get(d, 0) for d in dates[:mid])
        second_half = sum(daily.get(d, 0) for d in dates[mid:])
        
        if first_half > 0:
            change = ((second_half - first_half) / first_half) * 100
            if change < -10:
                return "improving", change
            elif change > 10:
                return "worsening", change
        
        return "stable", 0
    
    def _check_calibration(self) -> tuple:
        """Check calibration status."""
        calib_path = "configs/camera_calibration.json"
        
        if not Path(calib_path).exists():
            return "MISSING", None
        
        try:
            # Import validator if available
            from tools.calibrate_floor import CalibrationValidator
            result = CalibrationValidator.validate_calibration(calib_path)
            
            if not result['valid']:
                return "INVALID", result['score']
            elif result['score'] >= 80:
                return "OK", result['score']
            else:
                return "NEEDS_REVIEW", result['score']
        except ImportError:
            return "UNKNOWN", None
    
    def _generate_recommendations(
        self,
        events: List[SafetyEvent],
        by_type: Dict[str, EventStats],
        ack_rate: float,
        trend: str
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        # Acknowledgement rate
        if ack_rate < 0.5:
            recs.append(
                "⚠️ LOW ACKNOWLEDGEMENT RATE: Only {:.0%} of events acknowledged. "
                "Consider training staff on event review procedures.".format(ack_rate)
            )
        
        # High severity events
        high_sev = sum(1 for e in events if 
                      (e.severity.value if hasattr(e.severity, 'value') else str(e.severity)) == "HIGH")
        if high_sev > 10:
            recs.append(
                f"🔴 HIGH SEVERITY ALERT: {high_sev} high-severity events detected. "
                "Immediate review and corrective action recommended."
            )
        
        # Most common violation
        if by_type:
            most_common = max(by_type.items(), key=lambda x: x[1].total)
            if most_common[1].total > 20:
                recs.append(
                    f"📍 REPEAT VIOLATION: '{most_common[0]}' occurred {most_common[1].total} times. "
                    "Consider physical barriers, signage, or additional training."
                )
        
        # Trend worsening
        if trend == "worsening":
            recs.append(
                "📈 NEGATIVE TREND: Safety events are increasing. "
                "Root cause analysis recommended to identify contributing factors."
            )
        
        # Zone-specific issues
        zone_counts = defaultdict(int)
        for e in events:
            zone_counts[e.zone_id] += 1
        
        if zone_counts:
            hotspot = max(zone_counts.items(), key=lambda x: x[1])
            if hotspot[1] > 15:
                recs.append(
                    f"🔥 ZONE HOTSPOT: Zone '{hotspot[0]}' has {hotspot[1]} events. "
                    "Consider zone reconfiguration or additional controls."
                )
        
        # Calibration
        calib_status, _ = self._check_calibration()
        if calib_status in ("MISSING", "INVALID"):
            recs.append(
                "⚙️ CALIBRATION REQUIRED: System calibration is missing or invalid. "
                "Run: python -m tools.calibrate_floor --video <video>"
            )
        elif calib_status == "NEEDS_REVIEW":
            recs.append(
                "⚙️ CALIBRATION REVIEW: Calibration quality is below optimal. "
                "Run: python -m tools.calibrate_floor --validate configs/camera_calibration.json"
            )
        
        # If no issues
        if not recs:
            recs.append(
                "✅ GOOD STATUS: No critical issues identified. "
                "Continue monitoring and maintain current safety protocols."
            )
        
        return recs


# ============================================================
# Export Functions
# ============================================================

def export_json(report: AuditReport, output_path: str):
    """Export report to JSON file."""
    data = asdict(report)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"✅ JSON report saved to: {output_path}")


def export_pdf(report: AuditReport, output_path: str):
    """Export report to PDF file."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.platypus import PageBreak, HRFlowable
    except ImportError:
        print("⚠️ reportlab not installed. Falling back to text report.")
        print("   Install with: pip install reportlab")
        export_text(report, output_path.replace('.pdf', '.txt'))
        return
    
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1a1a2e')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#16213e')
    )
    
    # Title
    story.append(Paragraph("edge-runner Safety Audit Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0f4c75')))
    story.append(Spacer(1, 20))
    
    # Metadata
    meta_data = [
        ["Report ID", report.report_id],
        ["Generated", report.generated_at[:19].replace('T', ' ')],
        ["Period", f"{report.period_start} to {report.period_end}"],
        ["System", report.system_id]
    ]
    meta_table = Table(meta_data, colWidths=[1.5*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a4a4a')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 30))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    exec_data = [
        ["Total Safety Events", str(report.executive_summary['total_safety_events'])],
        ["High Severity Events", str(report.executive_summary['high_severity_events'])],
        ["Acknowledgement Rate", f"{report.executive_summary['acknowledgement_rate_pct']}%"],
        ["Most Common Violation", report.executive_summary['most_common_violation']],
        ["Trend", report.executive_summary['trend'].replace('_', ' ').title()],
        ["Calibration Status", report.executive_summary['calibration_status']]
    ]
    exec_table = Table(exec_data, colWidths=[2.5*inch, 3*inch])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 30))
    
    # Events by Severity
    story.append(Paragraph("Events by Severity", heading_style))
    sev_data = [["Severity", "Count"]]
    for sev in ["HIGH", "MEDIUM", "LOW"]:
        sev_data.append([sev, str(report.events_by_severity.get(sev, 0))])
    sev_table = Table(sev_data, colWidths=[2*inch, 2*inch])
    sev_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(sev_table)
    story.append(Spacer(1, 30))
    
    # Events by Zone
    if report.events_by_zone:
        story.append(Paragraph("Events by Zone", heading_style))
        zone_data = [["Zone", "Events"]]
        for zone, count in sorted(report.events_by_zone.items(), key=lambda x: x[1], reverse=True)[:10]:
            zone_data.append([zone, str(count)])
        zone_table = Table(zone_data, colWidths=[3*inch, 1.5*inch])
        zone_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(zone_table)
        story.append(Spacer(1, 30))
    
    # Peak Hours
    if report.peak_hours:
        story.append(Paragraph("Peak Activity Hours", heading_style))
        peak_text = ", ".join([f"{h:02d}:00" for h in report.peak_hours])
        story.append(Paragraph(f"Most events occur during: {peak_text}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(PageBreak())
    story.append(Paragraph("Recommendations", heading_style))
    for i, rec in enumerate(report.recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        story.append(Spacer(1, 10))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF report saved to: {output_path}")


def export_text(report: AuditReport, output_path: str):
    """Export report to plain text file."""
    lines = []
    lines.append("=" * 60)
    lines.append("edge-runner SAFETY AUDIT REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Report ID:    {report.report_id}")
    lines.append(f"Generated:    {report.generated_at}")
    lines.append(f"Period:       {report.period_start} to {report.period_end}")
    lines.append(f"System:       {report.system_id}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    for key, value in report.executive_summary.items():
        lines.append(f"  {key.replace('_', ' ').title()}: {value}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("EVENTS BY SEVERITY")
    lines.append("-" * 40)
    for sev, count in report.events_by_severity.items():
        lines.append(f"  {sev}: {count}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"  {i}. {rec}")
    lines.append("")
    lines.append("=" * 60)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Text report saved to: {output_path}")


def print_report_summary(report: AuditReport):
    """Print report summary to console."""
    print("\n" + "=" * 60)
    print("edge-runner SAFETY AUDIT REPORT")
    print("=" * 60)
    print(f"Report ID:    {report.report_id}")
    print(f"Period:       {report.period_start} to {report.period_end}")
    print(f"Total Events: {report.total_events}")
    print()
    print("Executive Summary:")
    print(f"  High Severity:    {report.executive_summary['high_severity_events']}")
    print(f"  Ack Rate:         {report.executive_summary['acknowledgement_rate_pct']}%")
    print(f"  Trend:            {report.executive_summary['trend']}")
    print(f"  Calibration:      {report.executive_summary['calibration_status']}")
    print()
    print("Events by Severity:")
    for sev, count in report.events_by_severity.items():
        print(f"  {sev:8}: {count}")
    print()
    print("Top Zones:")
    for zone, count in sorted(report.events_by_zone.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {zone}: {count}")
    print()
    print("Recommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec[:80]}...")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Safety Audit Reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tools.safety_audit_report --start 2026-01-01 --end 2026-01-31
    python -m tools.safety_audit_report --last 7 --format pdf
    python -m tools.safety_audit_report --last 30 --output monthly.pdf
    python -m tools.safety_audit_report --start 2026-01-01 --end 2026-01-07 --format json
        """
    )
    
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--last', type=int, metavar='DAYS',
                       help='Generate report for last N days')
    parser.add_argument('--format', choices=['pdf', 'json', 'text', 'summary'],
                       default='summary',
                       help='Output format (default: summary)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path')
    parser.add_argument('--db', type=str, default='events.db',
                       help='Database path (default: events.db)')
    parser.add_argument('--system-id', type=str, default='node_01',
                       help='System identifier')
    
    args = parser.parse_args()
    
    # Determine date range
    if args.last:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.last)).strftime('%Y-%m-%d')
    elif args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        # Default to last 7 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        print(f"ℹ️  No date range specified, using last 7 days: {start_date} to {end_date}")
    
    # Generate report
    print(f"\n📊 Generating Safety Audit Report...")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Database: {args.db}")
    
    generator = SafetyAuditReportGenerator(db_path=args.db)
    report = generator.generate(start_date, end_date, args.system_id)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        ext = {'pdf': '.pdf', 'json': '.json', 'text': '.txt'}.get(args.format, '')
        if ext:
            output_path = f"reports/audit_{start_date}_{end_date}{ext}"
        else:
            output_path = None
    
    # Export
    if args.format == 'pdf':
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_pdf(report, output_path)
    elif args.format == 'json':
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_json(report, output_path)
    elif args.format == 'text':
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_text(report, output_path)
    else:
        print_report_summary(report)
    
    return 0


if __name__ == "__main__":
    exit(main())
