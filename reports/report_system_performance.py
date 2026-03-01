"""
Report 3: SafeEdge System Performance — For hackathon judges and stakeholders.
Uses local alert JSON data only (no API calls needed).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

from reports.pdf_theme import SafeEdgePDF
from reports.data_fetcher import load_safeedge_alerts
from reports.ai_narrator import generate_narrative

CHARTS = Path(__file__).parent / "charts"
OUTPUT = Path(__file__).parent.parent / "docs"

COLORS = {
    "HIGH": "#DC3545",
    "WARNING": "#FFC107",
    "CRITICAL": "#7B2D26",
    "EARLY_WARNING": "#FD7E14",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


def _chart_confidence(alerts, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [f"Alert {i+1}" for i in range(len(alerts))]
    confs = [a["confidence"] for a in alerts]
    colors = [COLORS.get(a["risk_score"], "#6C757D") for a in alerts]
    bars = ax.barh(names, confs, color=colors, edgecolor="white", height=0.6)
    for bar, c in zip(bars, confs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{c:.1%}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Detection Confidence")
    ax.set_title("Detection Confidence by Alert")
    ax.invert_yaxis()
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in COLORS.items() if l in {a["risk_score"] for a in alerts}]
    ax.legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chart_event_pie(alerts, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    events = {}
    for a in alerts:
        e = a["event"].replace("_", " ").title()
        events[e] = events.get(e, 0) + 1
    colors = ["#DC3545" if "Fire" in k else "#FD7E14" for k in events]
    wedges, texts, autotexts = ax.pie(
        events.values(), labels=events.keys(), autopct="%1.0f%%",
        colors=colors, startangle=90, textprops={"fontsize": 12}
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    ax.set_title("Detection Event Types")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chart_edge_metrics(alerts, path):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(alerts))
    labels = [f"Alert {i+1}" for i in x]

    fps = [a["edge_metrics"]["fps_current"] for a in alerts]
    cpu = [a["edge_metrics"]["cpu_pct"] for a in alerts]

    ax1.bar([i - 0.2 for i in x], fps, 0.4, color="#0D6EFD", label="FPS (current)")
    ax1.set_ylabel("Frames per Second", color="#0D6EFD")
    ax1.set_xlabel("Alert")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.plot(list(x), cpu, color="#DC3545", marker="o", linewidth=2, label="CPU %")
    ax2.set_ylabel("CPU Usage %", color="#DC3545")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Edge Performance: FPS vs CPU Usage")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chart_memory(alerts, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(alerts))
    mem = [a["edge_metrics"]["mem_mb"] for a in alerts]
    ax.bar(x, mem, color="#198754", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Alert {i+1}" for i in x])
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage per Alert")
    ax.axhline(y=512, color="#DC3545", linestyle="--", alpha=0.7, label="512 MB threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chart_timeline(alerts, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    times = []
    for a in alerts:
        ts = a["timestamp"]
        dt = datetime.fromisoformat(ts)
        times.append(dt)

    confs = [a["confidence"] for a in alerts]
    colors = [COLORS.get(a["risk_score"], "#6C757D") for a in alerts]
    sizes = [c * 300 for c in confs]

    ax.scatter(times, confs, s=sizes, c=colors, alpha=0.8, edgecolors="black", linewidth=0.5)
    for i, a in enumerate(alerts):
        ax.annotate(
            f'{a["event"].split("_")[0].title()}\n{a["risk_score"]}',
            (times[i], confs[i]), textcoords="offset points",
            xytext=(0, 15), ha="center", fontsize=8,
        )
    ax.set_ylabel("Confidence")
    ax.set_xlabel("Time")
    ax.set_title("Alert Timeline (bubble size = confidence)")
    ax.set_ylim(0.5, 1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chart_positive_frames(alerts, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(alerts))
    pos = [a["positive_frames"] for a in alerts]
    win = [a["window_size"] for a in alerts]
    ax.bar([i - 0.15 for i in x], pos, 0.3, color="#0D6EFD", label="Positive Frames")
    ax.bar([i + 0.15 for i in x], win, 0.3, color="#ADB5BD", label="Window Size")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Alert {i+1}" for i in x])
    ax.set_ylabel("Frame Count")
    ax.set_title("Multi-Frame Confirmation Analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate(use_ai: bool = True):
    CHARTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    alerts = load_safeedge_alerts()
    if not alerts:
        print("  No alert data found. Skipping Report 3.")
        return

    print(f"  Loaded {len(alerts)} alerts")

    # Generate charts
    paths = {}
    charts = [
        ("confidence", _chart_confidence),
        ("event_pie", _chart_event_pie),
        ("edge_metrics", _chart_edge_metrics),
        ("memory", _chart_memory),
        ("timeline", _chart_timeline),
        ("positive_frames", _chart_positive_frames),
    ]
    for name, fn in charts:
        p = str(CHARTS / f"r3_{name}.png")
        fn(alerts, p)
        paths[name] = p
        print(f"  Chart: {name}")

    # Build PDF
    pdf = SafeEdgePDF(
        title="System Performance Report",
        subtitle="SafeEdge Edge-Based Fire Detection Analytics",
        audience="Hackathon Judges & Stakeholders",
    )
    pdf.add_cover_page()

    # Executive summary
    pdf.add_section("Executive Summary")
    avg_conf = sum(a["confidence"] for a in alerts) / len(alerts)
    avg_fps = sum(a["edge_metrics"]["fps_avg"] for a in alerts) / len(alerts)
    high_count = sum(1 for a in alerts if a["risk_score"] == "HIGH")
    warn_count = sum(1 for a in alerts if a["risk_score"] == "WARNING")
    fire_count = sum(1 for a in alerts if a["event"] == "fire_detected")
    smoke_count = sum(1 for a in alerts if a["event"] == "smoke_detected")

    summary_data = (
        f"Total alerts: {len(alerts)}. Fire: {fire_count}, Smoke: {smoke_count}. "
        f"HIGH risk: {high_count}, WARNING: {warn_count}. "
        f"Average confidence: {avg_conf:.1%}. Average FPS: {avg_fps:.1f}. "
        f"All detections from CAM_01 with multi-frame confirmation (5-8/8 positive frames)."
    )
    if use_ai:
        narrative = generate_narrative("Executive Summary", summary_data, "hackathon judges")
    else:
        narrative = summary_data
    pdf.add_narrative(narrative)

    pdf.add_stat_row([
        ("Total Alerts", str(len(alerts)), SafeEdgePDF.RED),
        ("Avg Confidence", f"{avg_conf:.0%}", SafeEdgePDF.BLUE),
        ("Avg FPS", f"{avg_fps:.1f}", SafeEdgePDF.GREEN),
        ("HIGH Risk", str(high_count), SafeEdgePDF.ORANGE),
    ])

    # Detection Performance
    pdf.add_section("Detection Performance")
    if use_ai:
        det_narrative = generate_narrative(
            "Detection Performance",
            f"Confidence range: {min(a['confidence'] for a in alerts):.1%} to {max(a['confidence'] for a in alerts):.1%}. "
            f"Event types: {fire_count} fire, {smoke_count} smoke. "
            f"All alerts confirmed via multi-frame sliding window (5-8 out of 8 frames positive). "
            f"Risk levels: {high_count} HIGH, {warn_count} WARNING.",
            "hackathon judges",
        )
    else:
        det_narrative = f"The system detected {len(alerts)} events with confidence ranging from {min(a['confidence'] for a in alerts):.1%} to {max(a['confidence'] for a in alerts):.1%}."
    pdf.add_narrative(det_narrative)
    pdf.add_chart(paths["confidence"], "Detection confidence by alert, color-coded by risk level")
    pdf.add_chart(paths["event_pie"], "Distribution of detection event types")

    # Edge Computing Metrics
    pdf.add_section("Edge Computing Metrics")
    avg_cpu = sum(a["edge_metrics"]["cpu_pct"] for a in alerts) / len(alerts)
    avg_mem = sum(a["edge_metrics"]["mem_mb"] for a in alerts) / len(alerts)
    if use_ai:
        edge_narrative = generate_narrative(
            "Edge Computing Performance",
            f"FPS range: {min(a['edge_metrics']['fps_current'] for a in alerts):.1f} to {max(a['edge_metrics']['fps_current'] for a in alerts):.1f}. "
            f"Average CPU: {avg_cpu:.0f}%. Average memory: {avg_mem:.0f} MB. "
            f"Model: YOLOv8n (~6MB). All inference on CPU, proving edge viability. "
            f"No GPU required. System runs on standard laptop hardware.",
            "hackathon judges",
        )
    else:
        edge_narrative = f"Average FPS: {avg_fps:.1f}. CPU: {avg_cpu:.0f}%. Memory: {avg_mem:.0f} MB. Edge-viable on standard hardware."
    pdf.add_narrative(edge_narrative)
    pdf.add_chart(paths["edge_metrics"], "FPS throughput vs CPU utilization per alert")
    pdf.add_chart(paths["memory"], "Memory footprint per detection alert")

    # Alert Timeline
    pdf.add_section("Alert Timeline & Escalation")
    if use_ai:
        timeline_narrative = generate_narrative(
            "Alert Timeline Analysis",
            f"Detection window: {alerts[0]['timestamp'][:19]} to {alerts[-1]['timestamp'][:19]}. "
            f"Pattern shows smoke detection progressing to fire confirmation. "
            f"Locations span multiple NTU campus buildings: {', '.join(set(a['location']['building'] for a in alerts))}.",
            "hackathon judges",
        )
    else:
        timeline_narrative = f"Alerts span from {alerts[0]['timestamp'][:19]} to {alerts[-1]['timestamp'][:19]}."
    pdf.add_narrative(timeline_narrative)
    pdf.add_chart(paths["timeline"], "Alert timeline with confidence as bubble size")

    # Alert detail table
    pdf.add_subsection("Alert Details")
    headers = ["#", "Event", "Confidence", "Risk", "Location", "Time"]
    rows = []
    for i, a in enumerate(alerts):
        rows.append([
            str(i + 1),
            a["event"].replace("_", " ").title(),
            f"{a['confidence']:.1%}",
            a["risk_score"],
            f"{a['location']['building']} F{a['location']['floor']}",
            a["timestamp"][11:19],
        ])
    pdf.add_table(headers, rows)

    # Multi-Frame Confirmation
    pdf.add_section("Multi-Frame Confirmation")
    if use_ai:
        mf_narrative = generate_narrative(
            "Multi-Frame Confirmation",
            f"Sliding window: {alerts[0]['window_size']} frames. "
            f"Positive frame range: {min(a['positive_frames'] for a in alerts)} to {max(a['positive_frames'] for a in alerts)}. "
            f"Confirmation threshold: 5 of 8 frames. "
            f"This reduces false positives from flickering lights, reflections, etc.",
            "hackathon judges",
        )
    else:
        mf_narrative = "Multi-frame confirmation requires 5 of 8 consecutive frames to confirm detection."
    pdf.add_narrative(mf_narrative)
    pdf.add_chart(paths["positive_frames"], "Positive frames vs window size per alert")

    # Save
    out_path = str(OUTPUT / "SafeEdge_System_Performance.pdf")
    pdf.output(out_path)
    print(f"  Saved: {out_path}")
