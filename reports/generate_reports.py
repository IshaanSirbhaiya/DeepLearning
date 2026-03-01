"""
SafeEdge Intelligence Report Generator
Generates 3 PDF reports with real data, matplotlib charts, and AI narratives.

Usage:
    python -m reports.generate_reports              # All 3 reports
    python -m reports.generate_reports --report 3   # System performance only
    python -m reports.generate_reports --no-ai      # Skip OpenAI narratives
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="SafeEdge Intelligence Report Generator")
    parser.add_argument("--report", type=int, choices=[1, 2, 3], help="Generate specific report (1=Fire Trends, 2=Emergency Response, 3=System Performance)")
    parser.add_argument("--no-ai", action="store_true", help="Skip OpenAI narrative generation (faster, offline)")
    args = parser.parse_args()

    use_ai = not args.no_ai
    reports = [args.report] if args.report else [3, 1, 2]  # Report 3 first (local data)

    print("=" * 60)
    print("  SafeEdge Intelligence Report Generator")
    print("=" * 60)
    if not use_ai:
        print("  Mode: --no-ai (skipping OpenAI narratives)\n")

    for num in reports:
        if num == 1:
            print("\n[Report 1] Fire Incident Trend Analysis")
            from reports.report_fire_trends import generate as gen1
            gen1(use_ai=use_ai)
        elif num == 2:
            print("\n[Report 2] Emergency Response Optimization")
            from reports.report_emergency_response import generate as gen2
            gen2(use_ai=use_ai)
        elif num == 3:
            print("\n[Report 3] SafeEdge System Performance")
            from reports.report_system_performance import generate as gen3
            gen3(use_ai=use_ai)

    print("\n" + "=" * 60)
    print("  All reports generated! Check docs/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
