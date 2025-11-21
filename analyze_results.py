#!/usr/bin/env python3
"""
Main script to run all result analysis and visualization
"""

import sys
from pathlib import Path

def main():
    """Run all analysis scripts"""
    print("="*80)
    print("MS DETECTION RESULTS - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print()
    
    results_dir = Path('results')
    
    # Check if results directory exists
    if not results_dir.exists():
        print("❌ Error: 'results' directory not found!")
        print("Please ensure test results are available.")
        sys.exit(1)
    
    print("✓ Results directory found")
    print()
    
    # Step 1: Generate visualizations
    print("="*80)
    print("STEP 1: GENERATING VISUALIZATIONS")
    print("="*80)
    print()
    
    try:
        from visualize_results import ResultsVisualizer
        
        visualizer = ResultsVisualizer(results_dir='results')
        visualizer.generate_all()
        
        print()
        print("✓ Visualizations generated successfully!")
        print()
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Generate markdown report
    print("="*80)
    print("STEP 2: GENERATING MARKDOWN REPORT")
    print("="*80)
    print()
    
    try:
        from generate_report import ReportGenerator
        
        generator = ReportGenerator(results_dir='results')
        output_path = generator.save_report(output_path='results/RESULTS_REPORT.md')
        
        print()
        print("✓ Report generated successfully!")
        print()
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("📁 Generated Files:")
    print()
    print("📊 Visualizations:")
    vis_dir = results_dir / 'visualizations'
    if vis_dir.exists():
        for file in sorted(vis_dir.glob('*')):
            size = file.stat().st_size / 1024  # KB
            print(f"   • {file.name} ({size:.1f} KB)")
    
    print()
    print("📄 Report:")
    report_path = results_dir / 'RESULTS_REPORT.md'
    if report_path.exists():
        size = report_path.stat().st_size / 1024  # KB
        print(f"   • RESULTS_REPORT.md ({size:.1f} KB)")
    
    print()
    print("="*80)
    print()
    print("Next Steps:")
    print("  1. View visualizations: Open files in results/visualizations/")
    print("  2. Read the report: cat results/RESULTS_REPORT.md")
    print("  3. Share results: Use the comprehensive_report.png for presentations")
    print()


if __name__ == '__main__':
    main()

