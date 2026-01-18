#!/usr/bin/env python
"""
PEST++ Results Cleanup and Summary
==================================

Utility for processing PEST++ calibration output:
- Detect success/failure based on output files and metrics
- Summarize results with key metrics
- Archive important files and clean up intermediates
- Preserve debug files on failure

Usage
-----
    # As module
    from swimrs.calibrate.pest_cleanup import PestResults

    results = PestResults(pest_dir, project_name)
    success, issues = results.is_successful()
    if success:
        summary = results.get_summary()
        results.cleanup(archive_dir='calibration_archive')

    # From command line
    python -m swimrs.calibrate.pest_cleanup /path/to/pest --archive ./archive
    python -m swimrs.calibrate.pest_cleanup /path/to/pest --check-only
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


class PestResults:
    """Parse, summarize, and clean up PEST++ IES results."""

    # Files to always archive on success
    ARCHIVE_FILES = [
        '{project}.pst',
        '{project}.rec',
        '{project}.phi.meas.csv',
        '{project}.phi.composite.csv',
        'params.csv',
        'localizer_summary.json',
        'loc.mat',
    ]

    # Files to keep only on failure (in addition to archive files)
    DEBUG_FILES = [
        'panther_master.rec',
        '{project}.*.obs.csv',
        '{project}.*.par.csv',
        '{project}.*.pdc.csv',
        '{project}.*.pcs.csv',
    ]

    # Patterns for files safe to delete
    CLEANUP_PATTERNS = [
        '*.jcb',
        '*.jco',
        '*.rei',
        '*.rst',
    ]

    def __init__(self, pest_dir: str, project_name: str):
        """
        Initialize results handler.

        Args:
            pest_dir: Path to pest/ directory (contains master/, pst file, etc.)
            project_name: Project name (e.g., '2_Fort_Peck')
        """
        self.pest_dir = Path(pest_dir)
        self.master_dir = self.pest_dir / 'master'
        self.project_name = project_name

        # Parent of pest_dir typically contains workers/
        self.pest_run_dir = self.pest_dir.parent
        self.workers_dir = self.pest_run_dir / 'workers'

        self._rec_content = None
        self._phi_data = None
        self._noptmax = None

    @property
    def rec_file(self) -> Path:
        """Path to main record file."""
        return self.master_dir / f'{self.project_name}.rec'

    @property
    def pst_file(self) -> Path:
        """Path to control file."""
        return self.pest_dir / f'{self.project_name}.pst'

    def _read_rec_file(self) -> str:
        """Read and cache record file content."""
        if self._rec_content is None:
            if self.rec_file.exists():
                self._rec_content = self.rec_file.read_text(errors='ignore')
            else:
                self._rec_content = ''
        return self._rec_content

    def _read_phi_data(self) -> pd.DataFrame | None:
        """Read and cache phi measurement data."""
        if self._phi_data is None:
            phi_file = self.master_dir / f'{self.project_name}.phi.meas.csv'
            if phi_file.exists():
                try:
                    self._phi_data = pd.read_csv(phi_file)
                except Exception:
                    self._phi_data = pd.DataFrame()
        return self._phi_data

    def _get_noptmax(self) -> int | None:
        """Extract noptmax from control file or record."""
        if self._noptmax is not None:
            return self._noptmax

        # Try to read from pst file
        if self.pst_file.exists():
            content = self.pst_file.read_text()
            match = re.search(r'NOPTMAX\s+(\d+)', content, re.IGNORECASE)
            if match:
                self._noptmax = int(match.group(1))
                return self._noptmax

        # Infer from existing par files
        par_files = list(self.master_dir.glob(f'{self.project_name}.*.par.csv'))
        if par_files:
            iterations = []
            for f in par_files:
                match = re.search(rf'{self.project_name}\.(\d+)\.par\.csv', f.name)
                if match:
                    iterations.append(int(match.group(1)))
            if iterations:
                self._noptmax = max(iterations)
                return self._noptmax

        return None

    def _get_par_files(self) -> list[Path]:
        """Get all parameter CSV files sorted by iteration."""
        par_files = list(self.master_dir.glob(f'{self.project_name}.*.par.csv'))
        # Sort by iteration number
        def get_iter(f):
            match = re.search(rf'{self.project_name}\.(\d+)\.par\.csv', f.name)
            return int(match.group(1)) if match else -1
        return sorted(par_files, key=get_iter)

    def is_successful(self) -> tuple[bool, list[str]]:
        """Check if calibration succeeded.

        Returns:
            Tuple of (success, issues) where:
            - success: True if all primary criteria pass
            - issues: List of problems found (empty if successful)
        """
        issues = []

        # 1. Check record file exists
        if not self.rec_file.exists():
            issues.append(f"Record file not found: {self.rec_file}")
            return False, issues

        rec_content = self._read_rec_file()

        # 2. Check for fatal errors
        fatal_patterns = [
            'FATAL ERROR',
            'terminated abnormally',
            'Traceback (most recent call last)',
            'PEST++ run failed',
        ]
        for pattern in fatal_patterns:
            if pattern.lower() in rec_content.lower():
                issues.append(f"Fatal error detected: '{pattern}' found in record file")

        # 3. Check final parameter file exists
        noptmax = self._get_noptmax()
        if noptmax is not None:
            final_par = self.master_dir / f'{self.project_name}.{noptmax}.par.csv'
            if not final_par.exists():
                issues.append(f"Final parameter file not found: {final_par.name}")
        else:
            issues.append("Could not determine noptmax from control file")

        # 4. Check phi improvement
        phi_data = self._read_phi_data()
        if phi_data is not None and not phi_data.empty:
            if 'mean' in phi_data.columns:
                phi_values = phi_data['mean'].dropna().values
                if len(phi_values) >= 2:
                    if phi_values[-1] >= phi_values[0]:
                        issues.append(
                            f"No phi improvement: {phi_values[0]:.2f} -> {phi_values[-1]:.2f}"
                        )
        else:
            issues.append("Could not read phi data to verify improvement")

        # 5. Check completion message in record
        completion_patterns = [
            'analysis complete',
            'all done',
            'optimization complete',
        ]
        has_completion = any(
            p.lower() in rec_content.lower() for p in completion_patterns
        )
        if not has_completion:
            issues.append("No completion message found in record file")

        success = len(issues) == 0
        return success, issues

    def get_summary(self) -> dict:
        """Extract key metrics from calibration results.

        Returns:
            Dictionary with summary metrics.
        """
        summary = {
            'project': self.project_name,
            'timestamp': datetime.now().isoformat(),
            'pest_dir': str(self.pest_dir),
        }

        # Success status
        success, issues = self.is_successful()
        summary['status'] = 'success' if success else 'failed'
        summary['issues'] = issues

        # Iterations
        noptmax = self._get_noptmax()
        summary['noptmax'] = noptmax

        par_files = self._get_par_files()
        summary['iterations_completed'] = len(par_files) - 1 if par_files else 0

        # Phi evolution
        phi_data = self._read_phi_data()
        if phi_data is not None and not phi_data.empty and 'mean' in phi_data.columns:
            phi_values = phi_data['mean'].dropna().values
            if len(phi_values) >= 1:
                summary['phi_initial'] = float(phi_values[0])
                summary['phi_final'] = float(phi_values[-1])
                if phi_values[0] > 0:
                    reduction = (phi_values[0] - phi_values[-1]) / phi_values[0] * 100
                    summary['phi_reduction_pct'] = float(reduction)
                summary['phi_history'] = [float(v) for v in phi_values]

        # Parameter changes
        if len(par_files) >= 2:
            try:
                initial_params = pd.read_csv(par_files[0], index_col=0)
                final_params = pd.read_csv(par_files[-1], index_col=0)

                # Get mean of ensemble for each parameter
                param_changes = {}
                for col in initial_params.columns:
                    if col in final_params.columns:
                        init_mean = initial_params[col].mean()
                        final_mean = final_params[col].mean()
                        if init_mean != 0:
                            change_pct = (final_mean - init_mean) / abs(init_mean) * 100
                        else:
                            change_pct = 0.0
                        param_changes[col] = {
                            'initial': float(init_mean),
                            'final': float(final_mean),
                            'change_pct': float(change_pct),
                        }
                summary['parameters'] = param_changes
            except Exception as e:
                summary['parameter_error'] = str(e)

        # Timing info from record
        rec_content = self._read_rec_file()
        time_match = re.search(r'took\s+([\d.]+)\s+minutes', rec_content)
        if time_match:
            summary['runtime_minutes'] = float(time_match.group(1))

        return summary

    def get_final_parameters(self) -> pd.DataFrame | None:
        """Get final calibrated parameter ensemble.

        Returns:
            DataFrame with parameter values, or None if not found.
        """
        par_files = self._get_par_files()
        if par_files:
            try:
                return pd.read_csv(par_files[-1], index_col=0)
            except Exception:
                pass
        return None

    def _calculate_dir_size(self, path: Path) -> float:
        """Calculate directory size in MB."""
        total = 0
        if path.exists():
            for f in path.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
        return total / (1024 * 1024)

    def cleanup(
        self,
        archive_dir: str | None = None,
        keep_debug: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Clean up calibration files based on success status.

        Args:
            archive_dir: Directory to archive important files (None = pest_dir/archive).
            keep_debug: Force keeping debug files even on success.
            dry_run: If True, report what would be done without doing it.

        Returns:
            Dictionary with cleanup report.
        """
        success, issues = self.is_successful()
        report = {
            'success': success,
            'dry_run': dry_run,
            'files_archived': [],
            'files_deleted': [],
            'dirs_deleted': [],
            'space_recovered_mb': 0.0,
        }

        # Determine archive directory
        if archive_dir is None:
            archive_path = self.pest_dir / 'archive'
        else:
            archive_path = Path(archive_dir)

        if not dry_run:
            archive_path.mkdir(parents=True, exist_ok=True)

        # Always archive key files
        for pattern in self.ARCHIVE_FILES:
            filename = pattern.format(project=self.project_name)
            # Check both pest_dir and master_dir
            for search_dir in [self.pest_dir, self.master_dir]:
                src = search_dir / filename
                if src.exists():
                    dst = archive_path / src.name
                    if not dry_run:
                        shutil.copy2(src, dst)
                    report['files_archived'].append(str(src.name))
                    break

        # Archive final parameter file
        par_files = self._get_par_files()
        if par_files:
            final_par = par_files[-1]
            if not dry_run:
                shutil.copy2(final_par, archive_path / final_par.name)
            report['files_archived'].append(final_par.name)

        # Save summary
        summary = self.get_summary()
        summary_file = 'calibration_summary.json' if success else 'calibration_failure.json'
        if not dry_run:
            with open(archive_path / summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        report['files_archived'].append(summary_file)

        # Decide what to clean up
        if success and not keep_debug:
            # Delete workers directory
            if self.workers_dir.exists():
                size = self._calculate_dir_size(self.workers_dir)
                report['space_recovered_mb'] += size
                if not dry_run:
                    shutil.rmtree(self.workers_dir)
                report['dirs_deleted'].append(str(self.workers_dir))

            # Delete intermediate files in master
            for pattern in self.CLEANUP_PATTERNS:
                for f in self.master_dir.glob(pattern):
                    size = f.stat().st_size / (1024 * 1024)
                    report['space_recovered_mb'] += size
                    if not dry_run:
                        f.unlink()
                    report['files_deleted'].append(f.name)

            # Delete non-final par/obs files (keep 0 and final)
            noptmax = self._get_noptmax()
            if noptmax and noptmax > 1:
                for i in range(1, noptmax):
                    for suffix in ['par.csv', 'obs.csv', 'pdc.csv', 'pcs.csv']:
                        f = self.master_dir / f'{self.project_name}.{i}.{suffix}'
                        if f.exists():
                            size = f.stat().st_size / (1024 * 1024)
                            report['space_recovered_mb'] += size
                            if not dry_run:
                                f.unlink()
                            report['files_deleted'].append(f.name)

        else:
            # Failure or keep_debug: preserve everything
            report['debug_preserved'] = True
            if issues:
                report['failure_reasons'] = issues
                report['recommendations'] = self._get_recommendations(issues)

        return report

    def _get_recommendations(self, issues: list[str]) -> list[str]:
        """Generate debugging recommendations based on issues."""
        recommendations = []

        for issue in issues:
            if 'record file not found' in issue.lower():
                recommendations.append(
                    "PEST++ may not have started. Check that pestpp-ies is in PATH."
                )
            elif 'fatal error' in issue.lower():
                recommendations.append(
                    "Check panther_master.rec and worker logs for error details."
                )
            elif 'parameter file not found' in issue.lower():
                recommendations.append(
                    "Calibration may have crashed mid-run. Check worker directories."
                )
            elif 'no phi improvement' in issue.lower():
                recommendations.append(
                    "Try increasing realizations or adjusting parameter bounds."
                )
            elif 'traceback' in issue.lower():
                recommendations.append(
                    "Python error in forward model. Check custom_forward_run.py."
                )

        if not recommendations:
            recommendations.append(
                "Check worker_0/panther_worker.rec for detailed error messages."
            )
            recommendations.append(
                "Verify observation files exist in obs/ directory."
            )

        return recommendations

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print(f"PEST++ Calibration Results: {summary['project']}")
        print(f"{'='*60}")

        status = summary.get('status', 'unknown')
        if status == 'success':
            print(f"Status: SUCCESS")
        else:
            print(f"Status: FAILED")
            for issue in summary.get('issues', []):
                print(f"  - {issue}")

        print(f"\nIterations: {summary.get('iterations_completed', '?')}/{summary.get('noptmax', '?')}")

        if 'phi_initial' in summary:
            print(f"\nPhi (objective function):")
            print(f"  Initial: {summary['phi_initial']:.2f}")
            print(f"  Final:   {summary['phi_final']:.2f}")
            if 'phi_reduction_pct' in summary:
                print(f"  Reduction: {summary['phi_reduction_pct']:.1f}%")

        if 'runtime_minutes' in summary:
            print(f"\nRuntime: {summary['runtime_minutes']:.2f} minutes")

        if 'parameters' in summary:
            print(f"\nParameter changes (ensemble mean):")
            params = summary['parameters']
            # Group by parameter type
            param_groups = {}
            for name, vals in params.items():
                # Extract base parameter name (e.g., 'aw' from 'aw_US-FPe')
                base = name.split('_')[0]
                if base not in param_groups:
                    param_groups[base] = []
                param_groups[base].append((name, vals))

            for base, items in sorted(param_groups.items()):
                for name, vals in items:
                    print(f"  {name}: {vals['initial']:.3f} -> {vals['final']:.3f} "
                          f"({vals['change_pct']:+.1f}%)")

        print(f"{'='*60}\n")


def main() -> int:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='PEST++ results cleanup and summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('pest_dir', help='Path to pest/ directory')
    parser.add_argument('--project', '-p', default=None,
                        help='Project name (auto-detected from .pst file if not provided)')
    parser.add_argument('--archive', '-a', default=None,
                        help='Archive directory (default: pest_dir/archive)')
    parser.add_argument('--check-only', action='store_true',
                        help='Check status and print summary without cleanup')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be cleaned without doing it')
    parser.add_argument('--keep-debug', action='store_true',
                        help='Keep debug files even on success')
    parser.add_argument('--json', action='store_true',
                        help='Output summary as JSON')
    args = parser.parse_args()

    pest_dir = Path(args.pest_dir)
    if not pest_dir.exists():
        print(f"Error: Directory not found: {pest_dir}")
        return 1

    # Auto-detect project name from .pst file
    project_name = args.project
    if project_name is None:
        pst_files = list(pest_dir.glob('*.pst'))
        if pst_files:
            project_name = pst_files[0].stem
        else:
            print("Error: Could not detect project name. Use --project option.")
            return 1

    results = PestResults(pest_dir, project_name)

    if args.json:
        summary = results.get_summary()
        print(json.dumps(summary, indent=2))
        return 0

    # Print summary
    results.print_summary()

    if args.check_only:
        success, _ = results.is_successful()
        return 0 if success else 1

    # Cleanup
    print("Cleaning up...")
    report = results.cleanup(
        archive_dir=args.archive,
        keep_debug=args.keep_debug,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n[DRY RUN - no changes made]")

    print(f"\nArchived {len(report['files_archived'])} files")
    if report['files_archived']:
        for f in report['files_archived'][:5]:
            print(f"  + {f}")
        if len(report['files_archived']) > 5:
            print(f"  ... and {len(report['files_archived']) - 5} more")

    if report['files_deleted'] or report['dirs_deleted']:
        print(f"\nDeleted {len(report['files_deleted'])} files, "
              f"{len(report['dirs_deleted'])} directories")
        print(f"Space recovered: {report['space_recovered_mb']:.1f} MB")

    if report.get('debug_preserved'):
        print("\nDebug files preserved for troubleshooting.")
        if 'recommendations' in report:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")

    return 0


if __name__ == '__main__':
    exit(main())
