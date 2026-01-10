#!/usr/bin/env python3
"""
Script to delete all run directories from data/runs/.

This script removes all run directories (e.g., 20260110_124236_run/) 
and optionally clears the last_run.txt marker file.
"""

import argparse
import shutil
from pathlib import Path


def delete_all_runs(runs_dir: Path, dry_run: bool = False, keep_last_run_file: bool = False) -> None:
    """
    Delete all run directories from the runs directory.
    
    Parameters
    ----------
    runs_dir : Path
        Path to the runs directory (e.g., data/runs)
    dry_run : bool
        If True, only print what would be deleted without actually deleting
    keep_last_run_file : bool
        If True, keep the last_run.txt file; otherwise delete it too
    """
    if not runs_dir.exists():
        print(f"Error: Directory {runs_dir} does not exist.")
        return
    
    # Find all run directories (directories, not files)
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    if not run_dirs:
        print(f"No run directories found in {runs_dir}")
        return
    
    # Sort for consistent output
    run_dirs.sort()
    
    print(f"Found {len(run_dirs)} run directories:")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")
    
    if dry_run:
        print("\n[DRY RUN] Would delete the above directories.")
        if not keep_last_run_file:
            last_run_file = runs_dir / "last_run.txt"
            if last_run_file.exists():
                print(f"[DRY RUN] Would also delete {last_run_file}")
        return
    
    # Confirm deletion
    response = input(f"\nAre you sure you want to delete all {len(run_dirs)} run directories? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Deletion cancelled.")
        return
    
    # Delete each run directory
    deleted_count = 0
    failed_count = 0
    
    for run_dir in run_dirs:
        try:
            shutil.rmtree(run_dir)
            print(f"Deleted: {run_dir.name}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {run_dir.name}: {e}")
            failed_count += 1
    
    # Handle last_run.txt
    if not keep_last_run_file:
        last_run_file = runs_dir / "last_run.txt"
        if last_run_file.exists():
            try:
                last_run_file.unlink()
                print(f"Deleted: {last_run_file.name}")
            except Exception as e:
                print(f"Error deleting {last_run_file.name}: {e}")
    
    print(f"\nSummary: {deleted_count} directories deleted, {failed_count} failed.")


def main():
    parser = argparse.ArgumentParser(
        description="Delete all run directories from data/runs/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be deleted (dry run)
  python delete_runs.py --dry-run
  
  # Delete all runs (with confirmation prompt)
  python delete_runs.py
  
  # Delete all runs but keep last_run.txt
  python delete_runs.py --keep-last-run-file
  
  # Delete all runs without confirmation (use with caution!)
  python delete_runs.py --yes
        """
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("data/runs"),
        help="Path to the runs directory (default: data/runs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--keep-last-run-file",
        action="store_true",
        help="Keep the last_run.txt file when deleting runs"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)"
    )
    
    args = parser.parse_args()
    
    # If --yes is used, we need to bypass the confirmation
    # We'll modify the function to accept a skip_confirm parameter
    if args.yes:
        # Delete without confirmation
        runs_dir = args.runs_dir
        if not runs_dir.exists():
            print(f"Error: Directory {runs_dir} does not exist.")
            return
        
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            print(f"No run directories found in {runs_dir}")
            return
        
        print(f"Deleting {len(run_dirs)} run directories (--yes flag used, no confirmation)...")
        deleted_count = 0
        failed_count = 0
        
        for run_dir in run_dirs:
            try:
                shutil.rmtree(run_dir)
                print(f"Deleted: {run_dir.name}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {run_dir.name}: {e}")
                failed_count += 1
        
        if not args.keep_last_run_file:
            last_run_file = runs_dir / "last_run.txt"
            if last_run_file.exists():
                try:
                    last_run_file.unlink()
                    print(f"Deleted: {last_run_file.name}")
                except Exception as e:
                    print(f"Error deleting {last_run_file.name}: {e}")
        
        print(f"\nSummary: {deleted_count} directories deleted, {failed_count} failed.")
    else:
        delete_all_runs(args.runs_dir, args.dry_run, args.keep_last_run_file)


if __name__ == "__main__":
    main()
