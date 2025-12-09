#!/usr/bin/env python3
"""
Python script to run 10 repeated TrivialAugment experiments sequentially.
Each experiment is an independent run with the same configuration but different random seed.

Usage: python run_experiments.py
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration for repeated experiments
CONFIG_PATH = "confs/wrn40x2/wresnet40x2_cifar10_b128_maxlr.1_ua_uasesp_nowarmup_200epochs.yaml"
EXPERIMENT_NAME = "WRN-40-2 UA (ua)"
NUM_RUNS = 10


def run_experiment(run_num):
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"[{run_num}/{NUM_RUNS}] Running: {EXPERIMENT_NAME} - Run #{run_num}")
    print(f"Config: {CONFIG_PATH}")
    print(f"{'='*70}\n")
    
    # Prepare command
    local_dir = Path("wrn_40x2")
    local_dir.mkdir(exist_ok=True)
    
    tag = f"expUAua_wrn40x2_{run_num}"
    checkpoint_path = local_dir / f"expUAua_wrn40x2_{run_num}.pth"
    
    cmd = [
        sys.executable, "-m",
        "TrivialAugment.train",
        "-c", CONFIG_PATH,
        "--dataroot", "data",
        "--tag", tag,
        "--save", str(checkpoint_path)
    ]
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n✓ Run {run_num}/{NUM_RUNS} completed successfully!")
        print(f"  Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"  Checkpoint: {checkpoint_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Run {run_num}/{NUM_RUNS} FAILED!")
        print(f"  Error code: {e.returncode}")
        print(f"  Time before failure: {elapsed:.0f}s")
        return False


def main():
    """Main function to run all experiments."""
    print("="*70)
    print("TrivialAugment Repeated Experiment Runner")
    print("="*70)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Total runs: {NUM_RUNS}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Track results
    results = []
    start_time = time.time()
    
    # Run experiments sequentially
    for run_num in range(1, NUM_RUNS + 1):
        success = run_experiment(run_num)
        results.append((run_num, success))
        
        # If experiment failed, ask user what to do
        if not success:
            print("\nExperiment failed. Options:")
            print("  1. Continue with next run")
            print("  2. Stop batch run")
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "2":
                print("\nStopping batch run...")
                break
    
    # Print summary
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*70)
    print("BATCH RUN SUMMARY")
    print("="*70)
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"\nResults:")
    
    success_count = 0
    for run_num, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: Run #{run_num}")
        if success:
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(results)} runs")
    print("="*70)
    
    if success_count == NUM_RUNS:
        print("\n🎉 ALL RUNS COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  {len(results) - success_count} run(s) failed or were skipped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user.")
        sys.exit(1)
