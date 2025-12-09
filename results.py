import glob
import torch
import pandas as pd
import numpy as np
from scipy import stats

# Define augmentation methods and their prefixes
aug_methods = {
    'FAA (RA)': 'expFAA',
    'UA (UA)': 'expUAua',
    'UA (RA)': 'expUAra',
    'TA (RA)': 'expTAra',
    'TA (Wide)': 'expTAwide'
}

def compute_95_ci(data):
    """Compute 95% confidence interval using t-distribution."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(n)  # Standard error
    # t critical value for 95% CI with n-1 degrees of freedom
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci = t_crit * se
    return mean, ci

def process_checkpoint_dir(checkpoint_dir, model_name, paper_results=None):
    """Process all checkpoints in a directory and print results."""
    drive_checkpoints = glob.glob(f'{checkpoint_dir}/*.pth')
    
    # Collect results grouped by augmentation method
    results_by_method = {method: [] for method in aug_methods.keys()}
    all_results = []

    for ckpt_path in drive_checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'log' in ckpt and 'test' in ckpt['log']:
                test_acc = ckpt['log']['test']['top1']
                test_error = 1 - test_acc
                epoch = ckpt.get('epoch', 'N/A')
                filename = ckpt_path.split('/')[-1].split('\\')[-1]
                
                # Determine which method this belongs to
                for method_name, prefix in aug_methods.items():
                    if filename.startswith(prefix):
                        results_by_method[method_name].append(test_acc * 100)
                        break
                
                all_results.append({
                    'Checkpoint': filename,
                    'Epoch': epoch,
                    'Test Accuracy (%)': f'{test_acc * 100:.2f}',
                    'Test Error (%)': f'{test_error * 100:.2f}'
                })
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        df_sorted = df.sort_values('Checkpoint')
        
        print("\n" + "="*80)
        print(f"INDIVIDUAL RESULTS - {model_name} on CIFAR-10")
        print("="*80)
        print(df_sorted.to_string(index=False))
        
        print("\n" + "="*80)
        print(f"AGGREGATED RESULTS - {model_name} (Mean ± 95% CI)")
        print("="*80)
        print(f"{'Method':<15} {'n':>5} {'Mean Acc (%)':>15} {'95% CI':>12} {'Result':>20}")
        print("-"*80)
        
        for method_name in aug_methods.keys():
            accs = results_by_method[method_name]
            if len(accs) > 0:
                mean, ci = compute_95_ci(accs)
                print(f"{method_name:<15} {len(accs):>5} {mean:>15.2f} {'±':>5} {ci:>6.2f} {mean:>12.2f} ± {ci:.2f}")
            else:
                print(f"{method_name:<15} {'N/A':>5} {'N/A':>15} {'N/A':>12}")
        
        if paper_results:
            print("\n" + "="*80)
            print(f"Expected Results from Paper ({model_name}):")
            for method, result in paper_results.items():
                print(f"  {method}:      {result}")
            print("="*80)
        
        return results_by_method
    else:
        print(f"No checkpoints found in {checkpoint_dir}!")
        return None

# ============================================================================
# WRN-40-2 Results
# ============================================================================
paper_results_wrn40x2 = {
    'FAA (RA)': '96.39 ± 0.06',
    'UA (UA)': '96.42 ± 0.04',
    'UA (RA)': '96.45 ± 0.06',
    'TA (RA)': '96.62 ± 0.09',
    'TA (Wide)': '96.32 ± 0.05'
}

print("\n" + "#"*80)
print("#" + " "*30 + "WRN-40-2 EXPERIMENTS" + " "*28 + "#")
print("#"*80)
results_wrn40x2 = process_checkpoint_dir('wrn_40x2', 'WRN-40-2', paper_results_wrn40x2)

# ============================================================================
# WRN-28-2 Results
# ============================================================================
# Note: Paper may not have WRN-28-2 results, so we leave paper_results as None
# If paper has these results, add them here:
paper_results_wrn28x2 = None  # Update if paper has WRN-28-2 results

print("\n" + "#"*80)
print("#" + " "*30 + "WRN-28-2 EXPERIMENTS" + " "*28 + "#")
print("#"*80)
results_wrn28x2 = process_checkpoint_dir('wrn_28x2', 'WRN-28-2', paper_results_wrn28x2)

# ============================================================================
# Comparison Summary
# ============================================================================
if results_wrn40x2 and results_wrn28x2:
    print("\n" + "#"*80)
    print("#" + " "*25 + "COMPARISON: WRN-40-2 vs WRN-28-2" + " "*20 + "#")
    print("#"*80)
    print(f"\n{'Method':<15} {'WRN-40-2':>20} {'WRN-28-2':>20} {'Difference':>15}")
    print("-"*80)
    
    for method_name in aug_methods.keys():
        accs_40 = results_wrn40x2[method_name]
        accs_28 = results_wrn28x2[method_name]
        
        if len(accs_40) > 0 and len(accs_28) > 0:
            mean_40, ci_40 = compute_95_ci(accs_40)
            mean_28, ci_28 = compute_95_ci(accs_28)
            diff = mean_40 - mean_28
            print(f"{method_name:<15} {mean_40:>12.2f} ± {ci_40:.2f} {mean_28:>12.2f} ± {ci_28:.2f} {diff:>+14.2f}")
    
    print("="*80)