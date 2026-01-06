#!/usr/bin/env python3
"""
Run multiple experiments with different configuration variations
Generates all combinations of USE_TRANSFORMER, USE_AUGMENTATION, USE_PATCH_TRAINING, 
USE_CLASS_SAMPLING, and LOSS_TYPE, runs the pipeline for each, and saves results.
"""

import os
import sys
import subprocess
import json
from itertools import product
from datetime import datetime
import argparse

# Add current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color codes for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


# Removed - no longer creating config files


def get_experiment_name(config):
    """Generate a descriptive name for the experiment"""
    parts = []
    parts.append("T" if config['USE_TRANSFORMER'] else "NoT")
    parts.append("A" if config['USE_AUGMENTATION'] else "NoA")
    parts.append("P" if config['USE_PATCH_TRAINING'] else "NoP")
    parts.append("S" if config['USE_CLASS_SAMPLING'] else "NoS")
    parts.append(config['LOSS_TYPE'][:3].upper())  # First 3 chars of loss type
    return "_".join(parts)


def run_experiment(experiment_dir, config_overrides, base_dir):
    """
    Run a single experiment with the given configuration
    
    Args:
        experiment_dir: Directory to save experiment results
        config_overrides: Dictionary of config overrides
        base_dir: Base directory of the project
    
    Returns:
        dict: Experiment metadata and status
    """
    exp_name = get_experiment_name(config_overrides)
    print(f"\n{Colors.CYAN}{'='*80}{Colors.NC}")
    print(f"{Colors.CYAN}Running Experiment: {exp_name}{Colors.NC}")
    print(f"{Colors.CYAN}{'='*80}{Colors.NC}")
    print(f"{Colors.BLUE}Configuration:{Colors.NC}")
    for key, value in config_overrides.items():
        print(f"  {key}: {value}")
    
    # Create experiment directory (all outputs go here, nothing in root)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment metadata
    metadata = {
        'experiment_name': exp_name,
        'config': config_overrides,
        'timestamp': datetime.now().isoformat(),
        'status': 'running'
    }
    
    metadata_path = os.path.join(experiment_dir, 'experiment_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save config overrides as JSON for the wrapper script
    config_json_path = os.path.join(experiment_dir, '_config_overrides.json')
    with open(config_json_path, 'w') as f:
        json.dump(config_overrides, f, indent=2)
    
    try:
        # Run the pipeline using wrapper script that doesn't modify config.py
        log_file = os.path.join(experiment_dir, 'pipeline.log')
        
        print(f"\n{Colors.YELLOW}Starting pipeline...{Colors.NC}")
        print(f"Log file: {log_file}")
        
        # Use the wrapper script that overrides config at runtime
        wrapper_script = os.path.join(base_dir, 'run_experiment_with_config.py')
        
        with open(log_file, 'w') as log:
            result = subprocess.run(
                [sys.executable, wrapper_script, config_json_path, experiment_dir],
                cwd=base_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        # Clean up temporary config JSON
        if os.path.exists(config_json_path):
            os.remove(config_json_path)
        
        if result.returncode == 0:
            metadata['status'] = 'completed'
            metadata['return_code'] = 0
            print(f"{Colors.GREEN}✓ Experiment {exp_name} completed successfully!{Colors.NC}")
        else:
            metadata['status'] = 'failed'
            metadata['return_code'] = result.returncode
            print(f"{Colors.RED}✗ Experiment {exp_name} failed with return code {result.returncode}{Colors.NC}")
        
    except Exception as e:
        # Clean up temporary files
        if os.path.exists(config_json_path):
            os.remove(config_json_path)
        
        metadata['status'] = 'error'
        metadata['error'] = str(e)
        print(f"{Colors.RED}✗ Experiment {exp_name} encountered an error: {e}{Colors.NC}")
    
    # Update metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def generate_all_combinations():
    """Generate all combinations of configuration parameters"""
    use_transformer = [True, False]
    use_augmentation = [True, False]
    use_patch_training = [True, False]
    use_class_sampling = [True, False]
    loss_types = ['bce', 'dice', 'focal', 'combined', 'weighted_combined']
    
    combinations = list(product(
        use_transformer,
        use_augmentation,
        use_patch_training,
        use_class_sampling,
        loss_types
    ))
    
    configs = []
    for combo in combinations:
        config = {
            'USE_TRANSFORMER': combo[0],
            'USE_AUGMENTATION': combo[1],
            'USE_PATCH_TRAINING': combo[2],
            'USE_CLASS_SAMPLING': combo[3],
            'LOSS_TYPE': combo[4]
        }
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple experiments with different configurations'
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='Directory to save all experiment results (default: experiments)'
    )
    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='Skip experiments that have already been completed'
    )
    parser.add_argument(
        '--max-experiments',
        type=int,
        default=None,
        help='Maximum number of experiments to run (for testing)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        nargs='+',
        help='Filter experiments by name pattern (e.g., --filter T_A_P)'
    )
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(base_dir, args.experiments_dir)
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Generate all combinations
    all_configs = generate_all_combinations()
    
    # Apply filters if specified
    if args.filter:
        filtered_configs = []
        for config in all_configs:
            exp_name = get_experiment_name(config)
            if any(f in exp_name for f in args.filter):
                filtered_configs.append(config)
        all_configs = filtered_configs
    
    # Limit number of experiments if specified
    if args.max_experiments:
        all_configs = all_configs[:args.max_experiments]
    
    print(f"{Colors.BLUE}{'='*80}{Colors.NC}")
    print(f"{Colors.BLUE}MS Detection - Experiment Runner{Colors.NC}")
    print(f"{Colors.BLUE}{'='*80}{Colors.NC}")
    print(f"\nTotal experiments to run: {len(all_configs)}")
    print(f"Experiments directory: {experiments_dir}\n")
    
    # Run each experiment
    results = []
    for i, config in enumerate(all_configs, 1):
        exp_name = get_experiment_name(config)
        experiment_dir = os.path.join(experiments_dir, exp_name)
        
        # Check if already completed
        if args.skip_completed:
            metadata_path = os.path.join(experiment_dir, 'experiment_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
                    if existing_metadata.get('status') == 'completed':
                        print(f"{Colors.YELLOW}[{i}/{len(all_configs)}] Skipping {exp_name} (already completed){Colors.NC}")
                        results.append(existing_metadata)
                        continue
        
        print(f"\n{Colors.MAGENTA}[{i}/{len(all_configs)}] Processing experiment...{Colors.NC}")
        metadata = run_experiment(experiment_dir, config, base_dir)
        results.append(metadata)
    
    # Save summary
    summary_path = os.path.join(experiments_dir, 'experiments_summary.json')
    summary = {
        'total_experiments': len(all_configs),
        'completed': sum(1 for r in results if r.get('status') == 'completed'),
        'failed': sum(1 for r in results if r.get('status') == 'failed'),
        'errors': sum(1 for r in results if r.get('status') == 'error'),
        'experiments': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{Colors.BLUE}{'='*80}{Colors.NC}")
    print(f"{Colors.BLUE}EXPERIMENTS SUMMARY{Colors.NC}")
    print(f"{Colors.BLUE}{'='*80}{Colors.NC}")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"{Colors.GREEN}Completed: {summary['completed']}{Colors.NC}")
    print(f"{Colors.RED}Failed: {summary['failed']}{Colors.NC}")
    if summary['errors'] > 0:
        print(f"{Colors.RED}Errors: {summary['errors']}{Colors.NC}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"{Colors.YELLOW}\nNext step: Run compare_experiments.py to analyze and compare all results{Colors.NC}")
    print(f"{Colors.BLUE}{'='*80}{Colors.NC}\n")


if __name__ == '__main__':
    main()

