#!/usr/bin/env python3
"""
Run multiple experiments with different configuration variations
Generates all combinations of USE_TRANSFORMER, USE_AUGMENTATION, USE_PATCH_TRAINING, 
USE_CLASS_SAMPLING, and LOSS_TYPE, runs the pipeline for each, and saves results.
"""

import os
import sys
import subprocess
import shutil
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


def create_experiment_config(base_config_path, output_config_path, config_overrides, experiment_dir):
    """
    Create a modified config file with the specified overrides
    
    Args:
        base_config_path: Path to the original config.py
        output_config_path: Path where the new config will be saved
        config_overrides: Dictionary of config variable overrides
        experiment_dir: Directory for this experiment (for results/checkpoints)
    """
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # Apply overrides
    import re
    for key, value in config_overrides.items():
        # Handle boolean values
        if isinstance(value, bool):
            value_str = 'True' if value else 'False'
        # Handle string values
        elif isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        
        # Replace the config line
        # Match patterns like: USE_TRANSFORMER = True
        pattern = f"({key})\\s*=\\s*[^\\n]+"
        replacement = f"\\1 = {value_str}"
        
        if re.search(pattern, config_content):
            config_content = re.sub(pattern, replacement, config_content)
        else:
            # If not found, add it at the end of the file
            config_content += f"\n{replacement}\n"
    
    # Override RESULTS_DIR and CHECKPOINT_DIR to be experiment-specific
    # Use os.path.join with proper escaping - need to handle the path correctly
    # experiment_dir is relative to BASE_DIR
    results_dir_override = f"RESULTS_DIR = os.path.join(BASE_DIR, '{experiment_dir}', 'results')"
    checkpoint_dir_override = f"CHECKPOINT_DIR = os.path.join(BASE_DIR, '{experiment_dir}', 'checkpoints')"
    
    # Replace existing directory definitions
    # Match: RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    config_content = re.sub(
        r"RESULTS_DIR\s*=\s*os\.path\.join\([^)]+\)",
        results_dir_override,
        config_content
    )
    config_content = re.sub(
        r"CHECKPOINT_DIR\s*=\s*os\.path\.join\([^)]+\)",
        checkpoint_dir_override,
        config_content
    )
    
    # Write the modified config
    with open(output_config_path, 'w') as f:
        f.write(config_content)


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
    
    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create temporary config file
    temp_config_path = os.path.join(experiment_dir, 'config.py')
    base_config_path = os.path.join(base_dir, 'config.py')
    
    # Get relative path from base_dir to experiment_dir for config
    # Use forward slashes for Python path compatibility
    rel_experiment_dir = os.path.relpath(experiment_dir, base_dir).replace('\\', '/')
    
    # Read base config and create modified version
    create_experiment_config(base_config_path, temp_config_path, config_overrides, rel_experiment_dir)
    
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
    
    # Backup original config and use temporary one
    original_config_backup = os.path.join(base_dir, 'config.py.backup')
    if not os.path.exists(original_config_backup):
        shutil.copy2(base_config_path, original_config_backup)
    
    # Temporarily replace config.py
    shutil.copy2(temp_config_path, base_config_path)
    
    # Update results and checkpoints directories in config to be experiment-specific
    # We'll need to modify the run script or handle this differently
    # For now, we'll copy results after the run
    
    try:
        # Run the pipeline
        log_file = os.path.join(experiment_dir, 'pipeline.log')
        script_path = os.path.join(base_dir, 'run_full_pipeline.sh')
        
        print(f"\n{Colors.YELLOW}Starting pipeline...{Colors.NC}")
        print(f"Log file: {log_file}")
        
        with open(log_file, 'w') as log:
            result = subprocess.run(
                ['bash', script_path],
                cwd=base_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        # Restore original config
        shutil.copy2(original_config_backup, base_config_path)
        
        if result.returncode == 0:
            # Results and checkpoints should already be in experiment_dir
            # due to config override, but verify and move if needed
            results_src = os.path.join(base_dir, 'results')
            results_dst = os.path.join(experiment_dir, 'results')
            
            checkpoints_src = os.path.join(base_dir, 'checkpoints')
            checkpoints_dst = os.path.join(experiment_dir, 'checkpoints')
            
            # If results were created in base_dir (fallback), move them
            if os.path.exists(results_src) and not os.path.exists(results_dst):
                shutil.move(results_src, results_dst)
            elif os.path.exists(results_src) and os.path.exists(results_dst):
                # Merge or replace - for safety, we'll keep the experiment-specific one
                shutil.rmtree(results_src)
            
            if os.path.exists(checkpoints_src) and not os.path.exists(checkpoints_dst):
                shutil.move(checkpoints_src, checkpoints_dst)
            elif os.path.exists(checkpoints_src) and os.path.exists(checkpoints_dst):
                shutil.rmtree(checkpoints_src)
            
            metadata['status'] = 'completed'
            metadata['return_code'] = 0
            print(f"{Colors.GREEN}✓ Experiment {exp_name} completed successfully!{Colors.NC}")
        else:
            metadata['status'] = 'failed'
            metadata['return_code'] = result.returncode
            print(f"{Colors.RED}✗ Experiment {exp_name} failed with return code {result.returncode}{Colors.NC}")
        
    except Exception as e:
        # Restore original config
        if os.path.exists(original_config_backup):
            shutil.copy2(original_config_backup, base_config_path)
        
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

