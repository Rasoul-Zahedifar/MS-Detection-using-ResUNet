#!/usr/bin/env python3
"""
Wrapper script to run pipeline with config overrides
This script modifies config values at runtime without touching config.py
"""

import os
import sys
import subprocess

# Add current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_with_config_overrides(config_overrides, experiment_dir):
    """
    Run the pipeline with config overrides using environment variables
    
    Args:
        config_overrides: Dictionary of config variable overrides
        experiment_dir: Directory for this experiment
    """
    # Set environment variables for config overrides
    env = os.environ.copy()
    
    # Convert config overrides to environment variables
    for key, value in config_overrides.items():
        if isinstance(value, bool):
            env[f'CONFIG_{key}'] = 'True' if value else 'False'
        elif isinstance(value, str):
            env[f'CONFIG_{key}'] = value
        else:
            env[f'CONFIG_{key}'] = str(value)
    
    # Set experiment-specific directories
    results_dir = os.path.join(experiment_dir, 'results')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    env['CONFIG_RESULTS_DIR'] = results_dir
    env['CONFIG_CHECKPOINT_DIR'] = checkpoint_dir
    
    # Create a Python script that will override config and run the pipeline
    # Build config_overrides dict
    config_dict_str = "{\n"
    for key, value in config_overrides.items():
        if isinstance(value, bool):
            config_dict_str += f"    '{key}': {value},\n"
        elif isinstance(value, str):
            # Escape single quotes in strings
            escaped_value = value.replace("'", "\\'")
            config_dict_str += f"    '{key}': '{escaped_value}',\n"
        else:
            config_dict_str += f"    '{key}': {value},\n"
    config_dict_str += "}"
    
    # Escape paths for use in string
    results_dir_escaped = results_dir.replace("\\", "\\\\")
    checkpoint_dir_escaped = checkpoint_dir.replace("\\", "\\\\")
    base_dir_escaped = os.path.dirname(os.path.abspath(__file__)).replace("\\", "\\\\")
    
    wrapper_script = f"""import os
import sys

# Add base directory to path so we can import config
base_dir = r'{base_dir_escaped}'
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Override config values before importing config
config_overrides = {config_dict_str}

# Import and override config
import config
for key, value in config_overrides.items():
    setattr(config, key, value)

# Override directories
config.RESULTS_DIR = r'{results_dir_escaped}'
config.CHECKPOINT_DIR = r'{checkpoint_dir_escaped}'
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

# Now run the pipeline steps
import subprocess

# base_dir is already set above

# Step 1: Training
print("="*80)
print("STEP 1: Training")
print("="*80)
result = subprocess.run([
    sys.executable, 'main.py', '--mode', 'train',
    '--batch-size', str(config.BATCH_SIZE),
    '--epochs', str(config.NUM_EPOCHS),
    '--lr', str(config.LEARNING_RATE)
], cwd=base_dir)
if result.returncode != 0:
    sys.exit(result.returncode)

# Step 2: Evaluate best_by_dice
print("\\n" + "="*80)
print("STEP 2: Evaluating best_by_dice")
print("="*80)
checkpoint_path_dice = os.path.join(config.CHECKPOINT_DIR, 'best_by_dice.pth')
if os.path.exists(checkpoint_path_dice):
    result = subprocess.run([
        sys.executable, 'main.py', '--mode', 'evaluate',
        '--checkpoint', checkpoint_path_dice, '--split', 'test'
    ], cwd=base_dir)
    if result.returncode != 0:
        print("Warning: Evaluation of best_by_dice failed")
else:
    print(f"Warning: {{checkpoint_path_dice}} not found")

# Step 3: Evaluate best_by_loss (optional)
print("\\n" + "="*80)
print("STEP 3: Evaluating best_by_loss")
print("="*80)
checkpoint_path_loss = os.path.join(config.CHECKPOINT_DIR, 'best_by_loss.pth')
if os.path.exists(checkpoint_path_loss):
    result = subprocess.run([
        sys.executable, 'main.py', '--mode', 'evaluate',
        '--checkpoint', checkpoint_path_loss, '--split', 'test'
    ], cwd=base_dir)
    if result.returncode != 0:
        print("Warning: Evaluation of best_by_loss failed")
else:
    print(f"Warning: {{checkpoint_path_loss}} not found, skipping")

# Step 4: Analyze results
print("\\n" + "="*80)
print("STEP 4: Analyzing results")
print("="*80)
result = subprocess.run([
    sys.executable, 'main.py', '--mode', 'analyze'
], cwd=base_dir)
if result.returncode != 0:
    print("Warning: Analysis failed")

print("\\n" + "="*80)
print("PIPELINE COMPLETED")
print("="*80)
"""
    
    # Write wrapper script to a temporary file in experiment directory
    temp_script = os.path.join(experiment_dir, '_run_pipeline_temp.py')
    with open(temp_script, 'w') as f:
        f.write(wrapper_script)
    
    try:
        # Run the wrapper script
        result = subprocess.run(
            [sys.executable, temp_script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env
        )
        return result.returncode
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)


if __name__ == '__main__':
    import json
    
    if len(sys.argv) < 3:
        print("Usage: run_experiment_with_config.py <config_json> <experiment_dir>")
        sys.exit(1)
    
    config_json = sys.argv[1]
    experiment_dir = sys.argv[2]
    
    with open(config_json, 'r') as f:
        config_overrides = json.load(f)
    
    return_code = run_with_config_overrides(config_overrides, experiment_dir)
    sys.exit(return_code)

