import re
import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt

# Assume the output was piped to this file:
# !python train_unsloth.py > unsloth_train.log
ROOT_DIR = Path(__file__).resolve().parents[1]
log_file = ROOT_DIR / 'episode_traces' / 'sft_train.log'
output_file = ROOT_DIR / 'assets' / 'unsloth_train_loss_epoch.png'

runs = []
current_epochs = []
current_losses = []

if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        for line in f:
            # Look for lines containing both 'loss' and 'epoch' keys
            if "{'loss':" in line and "'epoch':" in line:
                try:
                    # Extract the dictionary-like string
                    match = re.search(r"(\{.*?'loss':.*?'epoch':.*?\})", line)
                    if match:
                        # Safely evaluate the string into a Python dictionary
                        data = ast.literal_eval(match.group(1))
                        ep = float(data['epoch'])
                        ls = float(data['loss'])
                        
                        # If epoch drops, a new training run has started
                        if current_epochs and ep < current_epochs[-1]:
                            runs.append((current_epochs, current_losses))
                            current_epochs = []
                            current_losses = []
                            
                        current_epochs.append(ep)
                        current_losses.append(ls)
                except Exception as e:
                    pass
                    
    # Add the last run
    if current_epochs:
        runs.append((current_epochs, current_losses))
    
    if runs:
        plt.figure(figsize=(10, 6))
        for i, (eps, lss) in enumerate(runs):
            plt.plot(eps, lss, marker='o', linestyle='-', markersize=4, label=f'Training Run {i+1}')
            
        plt.title('Training Loss vs Epoch (Parsed from Log)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=180, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.show()
    else:
        print(f"No valid training log data found in {log_file}.")
else:
    print(f"Log file '{log_file}' not found. Make sure to run your training command like:\n!python train_unsloth.py > {log_file}")