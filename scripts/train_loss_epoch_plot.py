import re
import ast
from pathlib import Path
import matplotlib.pyplot as plt

# Define paths relative to where the script is executed
log_file = Path("assets/training.log")
loss_output_file = Path("assets/unsloth_train_loss_epoch.png")

def load_log_history(path: Path):
    """Parser for raw log dumps containing {'loss': ..., 'epoch': ...} lines."""
    if not path.exists():
        print(f"Error: Could not find '{path}'. Make sure it is in the same directory.")
        return []

    rows = []
    with open(path, "r") as f:
        for line in f:
            # Target lines that look like our metric dictionaries
            if "{'loss':" in line and "'epoch':" in line:
                try:
                    # Extract the dictionary string using regex
                    match = re.search(r"(\{.*?'loss':.*?'epoch':.*?\})", line)
                    if match:
                        # Safely evaluate the string into a Python dict
                        data = ast.literal_eval(match.group(1))
                        
                        rows.append({
                            "step": int(data.get("step", len(rows) + 1)),
                            "epoch": float(data["epoch"]),
                            "loss": float(data["loss"])
                        })
                except Exception:
                    # Skip lines that fail to parse
                    continue

    return rows

def split_runs(rows):
    """Split into runs if epoch resets (e.g., if multiple training sessions are in one log)."""
    runs = []
    current = []

    for row in rows:
        # If the epoch number goes down, it means a new training run started
        if current and row["epoch"] < current[-1]["epoch"]:
            runs.append(current)
            current = []
        current.append(row)

    if current:
        runs.append(current)
    return runs

def plot_loss(runs):
    """Generates and saves the loss vs epoch plot."""
    loss_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for i, run in enumerate(runs):
        epochs = [row["epoch"] for row in run]
        losses = [row["loss"] for row in run]
        plt.plot(epochs, losses, marker="o", linestyle="-", markersize=4, label=f"Training Run {i + 1}")

    plt.title("SFT Training Loss vs Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(loss_output_file, dpi=180, bbox_inches="tight")
    print(f"Saved loss plot to {loss_output_file}")

def main():
    rows = load_log_history(log_file)

    if not rows:
        print(f"No valid training data found. Expected lines with {{'loss': ..., 'epoch': ...}} in {log_file}.")
        return

    runs = split_runs(rows)
    print(f"Loaded {len(rows)} data points from {log_file}")
    
    # Generate and save the plot
    plot_loss(runs)

if __name__ == "__main__":
    main()