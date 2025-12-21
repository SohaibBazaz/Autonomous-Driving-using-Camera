import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION: Update these filenames to match yours exactly ---
LOG_FILES = {
    "Model 1: Standard": "F:\\Beamng Project\\Autonomous-Driving-using-Camera\\output\\log.csv",
    "Model 2: Deep": "F:\\Beamng Project\\Autonomous-Driving-using-Camera\\output2\\log.csv",
    "Model 3: Ultra": "F:\\Beamng Project\\Autonomous-Driving-using-Camera\\output3\\log.csv"
}

def analyze_and_compare(log_dict):
    plt.figure(figsize=(12, 7))
    
    # Using high-contrast colors and distinct markers to reveal overlaps
    styles = [
        {'color': '#1f77b4', 'marker': 'o', 'ms': 4}, # Blue - Circle
        {'color': '#2ca02c', 'marker': 's', 'ms': 4}, # Green - Square
        {'color': '#d62728', 'marker': '^', 'ms': 4}  # Red - Triangle
    ]
    
    performance_summary = []
    files_found = 0

    print("🔍 Searching for log files...")
    
    for i, (name, file_path) in enumerate(log_dict.items()):
        if not os.path.exists(file_path):
            print(f"❌ Could not find: {file_path}")
            continue
        
        # Load CSV and handle potential column name variations
        df = pd.read_csv(file_path)
        files_found += 1
        style = styles[i % len(styles)]
        
        # Determine column names (handles 'Train Loss' vs 'Train_Loss')
        train_col = [c for c in df.columns if 'Train' in c and 'Loss' in c][0]
        test_col = [c for c in df.columns if 'Test' in c and 'Loss' in c][0]
        epoch_col = [c for c in df.columns if 'Epoch' in c][0]

        # Plot Training Loss (Dashed/Transparent)
        plt.plot(df[epoch_col], df[train_col], 
                 linestyle='--', color=style['color'], alpha=0.2, 
                 label=f"{name} (Train)")
        
        # Plot Test Loss (Solid with Markers to show overlaps)
        plt.plot(df[epoch_col], df[test_col], 
                 linestyle='-', color=style['color'], 
                 marker=style['marker'], markevery=5, markersize=style['ms'],
                 linewidth=2, label=f"{name} (Test)")
            
        # Record best performance
        best_val = df[test_col].min()
        best_epoch = df.loc[df[test_col].idxmin(), epoch_col]
        performance_summary.append({
            'name': name,
            'best_loss': best_val,
            'epoch': int(best_epoch)
        })

    if files_found < 3:
        print(f"⚠️ Warning: Only {files_found}/3 files were found. Check your file paths!")

    # --- 2. THE WINNER LOGIC ---
    if performance_summary:
        # Sort by best (lowest) test loss
        performance_summary.sort(key=lambda x: x['best_loss'])
        winner = performance_summary[0]

        print("\n" + "="*55)
        print(f"{'MODEL RANKING':<25} | {'BEST TEST LOSS':<15} | {'EPOCH'}")
        print("-" * 55)
        for rank, model in enumerate(performance_summary, 1):
            medal = "🏆" if rank == 1 else f" {rank}."
            print(f"{medal} {model['name']:<22} | {model['best_loss']:<15.6f} | {model['epoch']}")
        print("="*55)
        print(f"CONCLUSION: The {winner['name']} is performing BEST.")
        print(f"It reached a minimum error of {winner['best_loss']:.6f} at Epoch {winner['epoch']}.")

    # --- 3. PLOT STYLING ---
    plt.title('Performance Comparison: Standard vs Deep vs Ultra', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Focus Y-axis on the important data (0.00 to 0.08 based on your image)
    plt.ylim(0, 0.08) 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_and_compare(LOG_FILES)