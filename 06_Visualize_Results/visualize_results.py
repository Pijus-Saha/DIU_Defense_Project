import matplotlib.pyplot as plt
import numpy as np
import os

def create_charts():
    print("--- RESULT VISUALIZER ---")
    print("Enter the values from your 'test_accuracy_detailed.py' output:")
    
    # 1. INPUT DATA
    try:
        count_pass = int(input("1. PASS (Perfect Match):      "))
        count_sys_pass = int(input("2. SYSTEM-PASS (Saved by DB): "))
        count_ai_pass = int(input("3. AI-PASS (Broken by DB):    "))
        count_fail = int(input("4. FAIL (Total Miss):         "))
    except ValueError:
        print("Please enter valid numbers!")
        return

    total = count_pass + count_sys_pass + count_ai_pass + count_fail
    
    # Calculate Accuracies
    raw_correct = count_pass + count_ai_pass
    sys_correct = count_pass + count_sys_pass
    
    raw_acc = (raw_correct / total) * 100
    sys_acc = (sys_correct / total) * 100

    print(f"\nGenerating charts based on {total} samples...")

    # --- CHART 1: PIE CHART (The "Impact" Chart) ---
    labels = ['Perfect AI Prediction', 'Saved by Post-Processing', 'Dictionary Error', 'Total Failure']
    sizes = [count_pass, count_sys_pass, count_ai_pass, count_fail]
    colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c'] # Green, Blue, Yellow, Red
    explode = (0, 0.1, 0, 0)  # "Explode" the 2nd slice (System Pass) to highlight it!

    plt.figure(figsize=(10, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12})
    plt.title(f"Impact of Hybrid Post-Processing Module\n(Total Samples: {total})", fontsize=14, fontweight='bold')
    
    save_path1 = "impact_analysis.png"
    plt.savefig(save_path1)
    print(f"1. Saved {save_path1}")
    plt.close()

    # --- CHART 2: BAR CHART (The "Before vs After" Chart) ---
    plt.figure(figsize=(8, 6))
    
    metrics = ['Raw CRNN Model', 'Hybrid System (CRNN + Dict)']
    values = [raw_acc, sys_acc]
    bar_colors = ['#95a5a6', '#27ae60'] # Grey vs Green

    bars = plt.bar(metrics, values, color=bar_colors, width=0.5)

    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold')

    plt.ylabel('Accuracy (%)')
    plt.title('Performance Improvement via Domain Adaptation', fontsize=14, fontweight='bold')
    plt.ylim(0, 110) # Give some headroom
    
    save_path2 = "accuracy_comparison.png"
    plt.savefig(save_path2)
    print(f"2. Saved {save_path2}")
    plt.close()

    print("\nDone! Check your folder for the PNG images.")

if __name__ == "__main__":
    create_charts()