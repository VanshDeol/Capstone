import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set publication-style aesthetics for seaborn
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def save_table_as_png(df, output_path, title):
    fig, ax = plt.subplots(figsize=(12, len(df)*0.4 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title, fontweight='bold', pad=20)
    
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')
            
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    # 1. Load the data
    csv_path = "/Users/vanshdeol/Capstone/results/summary_results.csv"
    out_dir = "/Users/vanshdeol/Capstone/results/paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Clean up model names for prettier plotting (e.g., 'Qwen/Qwen2.5-1.5B-Instruct' -> 'Qwen2.5-1.5B')
    df['Model_Name'] = df['Model'].apply(lambda x: x.split('/')[-1])
    
    # Aggregate metrics across seeds (mean and std)
    mean_df = df.groupby(['Model_Name', 'Exposure']).mean(numeric_only=True).reset_index()
    std_df = df.groupby(['Model_Name', 'Exposure']).std(numeric_only=True).fillna(0).reset_index()
    
    models = mean_df['Model_Name'].unique()
    exposures = sorted(mean_df['Exposure'].unique())

    # =========================================================================
    # FIGURE 1: Accuracy vs Exposure (Base, Input-only, Input-output)
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes1 = axes1.flatten()
    
    for i, model in enumerate(models):
        ax = axes1[i]
        m_df = mean_df[mean_df['Model_Name'] == model].sort_values('Exposure')
        
        # Plot lines
        ax.plot(m_df['Exposure'], m_df['Input_Only_Accuracy'], marker='o', linestyle='-', label='Input-Only', color='blue')
        
        ax.plot(m_df['Exposure'], m_df['Input_Output_Accuracy'], marker='s', linestyle='-', label='Input-Output', color='red')
        # Base is constant, so we can plot it as a horizontal dashed line
        base_acc = m_df['Base_Accuracy'].iloc[0]
        ax.axhline(y=base_acc, linestyle='--', color='black', label='Base Model (Zero-Shot)')
        
        ax.set_title(f"{model} - Accuracy vs Exposure", fontweight='bold')
        ax.set_xlabel("Exposure (Samples)")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(exposures)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.legend()
            
    fig1.suptitle('Accuracy vs Exposure (Averaged Across 3 Random Seeds)', fontsize=16, fontweight='bold')
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.92)
    fig1.savefig(os.path.join(out_dir, "Figure_1_Accuracy_vs_Exposure.pdf"), dpi=300)
    fig1.savefig(os.path.join(out_dir, "Figure_1_Accuracy_vs_Exposure.png"), dpi=300)
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2A: Wrong-to-Correct (W2C) Transitions
    # =========================================================================
    fig2a, axes2a = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes2a = axes2a.flatten()
    
    for i, model in enumerate(models):
        ax = axes2a[i]
        m_df = mean_df[mean_df['Model_Name'] == model].sort_values('Exposure')
        
        # Plot W2C
        ax.plot(m_df['Exposure'], m_df['Input_Only_W2C'], marker='o', linestyle='-', color='blue', label='Input-Only')
        ax.plot(m_df['Exposure'], m_df['Input_Output_W2C'], marker='s', linestyle='-', color='red', label='Input-Output')
        
        ax.set_title(f"{model} - W2C Transitions", fontweight='bold')
        ax.set_xlabel("Exposure (Samples)")
        ax.set_ylabel("Number of Questions")
        ax.set_xticks(exposures)
        if i == 0:
            ax.legend()
            
    fig2a.suptitle('Wrong-to-Correct (W2C) Transitions (Averaged Across 3 Random Seeds)', fontsize=16, fontweight='bold')
    fig2a.tight_layout()
    fig2a.subplots_adjust(top=0.92)
    fig2a.savefig(os.path.join(out_dir, "Figure_2A_W2C_Transitions.pdf"), dpi=300)
    fig2a.savefig(os.path.join(out_dir, "Figure_2A_W2C_Transitions.png"), dpi=300)
    plt.close(fig2a)

    # =========================================================================
    # FIGURE 2B: Correct-to-Wrong (C2W) Transitions
    # =========================================================================
    fig2b, axes2b = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes2b = axes2b.flatten()
    
    for i, model in enumerate(models):
        ax = axes2b[i]
        m_df = mean_df[mean_df['Model_Name'] == model].sort_values('Exposure')
        
        # Plot C2W
        ax.plot(m_df['Exposure'], m_df['Input_Only_C2W'], marker='o', linestyle='-', color='blue', label='Input-Only')
        ax.plot(m_df['Exposure'], m_df['Input_Output_C2W'], marker='s', linestyle='-', color='red', label='Input-Output')
        
        ax.set_title(f"{model} - C2W Transitions", fontweight='bold')
        ax.set_xlabel("Exposure (Samples)")
        ax.set_ylabel("Number of Questions")
        ax.set_xticks(exposures)
        if i == 0:
            ax.legend()
            
    fig2b.suptitle('Correct-to-Wrong (C2W) Transitions (Averaged Across 3 Random Seeds)', fontsize=16, fontweight='bold')
    fig2b.tight_layout()
    fig2b.subplots_adjust(top=0.92)
    fig2b.savefig(os.path.join(out_dir, "Figure_2B_C2W_Transitions.pdf"), dpi=300)
    fig2b.savefig(os.path.join(out_dir, "Figure_2B_C2W_Transitions.png"), dpi=300)
    plt.close(fig2b)

    # =========================================================================
    # FIGURE 3: Agreement vs Exposure
    # =========================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes3 = axes3.flatten()
    
    for i, model in enumerate(models):
        ax = axes3[i]
        m_df = mean_df[mean_df['Model_Name'] == model].sort_values('Exposure')
        
        ax.plot(m_df['Exposure'], m_df['Input_Only_Agreement'], marker='o', linestyle='-', label='Input-Only', color='blue')
        ax.plot(m_df['Exposure'], m_df['Input_Output_Agreement'], marker='s', linestyle='-', label='Input-Output', color='red')
        
        ax.set_title(f"{model} - Agreement w/ Base Model", fontweight='bold')
        ax.set_xlabel("Exposure (Samples)")
        ax.set_ylabel("Agreement Rate")
        ax.set_xticks(exposures)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.legend()
            
    fig3.suptitle('Agreement w/ Base Model (Averaged Across 3 Random Seeds)', fontsize=16, fontweight='bold')
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.92)
    fig3.savefig(os.path.join(out_dir, "Figure_3_Agreement.pdf"), dpi=300)
    fig3.savefig(os.path.join(out_dir, "Figure_3_Agreement.png"), dpi=300)
    plt.close(fig3)

    # =========================================================================
    # GENERATE CLEAN PAPER TABLES (Aggregated across seeds)
    # =========================================================================
    
    # Format function for clean percentages
    fmt = lambda x: f"{x:.3f}"
    fmt_int = lambda x: f"{x:.1f}"

    # Table 1 - Main Accuracy
    table1 = pd.DataFrame({
        'Model_Name': mean_df['Model_Name'],
        'Exposure': mean_df['Exposure'],
        'Base_Accuracy': mean_df['Base_Accuracy'].apply(fmt) + " ± " + std_df['Base_Accuracy'].apply(fmt),
        'Input_Only_Accuracy': mean_df['Input_Only_Accuracy'].apply(fmt) + " ± " + std_df['Input_Only_Accuracy'].apply(fmt),
        'Input_Output_Accuracy': mean_df['Input_Output_Accuracy'].apply(fmt) + " ± " + std_df['Input_Output_Accuracy'].apply(fmt)
    })
    
    print("\n--- TABLE 1: Main Accuracy Results ---")
    print(table1.to_string(index=False))
    table1.to_csv(os.path.join(out_dir, "Table_1_Accuracy.csv"), index=False)
    save_table_as_png(table1, os.path.join(out_dir, "Table_1_Accuracy_Image.png"), "Table 1: Main Accuracy Results (Averaged Across 3 Seeds)")

    # Table 2 - Transitions Analysis
    table2 = pd.DataFrame({
        'Model_Name': mean_df['Model_Name'],
        'Exposure': mean_df['Exposure'],
        'Input_Only_W2C': mean_df['Input_Only_W2C'].apply(fmt_int) + " ± " + std_df['Input_Only_W2C'].apply(fmt_int),
        'Input_Output_W2C': mean_df['Input_Output_W2C'].apply(fmt_int) + " ± " + std_df['Input_Output_W2C'].apply(fmt_int),
        'Input_Only_C2W': mean_df['Input_Only_C2W'].apply(fmt_int) + " ± " + std_df['Input_Only_C2W'].apply(fmt_int),
        'Input_Output_C2W': mean_df['Input_Output_C2W'].apply(fmt_int) + " ± " + std_df['Input_Output_C2W'].apply(fmt_int)
    })
        
    print("\n--- TABLE 2: Transition Analysis (W2C / C2W) ---")
    print(table2.to_string(index=False))
    table2.to_csv(os.path.join(out_dir, "Table_2_Transitions.csv"), index=False)
    save_table_as_png(table2, os.path.join(out_dir, "Table_2_Transitions_Image.png"), "Table 2: Transition Analysis (Averaged Across 3 Seeds)")

    # Table 3 - Agreement
    table3 = pd.DataFrame({
        'Model_Name': mean_df['Model_Name'],
        'Exposure': mean_df['Exposure'],
        'Input_Only_Agreement': mean_df['Input_Only_Agreement'].apply(fmt) + " ± " + std_df['Input_Only_Agreement'].apply(fmt),
        'Input_Output_Agreement': mean_df['Input_Output_Agreement'].apply(fmt) + " ± " + std_df['Input_Output_Agreement'].apply(fmt)
    })
    
    print("\n--- TABLE 3: Agreement Rates ---")
    print(table3.to_string(index=False))
    table3.to_csv(os.path.join(out_dir, "Table_3_Agreement.csv"), index=False)
    save_table_as_png(table3, os.path.join(out_dir, "Table_3_Agreement_Image.png"), "Table 3: Agreement Rates (Averaged Across 3 Seeds)")

    print(f"\nAll publication-style figures and tables have been saved to: {out_dir}")

if __name__ == "__main__":
    main()