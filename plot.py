import matplotlib.pyplot as plt
import numpy as np

# 1. 準備數據
hidden_dims = [256, 512, 1024]

# MLP Baseline Data
mlp_auc = [0.8301, 0.9125, 0.9140]
mlp_f1  = [0.3803, 0.8679, 0.8583]

# LLM-Enhanced MLP Data
llm_auc = [0.9026, 0.9207, 0.9148]
llm_f1  = [0.8416, 0.8333, 0.8252]

# 2. 設置畫布風格
plt.style.use('seaborn-v0_8-whitegrid') # 如果報錯可改為 'ggplot' 或刪除此行
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# 顏色與標記設定
color_mlp = '#d62728' # Brick Red
color_llm = '#1f77b4' # Muted Blue
marker_size = 8
line_width = 2.5

# =======================
# 子圖 1: AUC Comparison
# =======================
ax1.plot(hidden_dims, mlp_auc, marker='o', linestyle='--', color=color_mlp, 
         linewidth=line_width, markersize=marker_size, label='Baseline MLP')
ax1.plot(hidden_dims, llm_auc, marker='s', linestyle='-', color=color_llm, 
         linewidth=line_width, markersize=marker_size, label='LLM-Enhanced MLP (Ours)')

# 標註數值
for x, y in zip(hidden_dims, mlp_auc):
    ax1.text(x, y - 0.015, f'{y:.4f}', ha='center', va='top', color=color_mlp, fontsize=10, fontweight='bold')
for x, y in zip(hidden_dims, llm_auc):
    ax1.text(x, y + 0.005, f'{y:.4f}', ha='center', va='bottom', color=color_llm, fontsize=10, fontweight='bold')

ax1.set_title('AUC Sensitivity to Hidden Dimension', fontsize=14, pad=15)
ax1.set_ylabel('AUC Score', fontsize=12)
ax1.set_xlabel('Hidden Dimension', fontsize=12)
ax1.set_xticks(hidden_dims)
ax1.set_ylim(0.80, 0.95) # 根據數據範圍調整，突出差異
ax1.legend(loc='lower right', fontsize=11, frameon=True)
ax1.grid(True, linestyle=':', alpha=0.6)

# =======================
# 子圖 2: F1-Score Comparison
# =======================
ax2.plot(hidden_dims, mlp_f1, marker='o', linestyle='--', color=color_mlp, 
         linewidth=line_width, markersize=marker_size, label='Baseline MLP')
ax2.plot(hidden_dims, llm_f1, marker='s', linestyle='-', color=color_llm, 
         linewidth=line_width, markersize=marker_size, label='LLM-Enhanced MLP (Ours)')

# 標註數值
for x, y in zip(hidden_dims, mlp_f1):
    offset = -0.05 if y > 0.5 else 0.05
    ax2.text(x, y + offset, f'{y:.4f}', ha='center', va='center', color=color_mlp, fontsize=10, fontweight='bold')
for x, y in zip(hidden_dims, llm_f1):
    ax2.text(x, y + 0.02, f'{y:.4f}', ha='center', va='bottom', color=color_llm, fontsize=10, fontweight='bold')

ax2.set_title('F1-Score Sensitivity to Hidden Dimension', fontsize=14, pad=15)
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_xlabel('Hidden Dimension', fontsize=12)
ax2.set_xticks(hidden_dims)
ax2.set_ylim(0.30, 1.0) # 範圍拉大以容納 MLP 的崩潰點
ax2.legend(loc='lower right', fontsize=11, frameon=True)
ax2.grid(True, linestyle=':', alpha=0.6)

# =======================
# 整體調整與保存
# =======================
plt.suptitle('Hyperparameter Sensitivity Analysis: Hidden Dimension', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# 保存圖片
plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()