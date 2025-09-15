"""
normalization_effect_analysis.py

说明：
- 读取 /mnt/data/data_final.xlsx（或替换为本地路径）
- 从 SMILES 生成 1024-bit ECFP（radius=2）
- 构造输入：ECFP + ['log RCF','flipid','TPSA','HBD','HBA']
- 比较三种归一化策略：raw / physchem-only / all-features
- 对 FCNN（MLPRegressor）与 GBRT（GradientBoostingRegressor）做交叉验证评估（R2, MAE）
- 做配对 t 检验与 Wilcoxon 符号秩检验（raw vs all_scaled）
- 绘制：物化特征分布直方图+KDE（归一化前后），和若干高方差 ECFP bit 的直方图
- 输出：图像 + CV summary CSV + 统计检验 CSV 到 output_dir
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Settings (按需修改)
# -------------------------
DATA_FP = "../data/data_final.xlsx"   # 或者 "/mnt/data/data_final.xlsx"
OUTPUT_DIR = "normalization_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ECFP_BITS = 1024
ECFP_RADIUS = 2

# 模式选择：如果在受限环境（如 notebook 有时间限制）下，使用 FAST=True（小网络、少折 CV）
FAST = False   # <- 设 True 可显著加速（调试用）；论文最终结果请选择 FAST=False

# CV / 模型超参
if FAST:
    N_SPLITS = 3
    MLP_HIDDEN = (128, 64)
    MLP_MAXITER = 1000
    GBR_TREES = 100
else:
    N_SPLITS = 5
    MLP_HIDDEN = (512, 256)   # 论文中使用的
    MLP_MAXITER = 2000
    GBR_TREES = 300

RANDOM_STATE = 1412

PHYSCCHEM_COLS = ['log RCF', 'flipid', 'TPSA', 'HBD', 'HBA']   # 确认 Excel 中列名一致
TARGET_COL = 'log TF'
SMILES_COL = 'SMILES'

# -------------------------
# Helper: SMILES -> ECFP
# -------------------------
def smiles_to_ecfp_vec(smiles, n_bits=ECFP_BITS, radius=ECFP_RADIUS):
    if pd.isna(smiles) or str(smiles).strip() == "":
        return np.zeros(n_bits, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    arr = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(arr, dtype=np.uint8)

# -------------------------
# Load data
# -------------------------
df = pd.read_excel(DATA_FP)
# basic checks
for c in [SMILES_COL, TARGET_COL] + PHYSCCHEM_COLS:
    if c not in df.columns:
        raise ValueError(f"Required column '{c}' not found in {DATA_FP}")

# Build ECFP matrix
ecfp_list = [smiles_to_ecfp_vec(s) for s in df[SMILES_COL].fillna("")]
X_ecfp = np.vstack(ecfp_list)   # shape: (n_samples, ECFP_BITS)
# Physchem matrix
X_phys = df[PHYSCCHEM_COLS].astype(float).values
y = df[TARGET_COL].astype(float).values

# Final feature matrices for three strategies:
X_raw = np.hstack([X_ecfp, X_phys])              # 1) raw (no scaling)
scaler_phys = StandardScaler().fit(X_phys)
X_phys_scaled = scaler_phys.transform(X_phys)
X_phys_scaled_only = np.hstack([X_ecfp, X_phys_scaled])   # 2) scale physchem only
scaler_all = StandardScaler().fit(np.hstack([X_ecfp.astype(float), X_phys]))
X_all_scaled = scaler_all.transform(np.hstack([X_ecfp.astype(float), X_phys]))  # 3) scale all features

# -------------------------
# Models & CV
# -------------------------
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fcnn = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN, learning_rate_init=0.0003,
                    max_iter=MLP_MAXITER, solver="adam", activation="relu",
                    random_state=RANDOM_STATE)

gbrt = GradientBoostingRegressor(n_estimators=GBR_TREES, learning_rate=0.05,
                                 max_depth=4, random_state=RANDOM_STATE)

def evaluate_cv(model, X, y, cv):
    r2 = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=1)
    mae = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)
    return r2, mae

results = {}
for tag, X in [('raw', X_raw), ('phys_scaled', X_phys_scaled_only), ('all_scaled', X_all_scaled)]:
    print("Evaluating:", tag)
    r2_fcnn, mae_fcnn = evaluate_cv(fcnn, X, y, cv)
    r2_gbrt, mae_gbrt = evaluate_cv(gbrt, X, y, cv)
    results[tag] = {
        'fcnn_r2': r2_fcnn, 'fcnn_mae': mae_fcnn,
        'gbrt_r2': r2_gbrt, 'gbrt_mae': mae_gbrt
    }

# -------------------------
# Statistics: paired test (raw vs all_scaled)
# -------------------------
from scipy.stats import ttest_rel, wilcoxon
stat_rows = []
for model_key, r2_key, mae_key in [('FCNN','fcnn_r2','fcnn_mae'), ('GBRT','gbrt_r2','gbrt_mae')]:
    a = results['raw'][r2_key]
    b = results['all_scaled'][r2_key]
    t_r2, p_t_r2 = ttest_rel(a, b)
    try:
        w_r2, p_w_r2 = wilcoxon(a, b)
    except Exception:
        p_w_r2 = np.nan

    a_mae = results['raw'][mae_key]
    b_mae = results['all_scaled'][mae_key]
    t_mae, p_t_mae = ttest_rel(a_mae, b_mae)
    try:
        w_mae, p_w_mae = wilcoxon(a_mae, b_mae)
    except Exception:
        p_w_mae = np.nan

    stat_rows.append({
        'model': model_key,
        'r2_raw_mean': a.mean(), 'r2_all_mean': b.mean(), 'ttest_r2_p': p_t_r2, 'wilcoxon_r2_p': p_w_r2,
        'mae_raw_mean': a_mae.mean(), 'mae_all_mean': b_mae.mean(), 'ttest_mae_p': p_t_mae, 'wilcoxon_mae_p': p_w_mae
    })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(os.path.join(OUTPUT_DIR, "normalization_stat_tests.csv"), index=False)

# -------------------------
# Visualizations: physchem distributions & ECFP top bits
# -------------------------
sns.set(style="whitegrid")
# physchem before vs after (phys_scaled)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(PHYSCCHEM_COLS):
    ax = axes[i]
    sns.histplot(df[col].dropna(), kde=True, stat='density', ax=ax, label='raw', alpha=0.6)
    # transformed
    vals_scaled = scaler_phys.transform(df[PHYSCCHEM_COLS].astype(float).values)[:, i]
    sns.kdeplot(vals_scaled, ax=ax, label='phys_scaled', linestyle='--')
    ax.set_title(col)
    ax.legend()
# hide extra axis if present
if len(PHYSCCHEM_COLS) < len(axes):
    for j in range(len(PHYSCCHEM_COLS), len(axes)):
        axes[j].axis('off')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure_S3_physchem_distributions.png"), dpi=200)

# ECFP: 取方差最大的若干 bit（示例取 6 个），绘制 histogram（0/1）
bit_vars = X_ecfp.var(axis=0)
top_bits = np.argsort(bit_vars)[-6:][::-1]
fig2, axes2 = plt.subplots(2, 3, figsize=(12,6))
axes2 = axes2.flatten()
for i, b in enumerate(top_bits):
    ax = axes2[i]
    sns.countplot(x=X_ecfp[:, b], ax=ax)
    ax.set_title(f"ECFP bit {b} (var={bit_vars[b]:.4f})")
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "Figure_S3_ecfp_topbits.png"), dpi=200)

# -------------------------
# Save CV summary
# -------------------------
rows = []
for tag in results:
    res = results[tag]
    rows.append({
        'feature_set': tag,
        'fcnn_r2_mean': res['fcnn_r2'].mean(), 'fcnn_r2_std': res['fcnn_r2'].std(),
        'fcnn_mae_mean': res['fcnn_mae'].mean(), 'fcnn_mae_std': res['fcnn_mae'].std(),
        'gbrt_r2_mean': res['gbrt_r2'].mean(), 'gbrt_r2_std': res['gbrt_r2'].std(),
        'gbrt_mae_mean': res['gbrt_mae'].mean(), 'gbrt_mae_std': res['gbrt_mae'].std(),
    })
cv_summary = pd.DataFrame(rows)
cv_summary.to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"), index=False)

print("DONE. Outputs saved in:", OUTPUT_DIR)
print("Files:", os.listdir(OUTPUT_DIR))
print("\nStat tests (raw vs all_scaled):\n", stat_df.to_string(index=False))
plt.show()