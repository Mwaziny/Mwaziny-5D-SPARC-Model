import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

DATA_DIR = "SPARC_Data"

# ==========================================
# 1. Physical Constants & SI Conversion Factor
# ==========================================
# Cosmological minimum acceleration standard: a0 = 1.2e-10 m/s^2
# 1 kpc = 3.08567758e19 m, 1 km/s = 1000 m/s
# Scale Factor for (km/s)^2: 1000 * sqrt(1.2e-10 * 3.08567758e19) / 1e6 = 60.85076
SI_SCALE = 60.85076

if not os.path.exists(DATA_DIR):
    print(f"[!] ERROR: Directory '{DATA_DIR}' not found. Please ensure SPARC files are present.")
    exit()

galaxy_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.dat')]
print(f"[+] Loaded {len(galaxy_files)} observational galaxy profiles from SPARC archive.")

def load_sparc_galaxy(file_path):
    try:
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None,
                         names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
        df = df[df['Vobs'] > 0].reset_index(drop=True)
        return df
    except Exception as e:
        return None

# ==========================================
# 2. Mwaziny Dimensionless 5D Model Equation
# ==========================================
def predict_galaxy_curve(df, lambda_z, ml_disk=0.5, ml_bul=0.7):
    v_gas_sq = np.sign(df['Vgas']) * df['Vgas']**2
    v_disk_sq = ml_disk * (np.sign(df['Vdisk']) * df['Vdisk']**2)
    v_bul_sq = ml_bul * (np.sign(df['Vbul']) * df['Vbul']**2)
    
    v_bary_sq = v_gas_sq + v_disk_sq + v_bul_sq
    v_bary = np.sqrt(np.maximum(0, v_bary_sq))
    
    # Dimensionless Tension Term: lambda_z * 60.85076 * V_bary * sqrt(R_kpc)
    v_tension_sq = lambda_z * SI_SCALE * v_bary * np.sqrt(df['Rad'].values)
    
    v_pred_sq = np.maximum(0, v_bary_sq) + v_tension_sq
    return np.sqrt(np.maximum(0, v_pred_sq))

def fit_galaxy_local_ml(df, lambda_z):
    """Optimizes local Stellar Mass-to-Light ratio M/L_disk in [0.3, 0.8] for a given lambda_z."""
    def loss_func(ml):
        v_pred = predict_galaxy_curve(df, lambda_z, ml_disk=ml[0])
        return np.sum(((df['Vobs'] - v_pred) / (df['errV'] + 0.5))**2)

    res = minimize(loss_func, x0=[0.5], bounds=[(0.3, 0.8)], method='L-BFGS-B')
    best_ml = res.x[0]
    
    v_pred = predict_galaxy_curve(df, lambda_z, ml_disk=best_ml)
    ss_res = np.sum((df['Vobs'] - v_pred)**2)
    ss_tot = np.sum((df['Vobs'] - np.mean(df['Vobs']))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    chi2_red = np.sum(((df['Vobs'] - v_pred) / (df['errV'] + 0.5))**2) / len(df)
    return best_ml, r2, chi2_red, v_pred

# ==========================================
# 3. Blind Cross-Validation Protocol (50/50 Split)
# ==========================================
train_files, test_files = train_test_split(galaxy_files, test_size=0.5, random_state=42)

print("\n" + "="*70)
print("     PHASE 1: UNBIASED BLIND CROSS-VALIDATION PROTOCOL")
print("="*70)
print(f"[+] Training Set Size : {len(train_files)} galaxies (Used for lambda_z optimization)")
print(f"[+] Blind Test Set Size: {len(test_files)} galaxies (Unseen evaluation)")

def train_cost_function(params):
    lambda_z_val = params[0]
    total_chi2 = 0
    total_pts = 0
    for file in train_files:
        df = load_sparc_galaxy(file)
        if df is None or len(df) < 3:
            continue
        _, _, chi2_red, _ = fit_galaxy_local_ml(df, lambda_z_val)
        total_chi2 += chi2_red * len(df)
        total_pts += len(df)
    return total_chi2 / total_pts

res_train = minimize(train_cost_function, x0=[0.70], bounds=[(0.001, 10.0)], method='L-BFGS-B')
lambda_z_derived = res_train.x[0]

print(f"\n[=>] Derived Dimensionless Mwaziny Parameter (lambda_z): {lambda_z_derived:.6f}")

# Evaluate Training Set
train_r2_list = []
for file in train_files:
    df = load_sparc_galaxy(file)
    if df is None or len(df) < 3:
        continue
    _, r2, _, _ = fit_galaxy_local_ml(df, lambda_z_derived)
    train_r2_list.append(r2)

# Evaluate Unseen Test Set (Blind)
test_r2_list = []
for file in test_files:
    df = load_sparc_galaxy(file)
    if df is None or len(df) < 3:
        continue
    _, r2, _, _ = fit_galaxy_local_ml(df, lambda_z_derived)
    test_r2_list.append(r2)

train_median_r2 = np.median(train_r2_list)
test_median_r2 = np.median(test_r2_list)

print(f"[+] Training Set Median R^2             : {train_median_r2:.4f}")
print(f"[+] UNSEEN Test Set Median R^2 (BLIND)  : {test_median_r2:.4f}")

if test_median_r2 >= (train_median_r2 - 0.05):
    print("\n[VERDICT]: SUCCESSFUL PHYSICAL GENERALIZATION!")
    print("The dimensionless constant lambda_z predicted unseen galaxies with high stability.")
else:
    print("\n[VERDICT]: OVERFITTING DETECTED.")
print("="*70 + "\n")

# ==========================================
# 4. Full Dataset Global Evaluation & Export
# ==========================================
print("     PHASE 2: FULL DATASET UNBIASED GLOBAL EVALUATION & EXPORT")
print("="*70)

def global_cost_function(params):
    lambda_z_val = params[0]
    total_chi2 = 0
    total_pts = 0
    for file in galaxy_files:
        df = load_sparc_galaxy(file)
        if df is None or len(df) < 3:
            continue
        _, _, chi2_red, _ = fit_galaxy_local_ml(df, lambda_z_val)
        total_chi2 += chi2_red * len(df)
        total_pts += len(df)
    return total_chi2 / total_pts

res_global = minimize(global_cost_function, x0=[lambda_z_derived], bounds=[(0.001, 10.0)], method='L-BFGS-B')
lambda_z_global = res_global.x[0]

full_results = []
os.makedirs("Mwaziny_Plots", exist_ok=True)

for file in galaxy_files:
    gal_name = os.path.basename(file).replace('.dat', '')
    df = load_sparc_galaxy(file)
    if df is None or len(df) < 3:
        continue
    
    best_ml, r2, chi2_red, _ = fit_galaxy_local_ml(df, lambda_z_global)
    
    full_results.append({
        'Galaxy': gal_name,
        'R2_Score': r2,
        'Reduced_Chi2': chi2_red,
        'Optimal_ML_disk': best_ml,
        'Max_Vobs': np.max(df['Vobs']),
        'Data_Points': len(df)
    })

results_df = pd.DataFrame(full_results)
results_df.to_csv("Mwaziny_SPARC_Unbiased_Results.csv", index=False)

spirals_df = results_df[results_df['Max_Vobs'] >= 100.0]

print(f"[+] Universal Dimensionless Mwaziny Parameter (lambda_z) : {lambda_z_global:.6f}")
print(f"[+] Total SPARC Galaxies Analyzed                       : {len(results_df)}")
print(f"[+] All Galaxies Median R^2                             : {results_df['R2_Score'].median():.4f}")
print(f"[+] Major Spiral Galaxies (Vmax >= 100 km/s) Median R^2  : {spirals_df['R2_Score'].median():.4f}")
print(f"[+] Major Spiral Galaxies 75th Percentile R^2            : {spirals_df['R2_Score'].quantile(0.75):.4f}")
print(f"[+] Full Report Exported To                             : Mwaziny_SPARC_Unbiased_Results.csv")
print("="*70 + "\n")

# ==========================================
# 5. High-Precision Plotting (NGC3198 Benchmark)
# ==========================================
sample_gal = [f for f in galaxy_files if "NGC3198" in f]
if sample_gal:
    df_sample = load_sparc_galaxy(sample_gal[0])
    best_ml_sample, r2_sample, _, v_pred_sample = fit_galaxy_local_ml(df_sample, lambda_z_global)
    
    plt.figure(figsize=(9, 6))
    plt.errorbar(df_sample['Rad'], df_sample['Vobs'], yerr=df_sample['errV'], fmt='ko', 
                 capsize=3, label='SPARC Observational Data')
    plt.plot(df_sample['Rad'], v_pred_sample, 'r-', linewidth=2.5, 
             label=f'Mwaziny 5D Model (λ_z = {lambda_z_global:.4f}, M/L = {best_ml_sample:.2f})')
    
    v_bary = np.sqrt(np.maximum(0, df_sample['Vgas']**2 + best_ml_sample*df_sample['Vdisk']**2 + 0.7*df_sample['Vbul']**2))
    plt.plot(df_sample['Rad'], v_bary, 'b--', linewidth=1.5, label='Baryonic Newtonian Standard (No Dark Matter)')
    
    plt.xlabel('Galactocentric Radius R (kpc)', fontsize=12)
    plt.ylabel('Rotation Velocity V (km/s)', fontsize=12)
    plt.title(f'Mwaziny 5D Dimensionless Calibration - NGC3198 (R² = {r2_sample:.4f})', fontsize=13)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Mwaziny_Plots/NGC3198_Mwaziny_Fit_Unbiased.png', dpi=300)
    print("[+] Publication-ready plot generated: Mwaziny_Plots/NGC3198_Mwaziny_Fit_Unbiased.png")