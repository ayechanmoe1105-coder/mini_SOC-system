"""
CICIDS 2017 Dataset Loader & Preprocessor
==========================================
Downloads a small sample or loads a local CICIDS 2017 CSV file,
maps the features to the format used by this SOC project, and
prepares a clean DataFrame ready for ML training.

CICIDS 2017 Download:
  https://www.unb.ca/cic/datasets/ids-2017.html
  Recommended file: Thursday-WorkingHours-Morning-WebAttacks.csv
                    Tuesday-WorkingHours.csv (SSH Brute Force)
"""

import os, warnings
import numpy  as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Labels in CICIDS 2017 ──────────────────────────────────────────────────
def _make_attack_map():
    """Build label map that handles multiple encodings of the dash character."""
    base = {
        'BENIGN'              : 'Normal',
        'SSH-Patator'         : 'Brute Force',
        'FTP-Patator'         : 'Brute Force',
        'DoS Hulk'            : 'DoS Attack',
        'DoS GoldenEye'       : 'DoS Attack',
        'DoS slowloris'       : 'DoS Attack',
        'DoS Slowhttptest'    : 'DoS Attack',
        'DDoS'                : 'DDoS Attack',
        'PortScan'            : 'Port Scan',
        'Bot'                 : 'Botnet',
        'Infiltration'        : 'Infiltration',
        'Heartbleed'          : 'Exploit',
    }
    # Web attack labels may use en-dash (\u2013), replacement char (\ufffd), hyphen, or \x96
    for sep in ['\u2013', '\ufffd', '-', '\x96']:
        base[f'Web Attack {sep} Brute Force'] = 'Brute Force'
        base[f'Web Attack {sep} XSS']         = 'XSS'
        base[f'Web Attack {sep} Sql Injection'] = 'SQL Injection'
    return base

CICIDS_ATTACK_MAP = _make_attack_map()

# Features from CICIDS 2017 that map well to network/log anomaly detection
SELECTED_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Max',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Max',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Packet Length Mean',
    'Packet Length Std',
    'Average Packet Size',
]


def load_cicids_csv(filepath: str, max_rows: int = 50000) -> pd.DataFrame:
    """Load a CICIDS 2017 CSV file and return a cleaned DataFrame."""
    print(f"  Loading: {filepath}")
    df = pd.read_csv(filepath, nrows=max_rows, low_memory=False)

    # Strip whitespace from column names (CICIDS files sometimes have spaces)
    df.columns = df.columns.str.strip()

    # Find the label column (may be 'Label' or ' Label')
    label_col = None
    for col in df.columns:
        if col.strip().lower() == 'label':
            label_col = col
            break
    if label_col is None:
        raise ValueError("No 'Label' column found in the CSV file.")
    df.rename(columns={label_col: 'Label'}, inplace=True)

    # Map labels
    df['Attack_Type'] = df['Label'].map(CICIDS_ATTACK_MAP).fillna('Unknown')
    df['is_attack']   = (df['Attack_Type'] != 'Normal').astype(int)

    # Keep only selected features + label
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    df = df[available + ['Label', 'Attack_Type', 'is_attack']].copy()

    # Replace inf / -inf with NaN, then drop rows with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"  Rows kept : {len(df):,}")
    print(f"  Features  : {len(available)}")
    print(f"  Attack %  : {df['is_attack'].mean()*100:.1f}%")
    return df


def generate_sample_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a CICIDS-like synthetic dataset for demonstration when the
    real CSV is not yet available.  Each record has realistic feature
    distributions for Normal vs Attack traffic.
    """
    rng = np.random.default_rng(seed)

    n_normal = int(n_samples * 0.70)
    n_attack = n_samples - n_normal

    def normal_traffic(n):
        return {
            'Flow Duration'               : rng.integers(100, 5_000_000, n),
            'Total Fwd Packets'           : rng.integers(1, 50, n),
            'Total Backward Packets'      : rng.integers(1, 50, n),
            'Total Length of Fwd Packets' : rng.integers(40, 2000, n),
            'Total Length of Bwd Packets' : rng.integers(40, 2000, n),
            'Fwd Packet Length Max'       : rng.integers(40, 1500, n),
            'Fwd Packet Length Mean'      : rng.uniform(40, 800, n),
            'Bwd Packet Length Max'       : rng.integers(40, 1500, n),
            'Bwd Packet Length Mean'      : rng.uniform(40, 800, n),
            'Flow Bytes/s'                : rng.uniform(100, 100_000, n),
            'Flow Packets/s'              : rng.uniform(0.1, 500, n),
            'Flow IAT Mean'               : rng.uniform(1000, 500_000, n),
            'Flow IAT Max'                : rng.uniform(5000, 2_000_000, n),
            'Fwd IAT Mean'                : rng.uniform(1000, 500_000, n),
            'Bwd IAT Mean'                : rng.uniform(1000, 500_000, n),
            'Packet Length Mean'          : rng.uniform(40, 800, n),
            'Packet Length Std'           : rng.uniform(0, 300, n),
            'Average Packet Size'         : rng.uniform(40, 800, n),
        }

    def attack_traffic(n):
        # Attacks: many small packets, very short IAT, abnormal byte rates
        return {
            'Flow Duration'               : rng.integers(1, 50_000, n),
            'Total Fwd Packets'           : rng.integers(100, 5000, n),
            'Total Backward Packets'      : rng.integers(0, 10, n),
            'Total Length of Fwd Packets' : rng.integers(100, 500, n),
            'Total Length of Bwd Packets' : rng.integers(0, 100, n),
            'Fwd Packet Length Max'       : rng.integers(40, 300, n),
            'Fwd Packet Length Mean'      : rng.uniform(40, 200, n),
            'Bwd Packet Length Max'       : rng.integers(0, 100, n),
            'Bwd Packet Length Mean'      : rng.uniform(0, 50, n),
            'Flow Bytes/s'                : rng.uniform(500_000, 50_000_000, n),
            'Flow Packets/s'              : rng.uniform(5000, 500_000, n),
            'Flow IAT Mean'               : rng.uniform(1, 500, n),
            'Flow IAT Max'                : rng.uniform(10, 5000, n),
            'Fwd IAT Mean'                : rng.uniform(1, 500, n),
            'Bwd IAT Mean'                : rng.uniform(0, 100, n),
            'Packet Length Mean'          : rng.uniform(40, 200, n),
            'Packet Length Std'           : rng.uniform(0, 50, n),
            'Average Packet Size'         : rng.uniform(40, 200, n),
        }

    # Build attack type distribution
    attack_types  = ['Brute Force', 'Port Scan', 'DoS Attack',
                     'SQL Injection', 'XSS', 'DDoS Attack']
    attack_labels = rng.choice(attack_types, size=n_attack)

    df_norm = pd.DataFrame(normal_traffic(n_normal))
    df_norm['Attack_Type'] = 'Normal'
    df_norm['is_attack']   = 0

    df_atk = pd.DataFrame(attack_traffic(n_attack))
    df_atk['Attack_Type'] = attack_labels
    df_atk['is_attack']   = 1

    df = pd.concat([df_norm, df_atk], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  Synthetic dataset: {len(df):,} rows")
    print(f"  Normal: {n_normal:,}  |  Attack: {n_attack:,}")
    print(f"  Attack types: {', '.join(attack_types)}")
    return df


def load_dataset(csv_path: str = None, max_rows: int = 50000) -> pd.DataFrame:
    """
    Main entry point.
    - If csv_path points to a real CICIDS 2017 CSV → load it.
    - Otherwise → generate a realistic synthetic dataset for demo.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"\n[DATASET] Loading real CICIDS 2017 data from:\n  {csv_path}")
        return load_cicids_csv(csv_path, max_rows)
    else:
        print("\n[DATASET] Real CSV not found. Using synthetic CICIDS-style data.")
        print("  Expected path (see model_trainer.py):")
        print(f"    {csv_path or '(not set)'}")
        print("  Download CICIDS 2017 from: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("  Save as:  data/CICIDS2017_WebAttacks.csv  next to model_trainer.py\n")
        return generate_sample_dataset()


if __name__ == '__main__':
    df = load_dataset()
    print("\nSample rows:")
    print(df[['Attack_Type', 'is_attack', 'Flow Bytes/s', 'Flow Packets/s']].head(10).to_string())
    print(f"\nAttack type distribution:\n{df['Attack_Type'].value_counts().to_string()}")
