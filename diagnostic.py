# ============================================================
# DIAGNOSTIC — lancez ce fichier pour voir la structure exacte
# Sauvegardez sous : mon_projet_IRL/diagnostic.py
# ============================================================

import scipy.io as sio
import numpy as np

raw  = sio.loadmat("data/p1_5StridesData.mat")
data = raw['data']

# Première condition
cond = data[0, 0]

print("=== Type de cond ===")
print(type(cond))
print(cond.dtype)

# Link_Model_Based
lmb = cond['Link_Model_Based']
print("\n=== Link_Model_Based ===")
print(f"type    : {type(lmb)}")
print(f"dtype   : {lmb.dtype}")
print(f"shape   : {lmb.shape}")

# Descendre d'un niveau
lmb2 = lmb[0, 0] if lmb.ndim == 2 else lmb[0]
print(f"\naprès [0,0] ou [0] :")
print(f"type    : {type(lmb2)}")
print(f"dtype   : {lmb2.dtype}")

# Extraire l_hip_angle
hip = lmb2['l_hip_angle']
print(f"\nl_hip_angle :")
print(f"type    : {type(hip)}")
print(f"shape   : {hip.shape}")
print(f"dtype   : {hip.dtype}")

# Descendre encore si nécessaire
hip2 = hip[0, 0] if hip.ndim >= 2 and hip.shape[0] == 1 else hip
print(f"\naprès extraction :")
print(f"shape   : {hip2.shape}")
print(f"dtype   : {hip2.dtype}")
print(f"5 premières valeurs : {hip2[:5]}")