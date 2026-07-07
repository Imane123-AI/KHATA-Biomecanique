# diagnostic2.py
import scipy.io as sio
import numpy as np

raw  = sio.loadmat("data/p1_5StridesData.mat")
data = raw['data']
cond = data[0, 0]

lmb_raw = cond['Link_Model_Based']
print(f"shape : {lmb_raw.shape}")
print(f"ndim  : {lmb_raw.ndim}")
print(f"dtype : {lmb_raw.dtype}")

# Essayer différents accès
for acces in ["lmb_raw[0]", "lmb_raw[0,0]", "lmb_raw.flat[0]", "lmb_raw.item()"]:
    try:
        val = eval(acces)
        print(f"\n{acces} → OK, type={type(val)}, dtype={val.dtype}")
        # Tester l_hip_angle
        hip = val['l_hip_angle']
        print(f"  l_hip_angle shape : {hip.shape}")
    except Exception as e:
        print(f"\n{acces} → ERREUR : {e}")