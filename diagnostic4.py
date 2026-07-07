# diagnostic4.py
import scipy.io as sio
import numpy as np

raw  = sio.loadmat("data/p1_5StridesData.mat")
data = raw['data']

cond = data[0, 0]
lmb_raw = cond['Link_Model_Based']

print(f"lmb_raw type  : {type(lmb_raw)}")
print(f"lmb_raw shape : {lmb_raw.shape}")
print(f"lmb_raw ndim  : {lmb_raw.ndim}")
print(f"lmb_raw size  : {lmb_raw.size}")

# Essayer tous les accès possibles
print("\n--- Essai lmb_raw.item() ---")
try:
    x = lmb_raw.item()
    print(f"type : {type(x)}")
    if hasattr(x, 'dtype'):
        print(f"dtype: {x.dtype}")
        hip = x['l_hip_angle']
        print(f"l_hip_angle shape : {hip.shape}")
except Exception as e:
    print(f"ERREUR : {e}")

print("\n--- Essai lmb_raw.ravel()[0] ---")
try:
    x = lmb_raw.ravel()[0]
    print(f"type : {type(x)}")
    if hasattr(x, 'dtype'):
        print(f"dtype: {x.dtype}")
        hip = x['l_hip_angle']
        print(f"l_hip_angle shape : {hip.shape}")
except Exception as e:
    print(f"ERREUR : {e}")

print("\n--- Essai lmb_raw[(0,0)] ---")
try:
    x = lmb_raw[(0,0)]
    print(f"type : {type(x)}")
    if hasattr(x, 'dtype'):
        print(f"dtype: {x.dtype}")
        hip = x['l_hip_angle']
        print(f"l_hip_angle shape : {hip.shape}")
except Exception as e:
    print(f"ERREUR : {e}")