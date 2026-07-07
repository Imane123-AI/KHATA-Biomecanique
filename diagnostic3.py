# diagnostic3.py
import scipy.io as sio
import numpy as np

raw  = sio.loadmat("data/p1_5StridesData.mat")
data = raw['data']

print(f"data.shape : {data.shape}")
print(f"data.dtype : {data.dtype}")

# Accès à la première condition
cond = data[0, 0]
print(f"\ncond type  : {type(cond)}")
print(f"cond dtype : {cond.dtype}")

# Accès à Link_Model_Based
lmb_raw = cond['Link_Model_Based']
print(f"\nlmb_raw type  : {type(lmb_raw)}")
print(f"lmb_raw shape : {lmb_raw.shape}")
print(f"lmb_raw dtype : {lmb_raw.dtype}")
print(f"lmb_raw       : {lmb_raw}")

# Accès à Force
force_raw = cond['Force']
print(f"\nforce_raw type  : {type(force_raw)}")
print(f"force_raw shape : {force_raw.shape}")
print(f"force_raw       : {force_raw}")