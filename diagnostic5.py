# diagnostic5.py
import scipy.io as sio
import numpy as np

raw  = sio.loadmat("data/p1_5StridesData.mat")
data = raw['data']

# Tester toutes les 33 conditions
for i in range(data.shape[1]):
    cond = data[0, i]
    lmb_raw = cond['Link_Model_Based']
    print(f"condition {i:02d} — lmb_raw.shape={lmb_raw.shape}, size={lmb_raw.size}")