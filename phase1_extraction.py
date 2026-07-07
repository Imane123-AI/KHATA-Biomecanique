# ============================================================
# PHASE 1 — VERSION FINALE ROBUSTE
# ============================================================

import scipy.io as sio
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

DOSSIER_DATA      = "data"
DOSSIER_PROCESSED = "data/processed"
os.makedirs(DOSSIER_PROCESSED, exist_ok=True)

PARTICIPANTS = [
    "p1_5StridesData.mat",
    "p2_5StridesData.mat",
    "p3_5StridesData.mat",
    "p4_5StridesData.mat",
    "p5_5StridesData.mat",
    "p6_5StridesData.mat",
    "p7_5StridesData.mat",
    "p8_5StridesData.mat",
    "p9_5StridesData.mat",
]

def get_array(struct_void, field):
    """Extrait un champ — retourne None si impossible."""
    try:
        val = struct_void[field]
        val = np.array(val, dtype=np.float64)
        if val.ndim == 1:
            val = val.reshape(-1, 1)
        if val.size == 0:
            return None
        return val
    except Exception:
        return None

def extraire_condition(condition_raw):
    """Retourne (None, None, None) si condition invalide."""
    try:
        lmb_raw   = condition_raw['Link_Model_Based']
        force_raw = condition_raw['Force']

        if lmb_raw.size == 0 or force_raw.size == 0:
            return None, None, None

        lmb   = lmb_raw.ravel()[0]
        force = force_raw.ravel()[0]

        # Vérification rapide que le contenu est valide
        test = get_array(lmb, 'l_hip_angle')
        if test is None or test.shape[0] < 10:
            return None, None, None

    except Exception:
        return None, None, None

    # --- Angles ---
    l_hip_ang = get_array(lmb, 'l_hip_angle')
    r_hip_ang = get_array(lmb, 'r_hip_angle')
    l_kne_ang = get_array(lmb, 'l_kne_angle')
    r_kne_ang = get_array(lmb, 'r_kne_angle')
    l_ank_ang = get_array(lmb, 'l_ank_angle')
    r_ank_ang = get_array(lmb, 'r_ank_angle')

    # --- Vitesses ---
    l_hip_vel = get_array(lmb, 'l_hip_vel')
    r_hip_vel = get_array(lmb, 'r_hip_vel')
    l_kne_vel = get_array(lmb, 'l_kne_vel')
    r_kne_vel = get_array(lmb, 'r_kne_vel')
    l_ank_vel = get_array(lmb, 'l_ank_vel')
    r_ank_vel = get_array(lmb, 'r_ank_vel')

    # --- Couples ---
    l_hip_mom = get_array(lmb, 'l_hip_moment')
    r_hip_mom = get_array(lmb, 'r_hip_moment')
    l_kne_mom = get_array(lmb, 'l_kne_moment')
    r_kne_mom = get_array(lmb, 'r_kne_moment')
    l_ank_mom = get_array(lmb, 'l_ank_moment')
    r_ank_mom = get_array(lmb, 'r_ank_moment')

    # --- Puissances ---
    l_hip_pow = get_array(lmb, 'l_hip_power')
    r_hip_pow = get_array(lmb, 'r_hip_power')
    l_kne_pow = get_array(lmb, 'l_kne_power')
    r_kne_pow = get_array(lmb, 'r_kne_power')
    l_ank_pow = get_array(lmb, 'l_ank_power')
    r_ank_pow = get_array(lmb, 'r_ank_power')

    # --- GRF ---
    f1 = get_array(force, 'force1')
    f2 = get_array(force, 'force2')

    # Vérifier qu'aucun champ n'est None
    tous_champs = [l_hip_ang, r_hip_ang, l_kne_ang, r_kne_ang,
                   l_ank_ang, r_ank_ang, l_hip_vel, r_hip_vel,
                   l_kne_vel, r_kne_vel, l_ank_vel, r_ank_vel,
                   l_hip_mom, r_hip_mom, l_kne_mom, r_kne_mom,
                   l_ank_mom, r_ank_mom, l_hip_pow, r_hip_pow,
                   l_kne_pow, r_kne_pow, l_ank_pow, r_ank_pow,
                   f1, f2]
    if any(x is None for x in tous_champs):
        return None, None, None

    # Sous-échantillonnage GRF
    f1 = f1[::10]
    f2 = f2[::10]

    # --- Alignement ---
    T = min(x.shape[0] for x in [
        l_hip_ang, r_hip_ang, l_kne_ang, r_kne_ang,
        l_ank_ang, r_ank_ang, l_hip_vel, l_kne_vel,
        l_ank_vel, f1, f2
    ])

    def c(x): return x[:T]

    l_hip_ang = c(l_hip_ang); r_hip_ang = c(r_hip_ang)
    l_kne_ang = c(l_kne_ang); r_kne_ang = c(r_kne_ang)
    l_ank_ang = c(l_ank_ang); r_ank_ang = c(r_ank_ang)
    l_hip_vel = c(l_hip_vel); r_hip_vel = c(r_hip_vel)
    l_kne_vel = c(l_kne_vel); r_kne_vel = c(r_kne_vel)
    l_ank_vel = c(l_ank_vel); r_ank_vel = c(r_ank_vel)
    l_hip_mom = c(l_hip_mom); r_hip_mom = c(r_hip_mom)
    l_kne_mom = c(l_kne_mom); r_kne_mom = c(r_kne_mom)
    l_ank_mom = c(l_ank_mom); r_ank_mom = c(r_ank_mom)
    l_hip_pow = c(l_hip_pow); r_hip_pow = c(r_hip_pow)
    l_kne_pow = c(l_kne_pow); r_kne_pow = c(r_kne_pow)
    l_ank_pow = c(l_ank_pow); r_ank_pow = c(r_ank_pow)
    f1 = c(f1); f2 = c(f2)

    # --- État s(t) — 42 dims ---
    etats = np.concatenate([
        l_hip_ang, r_hip_ang,
        l_kne_ang, r_kne_ang,
        l_ank_ang, r_ank_ang,
        l_hip_vel, r_hip_vel,
        l_kne_vel, r_kne_vel,
        l_ank_vel, r_ank_vel,
        f1, f2,
    ], axis=1).astype(np.float32)

    # --- Action a(t) — 18 dims ---
    actions = np.concatenate([
        l_hip_mom, r_hip_mom,
        l_kne_mom, r_kne_mom,
        l_ank_mom, r_ank_mom,
    ], axis=1).astype(np.float32)

    # --- Récompenses proxy ---
    puissance_totale = (
        np.abs(l_hip_pow) + np.abs(r_hip_pow) +
        np.abs(l_kne_pow) + np.abs(r_kne_pow) +
        np.abs(l_ank_pow) + np.abs(r_ank_pow)
    ).sum(axis=1)
    asymetrie = np.abs(l_hip_pow - r_hip_pow).sum(axis=1)
    confort   = (l_hip_mom**2 + r_hip_mom**2).sum(axis=1)

    recomp = np.column_stack([
        -puissance_totale,
        -asymetrie,
        -confort,
    ]).astype(np.float32)

    return etats, actions, recomp


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

tous_etats     = []
toutes_actions = []
toutes_recomp  = []

for nom_fichier in PARTICIPANTS:
    chemin = os.path.join(DOSSIER_DATA, nom_fichier)
    if not os.path.exists(chemin):
        print(f"  MANQUANT : {nom_fichier}")
        continue

    print(f"\nTraitement : {nom_fichier}")
    raw  = sio.loadmat(chemin)
    data = raw['data']

    etats_sujet   = []
    actions_sujet = []
    recomp_sujet  = []
    n_ignores = 0

    for cond_idx in range(data.shape[1]):
        condition_raw = data[0, cond_idx]
        etats, actions, recomp = extraire_condition(condition_raw)

        if etats is None:
            n_ignores += 1
            continue

        etats_sujet.append(etats)
        actions_sujet.append(actions)
        recomp_sujet.append(recomp)

    if len(etats_sujet) == 0:
        print(f"  AUCUNE condition valide — ignoré")
        continue

    etats_sujet   = np.vstack(etats_sujet)
    actions_sujet = np.vstack(actions_sujet)
    recomp_sujet  = np.vstack(recomp_sujet)

    tous_etats.append(etats_sujet)
    toutes_actions.append(actions_sujet)
    toutes_recomp.append(recomp_sujet)

    print(f"  conditions valides : {data.shape[1] - n_ignores}/33")
    print(f"  états   : {etats_sujet.shape}")
    print(f"  actions : {actions_sujet.shape}")

print(f"\n{'='*50}")
print(f"Participants chargés : {len(tous_etats)}")

# ---- Normalisation ----
print("\nNormalisation en cours...")
all_etats   = np.vstack(tous_etats)
all_actions = np.vstack(toutes_actions)
all_recomp  = np.vstack(toutes_recomp)

scaler_etats   = StandardScaler().fit(all_etats)
scaler_actions = StandardScaler().fit(all_actions)

etats_norm   = scaler_etats.transform(all_etats)
actions_norm = scaler_actions.transform(all_actions)

# ---- Sauvegarde ----
np.save(f"{DOSSIER_PROCESSED}/etats.npy",       etats_norm.astype(np.float32))
np.save(f"{DOSSIER_PROCESSED}/actions.npy",     actions_norm.astype(np.float32))
np.save(f"{DOSSIER_PROCESSED}/recompenses.npy", all_recomp.astype(np.float32))

with open(f"{DOSSIER_PROCESSED}/scaler_etats.pkl",   'wb') as f:
    pickle.dump(scaler_etats, f)
with open(f"{DOSSIER_PROCESSED}/scaler_actions.pkl", 'wb') as f:
    pickle.dump(scaler_actions, f)

print(f"\nFichiers sauvegardés dans {DOSSIER_PROCESSED}/")
print(f"  etats.npy   : {etats_norm.shape}")
print(f"  actions.npy : {actions_norm.shape}")
print(f"\n✓ Phase 1 terminée ! Prête pour la Phase 2.")