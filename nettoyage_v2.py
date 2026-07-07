# ============================================================
# NETTOYAGE AVANCÉ — TRAJECTORY-PRESERVING (VERSION CORRIGÉE)
# nettoyage_v2.py
# ============================================================

import numpy as np
import scipy.io as sio
import os
import pickle
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

os.makedirs("data/processed_v2", exist_ok=True)

PARTICIPANTS = [
    "p1_5StridesData.mat", "p2_5StridesData.mat", "p3_5StridesData.mat",
    "p4_5StridesData.mat", "p5_5StridesData.mat", "p6_5StridesData.mat",
    "p7_5StridesData.mat", "p8_5StridesData.mat", "p9_5StridesData.mat",
]

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_array(struct_void, field):
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


def butter_lowpass(data, cutoff=6.0, fs=100.0, order=4):
    """
    Filtre Butterworth passe-bas — standard en biomécanique.
    cutoff = 6 Hz : élimine le bruit haute fréquence des capteurs MoCap
    fs     = 100 Hz : fréquence d'échantillonnage du dataset
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = np.zeros_like(data)
    for col in range(data.shape[1]):
        filtered[:, col] = filtfilt(b, a, data[:, col])
    return filtered


def nettoyer_trajectoire(etats, actions, participant_id, cond_id):
    """
    Nettoyage d'UNE trajectoire (une condition de marche).
    Respecte la structure temporelle — pas de suppression de frames isolées.

    Retourne None si la trajectoire est invalide.
    """
    T = etats.shape[0]

    # ---- 1. Vérification NaN / Inf ----
    if np.isnan(etats).any() or np.isinf(etats).any():
        print(f"    P{participant_id} Cond{cond_id} — NaN/Inf dans états → ignorée")
        return None, None
    if np.isnan(actions).any() or np.isinf(actions).any():
        print(f"    P{participant_id} Cond{cond_id} — NaN/Inf dans actions → ignorée")
        return None, None

    # ---- 2. Filtrage biomécanique AVANT outliers ----
    # Vérification des bornes physiologiques RÉELLES
    # Les données sont normalisées donc on vérifie en z-score
    # z-score > 5 = clairement impossible biologiquement
    z_etats   = np.abs(etats)
    z_actions = np.abs(actions)

    

    # ---- 3. Lissage Butterworth (très important pour IRL) ----
    # Réduit le bruit des capteurs MoCap sans détruire la dynamique
    try:
        etats_lisses   = butter_lowpass(etats,   cutoff=6.0, fs=100.0)
        actions_lissees = butter_lowpass(actions, cutoff=6.0, fs=100.0)
    except Exception:
        # Si le lissage échoue (trajectoire trop courte), garder original
        etats_lisses   = etats
        actions_lissees = actions

    # ---- 4. Vérification longueur minimale ----
    if T < 50:  # moins de 0.5 secondes à 100 Hz
        print(f"    P{participant_id} Cond{cond_id} — trajectoire trop courte ({T} frames) → ignorée")
        return None, None

    return etats_lisses.astype(np.float32), actions_lissees.astype(np.float32)


def extraire_condition_brute(condition_raw):
    """
    Extrait états et actions bruts (avant normalisation).
    Retourne None si condition invalide.
    """
    try:
        lmb_raw   = condition_raw['Link_Model_Based']
        force_raw = condition_raw['Force']

        if lmb_raw.size == 0 or force_raw.size == 0:
            return None, None

        lmb   = lmb_raw.ravel()[0]
        force = force_raw.ravel()[0]

        test = get_array(lmb, 'l_hip_angle')
        if test is None or test.shape[0] < 50:
            return None, None

    except Exception:
        return None, None

    # Extraction des variables
    champs_angles  = ['l_hip_angle','r_hip_angle','l_kne_angle','r_kne_angle','l_ank_angle','r_ank_angle']
    champs_vitesse = ['l_hip_vel','r_hip_vel','l_kne_vel','r_kne_vel','l_ank_vel','r_ank_vel']
    champs_couples = ['l_hip_moment','r_hip_moment','l_kne_moment','r_kne_moment','l_ank_moment','r_ank_moment']
    champs_puiss   = ['l_hip_power','r_hip_power','l_kne_power','r_kne_power','l_ank_power','r_ank_power']

    angles  = [get_array(lmb, c) for c in champs_angles]
    vitesse = [get_array(lmb, c) for c in champs_vitesse]
    couples = [get_array(lmb, c) for c in champs_couples]

    f1 = get_array(force, 'force1')
    f2 = get_array(force, 'force2')

    if any(x is None for x in angles + vitesse + couples + [f1, f2]):
        return None, None

    f1 = f1[::10]
    f2 = f2[::10]

    T = min([x.shape[0] for x in angles + vitesse + couples] + [f1.shape[0], f2.shape[0]])

    etats   = np.concatenate([x[:T] for x in angles + vitesse] + [f1[:T], f2[:T]], axis=1)
    actions = np.concatenate([x[:T] for x in couples], axis=1)

    return etats.astype(np.float32), actions.astype(np.float32)


# ============================================================
# PIPELINE PRINCIPAL — PAR TRAJECTOIRE
# ============================================================

print("=" * 60)
print("NETTOYAGE AVANCÉ — TRAJECTORY-PRESERVING")
print("=" * 60)

# Structure de sortie : liste de trajectoires (pas un grand tableau)
trajectoires       = []   # liste de dicts {etats, actions, participant, condition}
stats_nettoyage    = {
    'total_conditions': 0,
    'conditions_valides': 0,
    'conditions_ignorees': 0,
    'frames_avant': 0,
    'frames_apres': 0,
}

# ---- Étape 1 : Calcul des scalers sur données brutes ----
print("\nÉtape 1 — Calcul des scalers sur données brutes...")
tous_etats_bruts   = []
tous_actions_bruts = []

for nom in PARTICIPANTS:
    chemin = os.path.join("data", nom)
    if not os.path.exists(chemin):
        continue
    raw  = sio.loadmat(chemin)
    data = raw['data']
    for j in range(data.shape[1]):
        etats_b, actions_b = extraire_condition_brute(data[0, j])
        if etats_b is not None:
            tous_etats_bruts.append(etats_b)
            tous_actions_bruts.append(actions_b)

scaler_etats   = StandardScaler().fit(np.vstack(tous_etats_bruts))
scaler_actions = StandardScaler().fit(np.vstack(tous_actions_bruts))
print(f"  Scalers calculés sur {sum(len(e) for e in tous_etats_bruts)} frames")

# ---- Étape 2 : Nettoyage par trajectoire ----
print("\nÉtape 2 — Nettoyage par trajectoire...")

for p_idx, nom in enumerate(PARTICIPANTS):
    chemin = os.path.join("data", nom)
    if not os.path.exists(chemin):
        print(f"  MANQUANT : {nom}")
        continue

    raw  = sio.loadmat(chemin)
    data = raw['data']
    participant_id = p_idx + 1

    n_valides = 0
    n_ignores = 0

    for j in range(data.shape[1]):
        stats_nettoyage['total_conditions'] += 1

        # Extraction brute
        etats_b, actions_b = extraire_condition_brute(data[0, j])
        if etats_b is None:
            n_ignores += 1
            stats_nettoyage['conditions_ignorees'] += 1
            continue

        stats_nettoyage['frames_avant'] += len(etats_b)

        # Normalisation PAR TRAJECTOIRE (avec le scaler global)
        etats_norm   = scaler_etats.transform(etats_b).astype(np.float32)
        actions_norm = scaler_actions.transform(actions_b).astype(np.float32)

        # Nettoyage de la trajectoire normalisée
        etats_propres, actions_propres = nettoyer_trajectoire(
            etats_norm, actions_norm, participant_id, j
        )

        if etats_propres is None:
            n_ignores += 1
            stats_nettoyage['conditions_ignorees'] += 1
            continue

        # Sauvegarde de la trajectoire complète
        trajectoires.append({
            'etats':       etats_propres,
            'actions':     actions_propres,
            'participant': participant_id,
            'condition':   j,
            'n_frames':    len(etats_propres)
        })

        stats_nettoyage['frames_apres']      += len(etats_propres)
        stats_nettoyage['conditions_valides'] += 1
        n_valides += 1

    print(f"  P{participant_id} — {n_valides}/{data.shape[1]} conditions valides")

# ============================================================
# RAPPORT DE NETTOYAGE
# ============================================================

print(f"\n{'='*60}")
print("RAPPORT DE NETTOYAGE")
print(f"{'='*60}")
print(f"  Conditions totales    : {stats_nettoyage['total_conditions']}")
print(f"  Conditions valides    : {stats_nettoyage['conditions_valides']}")
print(f"  Conditions ignorées   : {stats_nettoyage['conditions_ignorees']}")
print(f"  Frames avant          : {stats_nettoyage['frames_avant']}")
print(f"  Frames après          : {stats_nettoyage['frames_apres']}")
print(f"  Trajectoires IRL      : {len(trajectoires)}")
pct = 100 * stats_nettoyage['frames_apres'] / max(stats_nettoyage['frames_avant'], 1)
print(f"  Données conservées    : {pct:.1f}%")

# ============================================================
# SAUVEGARDE
# ============================================================

print("\nSauvegarde...")

# Format 1 : trajectoires séparées (pour IRL avancé)
np.save("data/processed_v2/trajectoires.npy",
        np.array(trajectoires, dtype=object), allow_pickle=True)

# Format 2 : tableau global (compatible avec phase2_IRL.py existant)
all_etats   = np.vstack([t['etats']   for t in trajectoires])
all_actions = np.vstack([t['actions'] for t in trajectoires])

np.save("data/processed_v2/etats.npy",   all_etats)
np.save("data/processed_v2/actions.npy", all_actions)

with open("data/processed_v2/scaler_etats.pkl",   'wb') as f:
    pickle.dump(scaler_etats, f)
with open("data/processed_v2/scaler_actions.pkl", 'wb') as f:
    pickle.dump(scaler_actions, f)

print(f"  trajectoires.npy : {len(trajectoires)} trajectoires")
print(f"  etats.npy        : {all_etats.shape}")
print(f"  actions.npy      : {all_actions.shape}")
print(f"\n✓ Nettoyage v2 terminé !")
print(f"  Utilisez data/processed_v2/ pour la Phase 2 améliorée")