"""
Phase 5 — Importance des articulations (Feature Importance) VERSION V2
=======================================================================
Objectif : utiliser les gradients du réseau IRL pour quantifier
la contribution de chaque articulation (hanche / genou / cheville).

Méthode : Gradient × Input (méthode de saillance standard en deep learning)
    importance(i) = |∂R/∂x_i| × |x_i|   moyenné sur toutes les frames

Corrections v2 :
  - Charge depuis processed_v2/ et reward_network_v2.pth
  - Architecture identique à Phase 2 v2 (avec Dropout)
  - Utilise trajectoires.npy pour un échantillonnage équilibré par participant

Exécution :
    python phase5_importance_features.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─── Chemins ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed_v2"
RESULTATS_DIR = "resultats"
OUTPUT_DIR    = os.path.join(RESULTATS_DIR, "poster")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Architecture IDENTIQUE à Phase 2 v2 (avec Dropout) ──────────────────────
class RewardNetwork(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

# ─── Chargement ───────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 5 — Importance des articulations (v2)")
print("=" * 60)

etats        = np.load(os.path.join(PROCESSED_DIR, "etats.npy"))
actions      = np.load(os.path.join(PROCESSED_DIR, "actions.npy"))
trajectoires = np.load(os.path.join(PROCESSED_DIR, "trajectoires.npy"), allow_pickle=True)

model = RewardNetwork(input_dim=60)
model.load_state_dict(torch.load(
    os.path.join(RESULTATS_DIR, "reward_network_v2.pth"),
    map_location="cpu", weights_only=True
))
model.eval()  # Désactive le Dropout pour le calcul des gradients

n_params = sum(p.numel() for p in model.parameters())
print(f"\n  Modèle chargé     : {n_params} paramètres")
print(f"  États             : {etats.shape}")
print(f"  Actions           : {actions.shape}")
print(f"  Trajectoires      : {len(trajectoires)}")

# ─── Noms des 60 features [s(t) 42-dim + a(t) 18-dim] ───────────────────────
feature_names = []

# États — angles (6 dims par articulation : 3 axes × G/D)
for cote in ["G", "D"]:
    for artic in ["hanche", "genou", "cheville"]:
        for dim in ["x", "y", "z"]:
            feature_names.append(f"angle_{artic}_{cote}_{dim}")

# États — vitesses angulaires
for cote in ["G", "D"]:
    for artic in ["hanche", "genou", "cheville"]:
        for dim in ["x", "y", "z"]:
            feature_names.append(f"vit_{artic}_{cote}_{dim}")

# États — GRF (forces de réaction au sol)
for i in range(3):
    feature_names.append(f"GRF1_{['x','y','z'][i]}")
for i in range(3):
    feature_names.append(f"GRF2_{['x','y','z'][i]}")

# Actions — couples articulaires
for cote in ["G", "D"]:
    for artic in ["hanche", "genou", "cheville"]:
        for dim in ["x", "y", "z"]:
            feature_names.append(f"couple_{artic}_{cote}_{dim}")

assert len(feature_names) == 60, f"Attendu 60 features, obtenu {len(feature_names)}"
print(f"  Features nommées  : {len(feature_names)}")

# ─── Échantillonnage équilibré par participant ────────────────────────────────
# On prend le même nombre de frames par participant pour éviter les biais
# (un participant avec plus de conditions ne doit pas dominer l'importance)

print("\nÉchantillonnage équilibré par participant...")

offsets = {}
idx = 0
for traj in trajectoires:
    key = (traj['participant'], traj['condition'])
    offsets[key] = idx
    idx += traj['n_frames']

participants_data = {}
for traj in trajectoires:
    p = traj['participant']
    if p not in participants_data:
        participants_data[p] = []
    participants_data[p].append(traj)

# 2000 frames par participant = 18 000 frames au total (9 participants)
N_PAR_PARTICIPANT = 2000
indices_echantillon = []

for p in sorted(participants_data.keys()):
    trajs = participants_data[p]
    indices_p = []
    for traj in trajs:
        key   = (traj['participant'], traj['condition'])
        start = offsets[key]
        end   = start + traj['n_frames']
        indices_p.extend(range(start, end))

    # Sous-échantillonnage aléatoire
    if len(indices_p) > N_PAR_PARTICIPANT:
        indices_p = list(np.random.choice(indices_p, N_PAR_PARTICIPANT, replace=False))

    indices_echantillon.extend(indices_p)
    print(f"  P{p} : {len(indices_p)} frames sélectionnées")

indices_echantillon = np.array(indices_echantillon)
print(f"  Total : {len(indices_echantillon)} frames")

# ─── Calcul des gradients (Gradient × Input) ─────────────────────────────────
print("\nCalcul des gradients (Gradient × Input)...")

BATCH_SIZE      = 2048
etats_s         = etats[indices_echantillon]
actions_s       = actions[indices_echantillon]
importances_all = []

for start in range(0, len(indices_echantillon), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(indices_echantillon))
    sa  = np.concatenate([etats_s[start:end], actions_s[start:end]], axis=1)
    x   = torch.tensor(sa, dtype=torch.float32, requires_grad=True)

    r   = model(x)
    r.sum().backward()

    # Gradient × Input = saillance de chaque feature
    grad_input = (x.grad.detach().abs() * x.detach().abs()).numpy()
    importances_all.append(grad_input)

importances = np.concatenate(importances_all, axis=0)  # (N, 60)
imp_mean    = importances.mean(axis=0)                 # (60,)
imp_std     = importances.std(axis=0)

print(f"  ✓ Gradients calculés sur {len(importances)} frames")

# ─── Agrégation par groupe de features ───────────────────────────────────────
groupes = {
    "Hanche (angles)":   list(range(0, 6)),
    "Genou (angles)":    list(range(6, 12)),
    "Cheville (angles)": list(range(12, 18)),
    "Hanche (vitesses)": list(range(18, 24)),
    "Genou (vitesses)":  list(range(24, 30)),
    "Cheville (vit.)":   list(range(30, 36)),
    "GRF (forces)":      list(range(36, 42)),
    "Couple hanche":     list(range(42, 48)),
    "Couple genou":      list(range(48, 54)),
    "Couple cheville":   list(range(54, 60)),
}

imp_groupes = {nom: float(imp_mean[indices].sum()) for nom, indices in groupes.items()}
total_imp   = sum(imp_groupes.values())
imp_pct     = {k: 100 * v / total_imp for k, v in imp_groupes.items()}

# Agrégation par grande articulation (angles + vitesses + couples)
cat_articulaire = {
    "Hanche":   imp_pct["Hanche (angles)"]   + imp_pct["Hanche (vitesses)"] + imp_pct["Couple hanche"],
    "Genou":    imp_pct["Genou (angles)"]    + imp_pct["Genou (vitesses)"]  + imp_pct["Couple genou"],
    "Cheville": imp_pct["Cheville (angles)"] + imp_pct["Cheville (vit.)"]  + imp_pct["Couple cheville"],
    "GRF":      imp_pct["GRF (forces)"],
}

# ─── Affichage des résultats ──────────────────────────────────────────────────
print(f"\n{'─'*45}")
print("Importance par groupe de features :")
print(f"{'─'*45}")
for nom, pct in sorted(imp_pct.items(), key=lambda x: -x[1]):
    barre = "█" * int(pct / 2)
    print(f"  {nom:<25} {pct:6.2f}%  {barre}")

print(f"\n{'─'*45}")
print("Importance par articulation :")
print(f"{'─'*45}")
for artic, pct in sorted(cat_articulaire.items(), key=lambda x: -x[1]):
    barre = "█" * int(pct / 2)
    print(f"  {artic:<15} {pct:6.2f}%  {barre}")

# ─── Figures ──────────────────────────────────────────────────────────────────
print("\nGénération des figures...")

COULEURS = {
    "hanche":   "#534AB7",
    "genou":    "#1D9E75",
    "cheville": "#D85A30",
    "grf":      "#BA7517",
}

def couleur_groupe(nom):
    n = nom.lower()
    if "hanche" in n:   return COULEURS["hanche"]
    if "genou" in n:    return COULEURS["genou"]
    if "cheville" in n: return COULEURS["cheville"]
    return COULEURS["grf"]

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    "Phase 5 — Importance des articulations | IRL Biomécanique\n"
    "Méthode : Gradient × Input sur le réseau de récompense appris",
    fontsize=14, fontweight='bold'
)

# ── Graphique 1 : Barplot horizontal par groupe (toute la largeur) ────────────
ax1 = fig.add_subplot(gs[0, :])

noms_tries = sorted(imp_pct.keys(), key=lambda k: -imp_pct[k])
vals_tries = [imp_pct[k] for k in noms_tries]
cols_tries = [couleur_groupe(k) for k in noms_tries]

bars = ax1.barh(noms_tries, vals_tries, color=cols_tries, alpha=0.82, edgecolor='white')
for bar, val in zip(bars, vals_tries):
    ax1.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va='center', fontsize=11, fontweight='bold')

ax1.set_xlabel("Contribution à R(s,a) (%)", fontsize=12)
ax1.set_title(
    "Contribution de chaque groupe de variables à la récompense apprise R(s,a)\n"
    "Plus le pourcentage est élevé, plus le modèle s'appuie sur ce groupe",
    fontsize=11
)
ax1.set_xlim(0, max(vals_tries) * 1.18)
ax1.grid(alpha=0.3, axis='x')

# Légende des couleurs
from matplotlib.patches import Patch
legende = [Patch(color=COULEURS[k], label=k.capitalize(), alpha=0.82)
           for k in ["hanche", "genou", "cheville", "grf"]]
ax1.legend(handles=legende, fontsize=10, loc='lower right')

# ── Graphique 2 : Camembert par articulation ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

labels_pie   = list(cat_articulaire.keys())
sizes_pie    = [cat_articulaire[k] for k in labels_pie]
cols_pie     = [COULEURS[k.lower()] if k.lower() in COULEURS else COULEURS["grf"]
                for k in labels_pie]
explode      = [0.05] * len(labels_pie)

wedges, texts, autotexts = ax2.pie(
    sizes_pie, labels=labels_pie, colors=cols_pie,
    autopct='%1.1f%%', startangle=90, explode=explode,
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')

ax2.set_title(
    "Contribution par articulation\n(angles + vitesses + couples combinés)",
    fontsize=12, fontweight='bold'
)

# ── Graphique 3 : Top 15 features individuelles ───────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

top_k       = 15
top_idx     = np.argsort(imp_mean)[::-1][:top_k]
top_nms     = [feature_names[i] for i in top_idx]
top_vls_pct = 100 * imp_mean[top_idx] / imp_mean.sum()
top_cols    = [couleur_groupe(n) for n in top_nms]

ax3.barh(top_nms[::-1], top_vls_pct[::-1],
         color=top_cols[::-1], alpha=0.8, edgecolor='white')
ax3.set_xlabel("Contribution (%)", fontsize=11)
ax3.set_title(
    f"Top {top_k} features individuelles\n(Gradient × Input)",
    fontsize=12, fontweight='bold'
)
ax3.tick_params(axis='y', labelsize=8)
ax3.grid(alpha=0.3, axis='x')

plt.savefig(os.path.join(OUTPUT_DIR, "figure5_importance.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure sauvegardée : resultats/poster/figure5_importance.png")

# ─── Sauvegarde ───────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTATS_DIR, "phase5_importances.npy"), {
    "imp_mean":        imp_mean,
    "imp_std":         imp_std,
    "imp_groupes_pct": imp_pct,
    "cat_articulaire": cat_articulaire,
    "feature_names":   feature_names,
}, allow_pickle=True)
print("  ✓ Résultats sauvegardés : resultats/phase5_importances.npy")

# ─── Résumé final ─────────────────────────────────────────────────────────────
artic_dominant = max(cat_articulaire, key=cat_articulaire.get)

print(f"\n{'='*60}")
print("RÉSUMÉ PHASE 5")
print(f"{'='*60}")
for artic, pct in sorted(cat_articulaire.items(), key=lambda x: -x[1]):
    print(f"  {artic:<15} : {pct:.1f}%")
print(f"\n  Articulation dominante : {artic_dominant} ({cat_articulaire[artic_dominant]:.1f}%)")
print(f"\n  Phrase pour le poster :")
print(f'  "La {artic_dominant.lower()} contribue à {cat_articulaire[artic_dominant]:.0f}% ')
print(f'   de la fonction de récompense apprise R(s,a)"')
print(f"\n✓ Phase 5 terminée ! → Prochaine étape : python phase6_score_normalite.py")