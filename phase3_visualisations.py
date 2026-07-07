# ============================================================
# PHASE 3 — VISUALISATIONS AVANCÉES POUR LE POSTER (VERSION V2)
# Charge depuis processed_v2/ et reward_network_v2.pth
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scipy.io as sio
import torch
import torch.nn as nn
import os

os.makedirs("resultats/poster", exist_ok=True)

# ============================================================
# 1. CHARGEMENT DU MODÈLE ET DES DONNÉES (V2)
# ============================================================

class RewardNetwork(nn.Module):
    """Même architecture que Phase 2 v2 — avec Dropout."""
    def __init__(self, input_dim=60, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

print("=" * 60)
print("PHASE 3 — Visualisations (données v2)")
print("=" * 60)

print("\nChargement du modèle et des données...")

reward_net = RewardNetwork()
reward_net.load_state_dict(torch.load(
    "resultats/reward_network_v2.pth", weights_only=True
))
reward_net.eval()

# Données v2
etats        = np.load("data/processed_v2/etats.npy")
actions      = np.load("data/processed_v2/actions.npy")
recompenses  = np.load("resultats/recompenses_apprises_v2.npy")
trajectoires = np.load("data/processed_v2/trajectoires.npy", allow_pickle=True)

# Récompenses proxy — calculées directement depuis les données v2
print("  Calcul des récompenses proxy...")
n = len(etats)
recomp_proxy = np.zeros((n, 3), dtype=np.float32)
# Proxy 1 : efficacité (opposé des couples moyens)
recomp_proxy[:, 0] = -np.abs(actions).mean(axis=1)
# Proxy 2 : symétrie G/D (différence angles gauche vs droite)
angles_G = etats[:, [0, 2, 4]]
angles_D = etats[:, [1, 3, 5]]
recomp_proxy[:, 1] = -np.abs(angles_G - angles_D).mean(axis=1)
# Proxy 3 : confort (opposé des couples extrêmes)
recomp_proxy[:, 2] = -np.abs(actions).max(axis=1)
print(f"  Proxy calculés : {recomp_proxy.shape}")

print(f"  {len(etats)} frames chargées")
print(f"  {len(trajectoires)} trajectoires")
print(f"  Récompenses : min={recompenses.min():.2f}, max={recompenses.max():.2f}")

# ============================================================
# 2. EXTRACTION DES DONNÉES PAR CONDITION — via trajectoires.npy
# ============================================================

print("\nExtraction des données par condition...")

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

# Calcul des offsets de chaque trajectoire dans le tableau global
offsets = {}
idx = 0
for traj in trajectoires:
    key = (traj['participant'], traj['condition'])
    offsets[key] = idx
    idx += traj['n_frames']

# Charger p1 pour les analyses détaillées
raw  = sio.loadmat("data/p1_5StridesData.mat")
data = raw['data']

angles_par_condition  = []
couples_par_condition = []
recomp_par_condition  = []
indices_valides       = []

# Filtrer les trajectoires du participant 1
trajs_p1 = [t for t in trajectoires if t['participant'] == 1]

for traj in trajs_p1:
    j       = traj['condition']
    cond    = data[0, j]
    lmb_raw = cond['Link_Model_Based']
    if lmb_raw.size == 0:
        continue
    try:
        lmb = lmb_raw.ravel()[0]

        l_hip     = get_array(lmb, 'l_hip_angle')
        r_hip     = get_array(lmb, 'r_hip_angle')
        l_kne     = get_array(lmb, 'l_kne_angle')
        l_hip_mom = get_array(lmb, 'l_hip_moment')

        if any(x is None for x in [l_hip, r_hip, l_kne, l_hip_mom]):
            continue
        if l_hip.shape[0] < 10:
            continue

        # Récompense de cette trajectoire via les offsets
        key   = (1, j)
        start = offsets[key]
        end   = start + traj['n_frames']
        r_moy = recompenses[start:end].mean()

        angles_par_condition.append({
            'l_hip': l_hip[:, 0],
            'r_hip': r_hip[:, 0],
            'l_kne': l_kne[:, 0],
        })
        couples_par_condition.append(l_hip_mom[:, 0])
        recomp_par_condition.append(r_moy)
        indices_valides.append(j)

    except Exception:
        continue

print(f"  {len(angles_par_condition)} conditions valides pour P1")

# Normalisation pour la colormap
recomp_array = np.array(recomp_par_condition)
norm         = Normalize(vmin=recomp_array.min(), vmax=recomp_array.max())
cmap         = plt.cm.RdYlGn

# 5 conditions représentatives
n_cond   = len(angles_par_condition)
idx_cond = np.linspace(0, n_cond - 1, 5, dtype=int)


# ============================================================
# 3. FIGURE 1 — ANALYSE BIOMÉCANIQUE PAR CONDITION
# ============================================================

print("\nCréation Figure 1 — Analyse biomécanique...")

fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle(
    "Analyse Biomécanique de la Marche — Participant 1\n"
    "Angles et Récompenses IRL par Condition (données filtrées Butterworth 6Hz)",
    fontsize=13, fontweight='bold'
)

# --- Graphique 1 : Angle hanche gauche ---
ax = axes[0, 0]
for ic in idx_cond:
    ang   = angles_par_condition[ic]['l_hip']
    t     = np.linspace(0, 100, len(ang))
    r     = recomp_par_condition[ic]
    ax.plot(t, ang, color=cmap(norm(r)), linewidth=1.5,
            label=f"Cond {indices_valides[ic]} (R={r:.1f})", alpha=0.85)
ax.set_title("Angle hanche gauche", fontsize=11)
ax.set_xlabel("% cycle de marche")
ax.set_ylabel("Angle (°)")
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

# --- Graphique 2 : Angle genou gauche ---
ax = axes[0, 1]
for ic in idx_cond:
    ang = angles_par_condition[ic]['l_kne']
    t   = np.linspace(0, 100, len(ang))
    r   = recomp_par_condition[ic]
    ax.plot(t, ang, color=cmap(norm(r)), linewidth=1.5, alpha=0.85)
ax.set_title("Angle genou gauche", fontsize=11)
ax.set_xlabel("% cycle de marche")
ax.set_ylabel("Angle (°)")
ax.grid(alpha=0.3)

# --- Graphique 3 : Couple hanche gauche ---
ax = axes[0, 2]
for ic in idx_cond:
    mom = couples_par_condition[ic]
    t   = np.linspace(0, 100, len(mom))
    r   = recomp_par_condition[ic]
    ax.plot(t, mom, color=cmap(norm(r)), linewidth=1.5, alpha=0.85)
ax.set_title("Couple hanche gauche", fontsize=11)
ax.set_xlabel("% cycle de marche")
ax.set_ylabel("Couple (N·m)")
ax.grid(alpha=0.3)

# --- Graphique 4 : Récompense par condition ---
ax = axes[1, 0]
colors_bar = [cmap(norm(r)) for r in recomp_par_condition]
ax.bar(range(len(recomp_par_condition)), recomp_par_condition,
       color=colors_bar, edgecolor='white', linewidth=0.5)
ax.set_title("R(s,a) moyenne par condition de marche", fontsize=11)
ax.set_xlabel("Index condition")
ax.set_ylabel("Récompense R(s,a)")
ax.grid(alpha=0.3, axis='y')
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="R(s,a)")

# --- Graphique 5 : Asymétrie hanche G vs D ---
ax = axes[1, 1]
for ic in idx_cond:
    l     = angles_par_condition[ic]['l_hip']
    r_ang = angles_par_condition[ic]['r_hip']
    T     = min(len(l), len(r_ang))
    t     = np.linspace(0, 100, T)
    r_val = recomp_par_condition[ic]
    ax.plot(t, np.abs(l[:T] - r_ang[:T]),
            color=cmap(norm(r_val)), linewidth=1.5, alpha=0.85)
ax.set_title("Asymétrie hanche G/D", fontsize=11)
ax.set_xlabel("% cycle de marche")
ax.set_ylabel("|Hanche_G - Hanche_D| (°)")
ax.grid(alpha=0.3)

# --- Graphique 6 : Distribution R(s,a) globale ---
ax = axes[1, 2]
ax.hist(recompenses, bins=80, color='steelblue',
        alpha=0.8, edgecolor='white', density=True)
ax.axvline(recompenses.mean(), color='tomato', linewidth=2,
           linestyle='--', label=f"Moyenne = {recompenses.mean():.2f}")
ax.set_title("Distribution globale R(s,a) — 9 participants", fontsize=11)
ax.set_xlabel("Récompense R(s,a)")
ax.set_ylabel("Densité")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("resultats/poster/figure1_biomecanique.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 1 sauvegardée")


# ============================================================
# 4. FIGURE 2 — SQUELETTE 2D COLORÉ PAR RÉCOMPENSE
# ============================================================

print("Création Figure 2 — Squelette 2D...")

def dessiner_squelette(ax, l_hip, r_hip, l_kne, r_kne, recompense, titre, norm, cmap):
    longueur_cuisse = 0.4
    longueur_jambe  = 0.4
    hauteur_bassin  = 1.0

    def angle_to_pos(angle_deg, longueur, x0, y0):
        angle_rad = np.radians(angle_deg)
        x = x0 + longueur * np.sin(angle_rad)
        y = y0 - longueur * np.cos(angle_rad)
        return x, y

    color = cmap(norm(recompense))

    lhip_x, lhip_y = -0.1, hauteur_bassin
    lkne_x, lkne_y = angle_to_pos(l_hip, longueur_cuisse, lhip_x, lhip_y)
    lank_x, lank_y = angle_to_pos(l_kne, longueur_jambe,  lkne_x, lkne_y)

    rhip_x, rhip_y = 0.1, hauteur_bassin
    rkne_x, rkne_y = angle_to_pos(r_hip, longueur_cuisse, rhip_x, rhip_y)
    rank_x, rank_y = angle_to_pos(r_kne, longueur_jambe,  rkne_x, rkne_y)

    tronc_x, tronc_y = 0, hauteur_bassin + 0.5
    tete_x,  tete_y  = 0, hauteur_bassin + 0.7

    lw = 3
    # Jambe gauche (colorée par récompense)
    ax.plot([lhip_x, lkne_x], [lhip_y, lkne_y], color=color, linewidth=lw)
    ax.plot([lkne_x, lank_x], [lkne_y, lank_y], color=color, linewidth=lw)
    # Jambe droite (bleue fixe)
    ax.plot([rhip_x, rkne_x], [rhip_y, rkne_y], color='steelblue', linewidth=lw, alpha=0.5)
    ax.plot([rkne_x, rank_x], [rkne_y, rank_y], color='steelblue', linewidth=lw, alpha=0.5)
    # Bassin + tronc
    ax.plot([lhip_x, rhip_x], [lhip_y, rhip_y], color='gray', linewidth=lw)
    ax.plot([0, tronc_x], [hauteur_bassin, tronc_y], color='gray', linewidth=lw)
    ax.add_patch(plt.Circle((tete_x, tete_y), 0.06, color='gray', fill=True))

    for px, py in [(lhip_x, lhip_y), (lkne_x, lkne_y), (lank_x, lank_y)]:
        ax.plot(px, py, 'o', color=color, markersize=8,
                markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.15, 1.9)
    ax.set_aspect('equal')
    ax.set_title(f"{titre}\nR = {recompense:.2f}", fontsize=10)
    ax.axis('off')

fig2, axes2 = plt.subplots(1, 5, figsize=(18, 6))
fig2.suptitle(
    "Squelette 2D coloré par Récompense IRL\n"
    "Rouge = faible récompense  |  Vert = haute récompense",
    fontsize=13, fontweight='bold'
)

for ax, ic in zip(axes2, idx_cond):
    ang = angles_par_condition[ic]
    dessiner_squelette(
        ax,
        l_hip=ang['l_hip'].mean(),
        r_hip=ang['r_hip'].mean(),
        l_kne=ang['l_kne'].mean(),
        r_kne=ang['l_kne'].mean(),
        recompense=recomp_par_condition[ic],
        titre=f"Condition {indices_valides[ic]}",
        norm=norm, cmap=cmap
    )

sm2 = ScalarMappable(cmap=cmap, norm=norm)
sm2.set_array([])
fig2.colorbar(sm2, ax=axes2.tolist(), shrink=0.6, label="Récompense R(s,a)")

plt.tight_layout()
plt.savefig("resultats/poster/figure2_squelette.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 2 sauvegardée")


# ============================================================
# 5. FIGURE 3 — INTERPRÉTATION DE LA RÉCOMPENSE APPRISE
# ============================================================

print("Création Figure 3 — Interprétation...")

# Aligner les tailles (recomp_proxy vient de processed v1)
n_min = min(len(recomp_proxy), len(recompenses))

correlations = [
    np.corrcoef(recomp_proxy[:n_min, 0], recompenses[:n_min])[0, 1],
    np.corrcoef(recomp_proxy[:n_min, 1], recompenses[:n_min])[0, 1],
    np.corrcoef(recomp_proxy[:n_min, 2], recompenses[:n_min])[0, 1],
]

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle(
    "Interprétation de la Fonction de Récompense Apprise\n"
    "Corrélation R(s,a) IRL vs composantes biomécaniques manuelles",
    fontsize=13, fontweight='bold'
)

labels  = ["Efficacité énergétique (proxy)", "Symétrie gauche/droite (proxy)", "Confort articulaire (proxy)"]
couleurs = ['steelblue', 'tomato', 'seagreen']

for i, (ax, label, couleur, corr) in enumerate(zip(axes3, labels, couleurs, correlations)):
    n_plot = min(3000, n_min)
    ax.scatter(recomp_proxy[:n_plot, i], recompenses[:n_plot],
               alpha=0.3, s=5, color=couleur)
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("R(s,a) apprise par IRL", fontsize=10)
    ax.set_title(f"R(s,a) vs {label.split(' (')[0]}", fontsize=11)
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, f"r = {corr:.3f}",
            transform=ax.transAxes, fontsize=12,
            color='black', fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig("resultats/poster/figure3_interpretation.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 3 sauvegardée")


# ============================================================
# 6. FIGURE 4 — RÉSUMÉ PRINCIPAL POUR LE POSTER
# ============================================================

print("Création Figure 4 — Résumé poster...")

fig4 = plt.figure(figsize=(20, 12))
gs   = gridspec.GridSpec(2, 4, figure=fig4, hspace=0.4, wspace=0.35)
fig4.suptitle(
    "Apprentissage par Renforcement Inverse (MaxEnt IRL)\n"
    "pour l'Analyse Biomécanique de la Marche Humaine",
    fontsize=16, fontweight='bold', y=0.98
)

# --- Panel A : Pipeline IRL ---
ax_a = fig4.add_subplot(gs[0, 0])
ax_a.axis('off')
ax_a.set_title("A — Pipeline IRL", fontsize=12, fontweight='bold')
pipeline = [
    "Données MoCap\n(9 sujets, 33 conditions)",
    "États s(t)\n(angles, vitesses, GRF)\n42 dims",
    "Actions a(t)\n(couples articulaires)\n18 dims",
    "MaxEnt IRL\n(réseau 60→128→128→64→1)",
    "Récompense R(s,a)\napprise automatiquement"
]
couleurs_pip = ['#AED6F1', '#A9DFBF', '#A9DFBF', '#F9E79F', '#F1948A']
for k, (texte, col) in enumerate(zip(pipeline, couleurs_pip)):
    y_pos = 0.90 - k * 0.17
    ax_a.text(0.5, y_pos, texte, transform=ax_a.transAxes,
              ha='center', va='center', fontsize=8,
              bbox=dict(boxstyle='round,pad=0.3', facecolor=col, alpha=0.85))
    if k < len(pipeline) - 1:
        ax_a.annotate("", xy=(0.5, y_pos - 0.055),
                       xytext=(0.5, y_pos - 0.02),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# --- Panel B : R(s,a) sur 500 pas ---
ax_b = fig4.add_subplot(gs[0, 1])
ax_b.plot(recompenses[:500], color='steelblue', linewidth=1, alpha=0.7)
ax_b.axhline(recompenses[:500].mean(), color='tomato', linewidth=1.5,
             linestyle='--', label=f"μ = {recompenses[:500].mean():.2f}")
ax_b.set_title("B — R(s,a) sur 500 frames", fontsize=12, fontweight='bold')
ax_b.set_xlabel("Frame")
ax_b.set_ylabel("R(s,a)")
ax_b.legend(fontsize=9)
ax_b.grid(alpha=0.3)

# --- Panel C : Distribution R(s,a) ---
ax_c = fig4.add_subplot(gs[0, 2])
ax_c.hist(recompenses, bins=60, color='steelblue',
          alpha=0.8, edgecolor='white', density=True)
ax_c.axvline(recompenses.mean(), color='tomato', linewidth=2,
             linestyle='--', label=f"μ = {recompenses.mean():.2f}")
ax_c.axvline(recompenses.mean() - recompenses.std(), color='orange',
             linewidth=1.5, linestyle=':', label=f"±σ = {recompenses.std():.2f}")
ax_c.axvline(recompenses.mean() + recompenses.std(), color='orange',
             linewidth=1.5, linestyle=':')
ax_c.set_title("C — Distribution R(s,a)", fontsize=12, fontweight='bold')
ax_c.set_xlabel("R(s,a)")
ax_c.set_ylabel("Densité")
ax_c.legend(fontsize=8)
ax_c.grid(alpha=0.3)

# --- Panel D : R(s,a) par participant (via trajectoires) ---
ax_d = fig4.add_subplot(gs[0, 3])

participants_data = {}
for traj in trajectoires:
    p = traj['participant']
    if p not in participants_data:
        participants_data[p] = []
    participants_data[p].append(traj)

r_par_p   = []
noms_par_p = []
for p in sorted(participants_data.keys()):
    all_r = []
    for traj in participants_data[p]:
        key   = (traj['participant'], traj['condition'])
        start = offsets[key]
        end   = start + traj['n_frames']
        all_r.append(recompenses[start:end])
    r_par_p.append(np.concatenate(all_r).mean())
    noms_par_p.append(f"P{p}")

colors_p = plt.cm.Blues(np.linspace(0.4, 0.9, len(r_par_p)))
bars = ax_d.bar(noms_par_p, r_par_p, color=colors_p, edgecolor='white')
ax_d.set_title("D — R(s,a) par participant", fontsize=12, fontweight='bold')
ax_d.set_xlabel("Participant")
ax_d.set_ylabel("R(s,a) moyen")
ax_d.grid(alpha=0.3, axis='y')
y_off = (max(r_par_p) - min(r_par_p)) * 0.05
for bar, val in zip(bars, r_par_p):
    ax_d.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + y_off,
              f"{val:.2f}", ha='center', va='bottom', fontsize=7)

# --- Panel E : Condition R_min vs R_max ---
ax_e = fig4.add_subplot(gs[1, 0:2])
ic_min = int(np.argmin(recomp_par_condition))
ic_max = int(np.argmax(recomp_par_condition))
ang_min = angles_par_condition[ic_min]['l_hip']
ang_max = angles_par_condition[ic_max]['l_hip']
ax_e.plot(np.linspace(0, 100, len(ang_min)), ang_min,
          color='tomato', linewidth=2,
          label=f"R faible = {recomp_par_condition[ic_min]:.2f} (Cond {indices_valides[ic_min]})")
ax_e.plot(np.linspace(0, 100, len(ang_max)), ang_max,
          color='seagreen', linewidth=2,
          label=f"R élevée = {recomp_par_condition[ic_max]:.2f} (Cond {indices_valides[ic_max]})")
ax_e.set_title("E — Hanche G : condition R_min vs R_max",
               fontsize=12, fontweight='bold')
ax_e.set_xlabel("% cycle de marche")
ax_e.set_ylabel("Angle hanche (°)")
ax_e.legend(fontsize=9)
ax_e.grid(alpha=0.3)

# --- Panel F : Corrélations ---
ax_f = fig4.add_subplot(gs[1, 2:4])
composantes  = ['Efficacité\nénergétique', 'Symétrie\nG/D', 'Confort\narticulaire']
colors_corr  = ['steelblue', 'tomato', 'seagreen']
bars_f = ax_f.bar(composantes, correlations, color=colors_corr,
                   alpha=0.8, edgecolor='white', width=0.5)
ax_f.axhline(0, color='black', linewidth=0.8)
ax_f.set_title("F — Corrélation R(s,a) vs composantes biomécaniques",
               fontsize=12, fontweight='bold')
ax_f.set_ylabel("Coefficient de corrélation r")
ax_f.set_ylim(-0.3, 0.3)
ax_f.grid(alpha=0.3, axis='y')
for bar, val in zip(bars_f, correlations):
    y = val + 0.01 if val >= 0 else val - 0.02
    ax_f.text(bar.get_x() + bar.get_width()/2, y,
              f"r = {val:.3f}", ha='center', va='bottom',
              fontsize=11, fontweight='bold')

plt.savefig("resultats/poster/figure4_poster_principal.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 4 sauvegardée")


# ============================================================
# 7. RÉSUMÉ FINAL
# ============================================================

print(f"\n{'='*60}")
print("PHASE 3 TERMINÉE — Fichiers créés :")
print(f"  resultats/poster/figure1_biomecanique.png")
print(f"  resultats/poster/figure2_squelette.png")
print(f"  resultats/poster/figure3_interpretation.png")
print(f"  resultats/poster/figure4_poster_principal.png")
print(f"\nCorrelations R(s,a) vs composantes biomécaniques :")
print(f"  Efficacité énergétique : r = {correlations[0]:.3f}")
print(f"  Symétrie G/D           : r = {correlations[1]:.3f}")
print(f"  Confort articulaire    : r = {correlations[2]:.3f}")
print(f"\n✓ Prête pour la Phase 4 ou Phase 5 !")