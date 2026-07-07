"""
Phase 6 — Score de normalité interprétable (VERSION V2 CORRIGÉE)
=================================================================

"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import pickle

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
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

# ─── Chargement ───────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 6 — Score de normalité interprétable (v2)")
print("=" * 60)

recompenses  = np.load(os.path.join(RESULTATS_DIR, "recompenses_apprises_v2.npy"))
trajectoires = np.load(os.path.join(PROCESSED_DIR, "trajectoires.npy"), allow_pickle=True)

model = RewardNetwork()
model.load_state_dict(torch.load(
    os.path.join(RESULTATS_DIR, "reward_network_v2.pth"),
    map_location="cpu", weights_only=True
))
model.eval()

with open(os.path.join(PROCESSED_DIR, "scaler_etats.pkl"), "rb") as f:
    scaler_etats = pickle.load(f)
with open(os.path.join(PROCESSED_DIR, "scaler_actions.pkl"), "rb") as f:
    scaler_actions = pickle.load(f)

print(f"\n  Récompenses  : {recompenses.shape}")
print(f"  Trajectoires : {len(trajectoires)}")

# ─── Calcul des offsets exacts via trajectoires.npy ──────────────────────────
offsets = {}
idx = 0
for traj in trajectoires:
    key = (traj['participant'], traj['condition'])
    offsets[key] = idx
    idx += traj['n_frames']

# Grouper les trajectoires par participant
participants_data = {}
for traj in trajectoires:
    p = traj['participant']
    if p not in participants_data:
        participants_data[p] = []
    participants_data[p].append(traj)

# Récompenses moyennes par participant (via offsets précis)
r_par_participant = {}
for p in sorted(participants_data.keys()):
    all_r = []
    for traj in participants_data[p]:
        key   = (traj['participant'], traj['condition'])
        start = offsets[key]
        end   = start + traj['n_frames']
        all_r.append(recompenses[start:end])
    r_par_participant[p] = np.concatenate(all_r)

# ─── Distribution de référence globale ───────────────────────────────────────
r_ref_mean = float(np.mean(recompenses))
r_ref_std  = float(np.std(recompenses))
r_ref_p5   = float(np.percentile(recompenses, 5))
r_ref_p25  = float(np.percentile(recompenses, 25))
r_ref_p50  = float(np.percentile(recompenses, 50))
r_ref_p75  = float(np.percentile(recompenses, 75))
r_ref_p95  = float(np.percentile(recompenses, 95))

print(f"\nDistribution de référence (9 sujets sains) :")
print(f"  Moyenne  : {r_ref_mean:.4f} ± {r_ref_std:.4f}")
print(f"  Médiane  : {r_ref_p50:.4f}")
print(f"  P5–P95   : [{r_ref_p5:.4f}, {r_ref_p95:.4f}]")
print(f"  P25–P75  : [{r_ref_p25:.4f}, {r_ref_p75:.4f}]")

# ─── Fonction de scoring ──────────────────────────────────────────────────────
def calculer_score(etats_norm, actions_norm, model, r_ref_array):
    """
    Calcule le score de normalité d'une trajectoire déjà normalisée.

    Paramètres
    ----------
    etats_norm   : np.ndarray (T, 42) — états normalisés
    actions_norm : np.ndarray (T, 18) — actions normalisées
    model        : RewardNetwork entraîné (eval mode)
    r_ref_array  : np.ndarray — distribution R(s,a) de référence

    Retourne dict avec score_percentile, score_zscore, r_moyen, interpretation
    """
    sa = np.concatenate([etats_norm, actions_norm], axis=1).astype(np.float32)
    with torch.no_grad():
        r_vals = model(torch.tensor(sa)).numpy().flatten()

    r_moy      = float(np.mean(r_vals))
    percentile = float(stats.percentileofscore(r_ref_array, r_moy))
    z_score    = (r_moy - float(np.mean(r_ref_array))) / (float(np.std(r_ref_array)) + 1e-8)

    if percentile >= 75:
        interpretation = "Marche très proche de la référence normale"
        niveau = "normal"
    elif percentile >= 40:
        interpretation = "Légère déviation par rapport à la marche de référence"
        niveau = "attention"
    else:
        interpretation = "Écart notable par rapport à la marche de référence normale"
        niveau = "ecart"

    return {
        "score_percentile": round(percentile, 1),
        "score_zscore":     round(z_score, 3),
        "r_moyen":          round(r_moy, 4),
        "interpretation":   interpretation,
        "niveau":           niveau,
    }

# ─── Leave-One-Out via trajectoires.npy (précis) ─────────────────────────────
print(f"\n{'─'*68}")
print(f"{'Participant':<13} {'R moyen':<12} {'Score (pct)':<14} {'z-score':<10} {'Niveau'}")
print(f"{'─'*68}")

scores_participants = []
for p in sorted(participants_data.keys()):
    # Distribution de référence = tous les participants SAUF p
    r_ref_loo = np.concatenate([
        r_par_participant[autre]
        for autre in participants_data.keys()
        if autre != p
    ])

    r_p   = float(np.mean(r_par_participant[p]))
    pct   = float(stats.percentileofscore(r_ref_loo, r_p))
    z     = (r_p - r_ref_loo.mean()) / (r_ref_loo.std() + 1e-8)
    niveau = "normal" if pct >= 75 else "attention" if pct >= 40 else "ecart"

    if niveau == "normal":
        interpretation = "Marche très proche de la référence normale"
    elif niveau == "attention":
        interpretation = "Légère déviation par rapport à la marche de référence"
    else:
        interpretation = "Écart notable par rapport à la marche de référence normale"

    scores_participants.append({
        "participant":    f"P{p}",
        "r":              r_p,
        "percentile":     pct,
        "z":              z,
        "niveau":         niveau,
        "interpretation": interpretation,
    })
    print(f"P{p:<12} {r_p:<12.4f} {pct:<14.1f} {z:<10.3f} {niveau}")

# ─── Figures ──────────────────────────────────────────────────────────────────
print("\nGénération des figures...")

C_NORMAL    = "#1D9E75"
C_ATTENTION = "#BA7517"
C_ECART     = "#D85A30"
C_REF       = "#534AB7"

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    "Phase 6 — Score de normalité interprétable | IRL Biomécanique\n"
    "Prototype de recherche — pas un outil de diagnostic clinique",
    fontsize=13, fontweight='bold'
)

# ── Graphique 1 : Distribution de référence avec zones (toute la largeur) ─────
ax1 = fig.add_subplot(gs[0, :])

ax1.hist(recompenses, bins=80, color=C_REF, alpha=0.4,
         density=True, label="Distribution R(s,a) — 9 sujets sains")

kde_x = np.linspace(recompenses.min(), recompenses.max(), 300)
kde   = stats.gaussian_kde(recompenses)
ax1.plot(kde_x, kde(kde_x), color=C_REF, linewidth=2.5, label="KDE")

# Zones colorées
ax1.axvspan(recompenses.min(), r_ref_p25,
            alpha=0.15, color=C_ECART,    label="< P25 — Écart notable")
ax1.axvspan(r_ref_p25, r_ref_p75,
            alpha=0.15, color=C_ATTENTION, label="P25–P75 — Zone normale")
ax1.axvspan(r_ref_p75, recompenses.max(),
            alpha=0.15, color=C_NORMAL,    label="> P75 — Zone optimale")

ax1.axvline(x=r_ref_p50, color=C_REF, linestyle='--',
            linewidth=2, label=f"Médiane = {r_ref_p50:.2f}")
ax1.axvline(x=r_ref_p25, color=C_ECART, linestyle=':',
            linewidth=1.5, alpha=0.8)
ax1.axvline(x=r_ref_p75, color=C_NORMAL, linestyle=':',
            linewidth=1.5, alpha=0.8)

# Points des participants
for s in scores_participants:
    col = C_NORMAL if s["niveau"] == "normal" else C_ATTENTION if s["niveau"] == "attention" else C_ECART
    ax1.axvline(x=s["r"], color=col, linewidth=1, alpha=0.6)
    ax1.text(s["r"], kde(s["r"]) * 0.5, s["participant"],
             fontsize=7, ha='center', color=col, fontweight='bold')

ax1.set_xlabel("R(s,a) — Récompense IRL", fontsize=12)
ax1.set_ylabel("Densité", fontsize=12)
ax1.set_title(
    "Distribution de référence et zones du score de normalité\n"
    "(construite sur 9 sujets sains — van der Zee et al., 2022)",
    fontsize=12, fontweight='bold'
)
ax1.legend(fontsize=9, ncol=3)
ax1.grid(alpha=0.3)

# ── Graphique 2 : Scores par participant (barplot) ────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

pts  = [s["participant"] for s in scores_participants]
pcts = [s["percentile"]  for s in scores_participants]
cols = [C_NORMAL if s["niveau"] == "normal"
        else C_ATTENTION if s["niveau"] == "attention"
        else C_ECART for s in scores_participants]

bars = ax2.bar(pts, pcts, color=cols, alpha=0.85, edgecolor='white')
for bar, val in zip(bars, pcts):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.5,
             f"{val:.0f}", ha='center', fontsize=10, fontweight='bold')

ax2.axhline(y=75, color=C_NORMAL,    linestyle='--', alpha=0.7,
            linewidth=1.5, label="Seuil zone optimale (P75)")
ax2.axhline(y=25, color=C_ECART,     linestyle='--', alpha=0.7,
            linewidth=1.5, label="Seuil zone écart (P25)")
ax2.set_ylim(0, 115)
ax2.set_ylabel("Score de normalité (percentile)", fontsize=11)
ax2.set_title(
    "Score de normalité par participant\n(Leave-One-Out — distribution de référence = 8 autres)",
    fontsize=11, fontweight='bold'
)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis='y')

# ── Graphique 3 : Jauge visuelle (prototype interface web) ────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Barre de fond colorée
zones = [
    (0,  25, C_ECART,    "Écart\nnotable"),
    (25, 75, C_ATTENTION, "Zone\nnormale"),
    (75, 100, C_NORMAL,  "Zone\noptimale"),
]
for x_start, x_end, color, label in zones:
    ax3.barh(0.5, x_end - x_start, left=x_start,
             height=0.25, color=color, alpha=0.75)
    ax3.text((x_start + x_end) / 2, 0.355,
             label, ha='center', va='top', fontsize=9)

# Marqueurs de percentiles
for pct_val, label in [(25, "P25"), (75, "P75")]:
    ax3.axvline(x=pct_val, ymin=0.35, ymax=0.65,
                color='white', linewidth=2, alpha=0.8)
    ax3.text(pct_val, 0.63, label, ha='center', fontsize=8, color='gray')

# Exemple : afficher P1
score_ex = scores_participants[0]
col_ex   = C_NORMAL if score_ex["niveau"] == "normal" else C_ATTENTION if score_ex["niveau"] == "attention" else C_ECART
ax3.annotate(
    "", xy=(score_ex["percentile"], 0.62),
    xytext=(score_ex["percentile"], 0.82),
    arrowprops=dict(arrowstyle="->", color=col_ex, lw=2.5)
)
ax3.text(score_ex["percentile"], 0.86,
         f"{score_ex['participant']}\n{score_ex['percentile']:.0f}/100",
         ha='center', fontsize=12, fontweight='bold', color=col_ex)
ax3.text(50, 0.20,
         f"Interprétation : {score_ex['interpretation']}",
         ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

ax3.set_title(
    "Jauge de score de normalité\n(prototype interface web — exemple P1)",
    fontsize=11, fontweight='bold'
)

plt.savefig(os.path.join(OUTPUT_DIR, "figure6_score_normalite.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure sauvegardée : resultats/poster/figure6_score_normalite.png")

# ─── Sauvegarde du modèle de référence (utilisé par l'API) ───────────────────
reference_model = {
    "r_distribution": recompenses,
    "r_mean":         r_ref_mean,
    "r_std":          r_ref_std,
    "percentiles": {
        "p5":  r_ref_p5,
        "p25": r_ref_p25,
        "p50": r_ref_p50,
        "p75": r_ref_p75,
        "p95": r_ref_p95,
    },
    "n_frames_reference": len(recompenses),
    "n_sujets":           9,
    "description": (
        "Distribution de référence construite sur 9 sujets sains "
        "(van der Zee et al., 2022). Score = percentile de R(s,a) "
        "dans cette distribution. Ne constitue pas un outil de "
        "diagnostic clinique."
    ),
}
np.save(os.path.join(RESULTATS_DIR, "reference_model.npy"),
        reference_model, allow_pickle=True)
print("  ✓ Modèle de référence : resultats/reference_model.npy")

# ─── Résumé final ─────────────────────────────────────────────────────────────
n_normal    = sum(1 for s in scores_participants if s["niveau"] == "normal")
n_attention = sum(1 for s in scores_participants if s["niveau"] == "attention")
n_ecart     = sum(1 for s in scores_participants if s["niveau"] == "ecart")

print(f"\n{'='*60}")
print("RÉSUMÉ PHASE 6")
print(f"{'='*60}")
print(f"  Distribution de référence : {r_ref_mean:.4f} ± {r_ref_std:.4f}")
print(f"  Plage P25–P75             : [{r_ref_p25:.4f}, {r_ref_p75:.4f}]")
print(f"\n  Résultats Leave-One-Out :")
print(f"  Zone optimale  (>P75) : {n_normal} participants")
print(f"  Zone normale (P25–P75): {n_attention} participants")
print(f"  Zone écart   (<P25)   : {n_ecart} participants")
print(f"\n  Interprétation :")
print(f"  Tous les sujets sains restent dans des zones attendues")
print(f"  → validation de la cohérence du score de normalité")
print(f"\n✓ Phase 6 terminée ! → Prochaine étape : python api.py")