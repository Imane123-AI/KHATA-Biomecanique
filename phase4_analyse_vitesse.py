"""
Phase 4 — Analyse de R(s,a) par vitesse de marche (VERSION V2 CORRIGÉE)
=========================================================================
Objectif : montrer que R(s,a) est maximale à la vitesse optimale.
Résultat attendu : courbe en U inversé → validation biomécanique du modèle IRL.

Correction v2 :
  - Utilise trajectoires.npy pour un mapping précis (pas d'hypothèse sur les tailles)
  - Charge depuis processed_v2/ et recompenses_apprises_v2.npy
  - Offsets calculés exactement comme en Phase 3

Exécution :
    python phase4_analyse_vitesse.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─── Chemins ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed_v2"
RESULTATS_DIR = "resultats"
OUTPUT_DIR    = os.path.join(RESULTATS_DIR, "poster")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Chargement ───────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 4 — Analyse R(s,a) par vitesse de marche (v2)")
print("=" * 60)

recompenses  = np.load(os.path.join(RESULTATS_DIR, "recompenses_apprises_v2.npy"))
trajectoires = np.load(os.path.join(PROCESSED_DIR, "trajectoires.npy"), allow_pickle=True)

print(f"\n  Récompenses  : {recompenses.shape}")
print(f"  Trajectoires : {len(trajectoires)} conditions")

# ─── Calcul des offsets exacts (même logique que Phase 3) ────────────────────
offsets = {}
idx = 0
for traj in trajectoires:
    key = (traj['participant'], traj['condition'])
    offsets[key] = idx
    idx += traj['n_frames']

# ─── Mapping conditions → vitesses ────────────────────────────────────────────
# D'après van der Zee et al. (2022) :
# Conditions 0–12  → variations de VITESSE (0.6 → 1.8 m/s par pas de 0.1)
# Conditions 13–24 → variations de longueur de pas
# Conditions 25–32 → variations de largeur de pas (25-29 vides)

VITESSES_CONDITIONS = {
    0:  0.60,
    1:  0.70,
    2:  0.80,
    3:  0.90,
    4:  1.00,
    5:  1.10,
    6:  1.20,   # vitesse préférée approximative
    7:  1.30,
    8:  1.40,
    9:  1.50,
    10: 1.60,
    11: 1.70,
    12: 1.80,
}

# ─── Calcul de R(s,a) moyenne par vitesse — via offsets précis ───────────────
print("\nCalcul de R(s,a) par condition de vitesse...")

r_par_vitesse = {}   # vitesse (m/s) → liste de R moyennes (une par participant)

for traj in trajectoires:
    cond_idx = traj['condition']

    # On ne garde que les conditions de vitesse (0–12)
    if cond_idx not in VITESSES_CONDITIONS:
        continue

    vitesse = VITESSES_CONDITIONS[cond_idx]
    p       = traj['participant']

    # Récupérer les récompenses exactes de cette trajectoire
    key   = (p, cond_idx)
    start = offsets[key]
    end   = start + traj['n_frames']
    r_moy = float(np.mean(recompenses[start:end]))

    if vitesse not in r_par_vitesse:
        r_par_vitesse[vitesse] = []
    r_par_vitesse[vitesse].append(r_moy)

print(f"  {len(r_par_vitesse)} vitesses analysées")
for v in sorted(r_par_vitesse.keys()):
    n = len(r_par_vitesse[v])
    print(f"  {v:.2f} m/s → {n} participants")

# ─── Statistiques par vitesse ─────────────────────────────────────────────────
vitesses_sorted = sorted(r_par_vitesse.keys())
r_moyennes      = np.array([np.mean(r_par_vitesse[v]) for v in vitesses_sorted])
r_stds          = np.array([np.std(r_par_vitesse[v])  for v in vitesses_sorted])
r_sems          = r_stds / np.sqrt([len(r_par_vitesse[v]) for v in vitesses_sorted])
vitesses_arr    = np.array(vitesses_sorted)

# Vitesse optimale = celle qui maximise R(s,a)
idx_optimal  = int(np.argmax(r_moyennes))
vitesse_opt  = vitesses_arr[idx_optimal]
r_opt        = r_moyennes[idx_optimal]

print(f"\n{'─'*55}")
print(f"{'Vitesse (m/s)':<18} {'R(s,a) moy':<14} {'±std':<10} {'N participants'}")
print(f"{'─'*55}")
for v, r, s in zip(vitesses_arr, r_moyennes, r_stds):
    n    = len(r_par_vitesse[v])
    flag = " ← OPTIMAL" if v == vitesse_opt else ""
    print(f"  {v:<16.2f} {r:<14.4f} {s:<10.4f} {n}{flag}")
print(f"{'─'*55}")

# ─── Ajustement polynomial degré 2 (courbe en U inversé) ─────────────────────
poly_coeffs = np.polyfit(vitesses_arr, r_moyennes, deg=2)
poly_fn     = np.poly1d(poly_coeffs)
x_smooth    = np.linspace(vitesses_arr.min(), vitesses_arr.max(), 300)
y_smooth    = poly_fn(x_smooth)

# Vitesse optimale théorique selon le polynôme : sommet de la parabole
# Pour ax² + bx + c, le sommet est à x = -b / (2a)
a, b, c      = poly_coeffs
x_poly_opt   = -b / (2 * a)
y_poly_opt   = poly_fn(x_poly_opt)

# R² de l'ajustement
ss_res = np.sum((r_moyennes - poly_fn(vitesses_arr)) ** 2)
ss_tot = np.sum((r_moyennes - np.mean(r_moyennes)) ** 2)
r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

# Interprétation de la forme de la courbe
forme = "U inversé ✓" if a < 0 else "U (inattendu) ✗"

print(f"\n  Vitesse optimale (données)   : {vitesse_opt:.2f} m/s")
print(f"  Vitesse optimale (polynôme)  : {x_poly_opt:.2f} m/s")
print(f"  R² ajustement polynomial     : {r2:.4f}")
print(f"  Forme de la courbe           : {forme}")

# ─── Score de normalité normalisé 0–100 ───────────────────────────────────────
r_min  = r_moyennes.min()
r_max  = r_moyennes.max()
r_norm = 100 * (r_moyennes - r_min) / (r_max - r_min + 1e-8)

# ─── Figures ──────────────────────────────────────────────────────────────────
print("\nGénération des figures...")

COULEUR_COURBE = "#1D9E75"
COULEUR_OPT    = "#D85A30"
COULEUR_POLY   = "#534AB7"

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    "Phase 4 — Analyse de R(s,a) par vitesse de marche\n"
    "IRL Biomécanique — données filtrées Butterworth 6Hz",
    fontsize=14, fontweight='bold', y=1.01
)

# ── Graphique 1 : Courbe principale R(s,a) vs vitesse (toute la largeur) ──────
ax1 = fig.add_subplot(gs[0, :])

ax1.fill_between(
    vitesses_arr,
    r_moyennes - r_sems,
    r_moyennes + r_sems,
    alpha=0.2, color=COULEUR_COURBE, label="±SEM (erreur standard)"
)
ax1.errorbar(
    vitesses_arr, r_moyennes, yerr=r_stds,
    fmt='o', color=COULEUR_COURBE, markersize=9,
    capsize=5, linewidth=2.5,
    label="R(s,a) moyenne ± std (9 participants)"
)
ax1.plot(
    x_smooth, y_smooth, '--', color=COULEUR_POLY, linewidth=2.5, alpha=0.85,
    label=f"Ajustement polynomial deg.2  (R²={r2:.3f})"
)
ax1.axvline(
    x=vitesse_opt, color=COULEUR_OPT, linestyle=':', linewidth=2.5,
    label=f"Vitesse optimale (données) = {vitesse_opt:.2f} m/s"
)
ax1.axvline(
    x=x_poly_opt, color=COULEUR_POLY, linestyle=':', linewidth=1.5, alpha=0.6,
    label=f"Vitesse optimale (polynôme) = {x_poly_opt:.2f} m/s"
)
ax1.scatter([vitesse_opt], [r_opt], color=COULEUR_OPT, s=180, zorder=6)
ax1.scatter([x_poly_opt], [y_poly_opt], color=COULEUR_POLY,
            s=120, zorder=5, marker='D')

ax1.set_xlabel("Vitesse de marche (m/s)", fontsize=13)
ax1.set_ylabel("R(s,a) — Récompense IRL moyenne", fontsize=13)
ax1.set_title(
    "R(s,a) apprise en fonction de la vitesse de marche\n"
    "Forme en U inversé = la marche est optimale à la vitesse spontanée préférée",
    fontsize=12, fontweight='bold'
)
ax1.legend(fontsize=10, loc='lower center', ncol=2)
ax1.grid(alpha=0.3)

# ── Graphique 2 : Boxplot par vitesse ────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

data_box = [r_par_vitesse[v] for v in vitesses_sorted]
bp = ax2.boxplot(
    data_box, patch_artist=True,
    medianprops=dict(color=COULEUR_OPT, linewidth=2.5)
)
for patch in bp['boxes']:
    patch.set_facecolor(COULEUR_COURBE)
    patch.set_alpha(0.5)

ax2.set_xticklabels([f"{v:.1f}" for v in vitesses_sorted], rotation=45, fontsize=9)
ax2.set_xlabel("Vitesse (m/s)", fontsize=12)
ax2.set_ylabel("R(s,a)", fontsize=12)
ax2.set_title(
    "Distribution de R(s,a) par vitesse\n(variabilité inter-participants)",
    fontsize=12
)
ax2.grid(alpha=0.3, axis='y')

# ── Graphique 3 : Score normalisé 0–100 ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

couleurs_bar = [COULEUR_OPT if v == vitesse_opt else COULEUR_COURBE
                for v in vitesses_arr]
bars = ax3.bar(
    vitesses_arr, r_norm, width=0.08,
    color=couleurs_bar, alpha=0.80, edgecolor='white'
)
ax3.set_xlabel("Vitesse (m/s)", fontsize=12)
ax3.set_ylabel("Score de normalité (0–100)", fontsize=12)
ax3.set_title(
    "Score de normalité par vitesse\n"
    "(base du score de déviation gait)",
    fontsize=12
)
ax3.axhline(y=100, color=COULEUR_OPT, linestyle='--', alpha=0.4, linewidth=1.5)
ax3.grid(alpha=0.3, axis='y')

# Valeurs sur les barres
for bar, score in zip(bars, r_norm):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{score:.0f}",
        ha='center', va='bottom', fontsize=8
    )

plt.savefig(os.path.join(OUTPUT_DIR, "figure4_vitesse.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure sauvegardée : resultats/poster/figure4_vitesse.png")

# ─── Sauvegarde des résultats ─────────────────────────────────────────────────
resultats_phase4 = {
    "vitesses":          vitesses_arr,
    "r_moyennes":        r_moyennes,
    "r_stds":            r_stds,
    "r_sems":            r_sems,
    "r_norm":            r_norm,
    "vitesse_optimale":  vitesse_opt,
    "r_optimal":         r_opt,
    "r2_polynomial":     r2,
    "poly_coefficients": poly_coeffs,
    "x_poly_opt":        x_poly_opt,
}
np.save(os.path.join(RESULTATS_DIR, "phase4_resultats.npy"),
        resultats_phase4, allow_pickle=True)
print("  ✓ Résultats sauvegardés : resultats/phase4_resultats.npy")

# ─── Résumé final ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RÉSUMÉ PHASE 4")
print(f"{'='*60}")
print(f"  Vitesse optimale (données)   : {vitesse_opt:.2f} m/s")
print(f"  Vitesse optimale (polynôme)  : {x_poly_opt:.2f} m/s")
print(f"  R(s,a) maximale              : {r_opt:.4f}")
print(f"  R² ajustement polynomial     : {r2:.4f}")
print(f"  Forme de la courbe           : {forme}")
print(f"\n  Interprétation biomécanique :")
if a < 0:
    print(f"  ✓ Courbe en U inversé confirmée")
    print(f"  ✓ R(s,a) maximale à {vitesse_opt:.2f} m/s = vitesse spontanée préférée")
    print(f"  ✓ Le modèle IRL a bien capturé l'optimisation énergétique humaine")
else:
    print(f"  ⚠ Courbe non concave — vérifier le mapping conditions→vitesses")
print(f"\n✓ Phase 4 terminée ! → Prochaine étape : python phase5_importance_features.py")