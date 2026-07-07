# ============================================================
# PHASE 2 — MaxEnt IRL (VERSION V2 — DONNÉES NETTOYÉES)
# Charge depuis data/processed_v2/ (filtre Butterworth appliqué)
# Exploite la structure des trajectoires pour un IRL plus rigoureux
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

os.makedirs("resultats", exist_ok=True)

# ============================================================
# 1. CHARGEMENT DES DONNÉES (depuis processed_v2)
# ============================================================

print("=" * 60)
print("PHASE 2 — MaxEnt IRL (données nettoyées v2)")
print("=" * 60)

print("\nChargement des données nettoyées...")

etats   = np.load("data/processed_v2/etats.npy")
actions = np.load("data/processed_v2/actions.npy")

# Trajectoires structurées (pour l'analyse par participant/condition)
trajectoires = np.load(
    "data/processed_v2/trajectoires.npy",
    allow_pickle=True
)

print(f"  états        : {etats.shape}")
print(f"  actions      : {actions.shape}")
print(f"  trajectoires : {len(trajectoires)} conditions")

# Vérification des données
assert not np.isnan(etats).any(),   "NaN détecté dans états !"
assert not np.isnan(actions).any(), "NaN détecté dans actions !"
print("  ✓ Aucun NaN détecté")

# Concaténation état+action → entrée du réseau (60 dims)
sa        = np.concatenate([etats, actions], axis=1).astype(np.float32)
sa_tensor = torch.FloatTensor(sa)

print(f"  Vecteur [s,a] : {sa.shape}  (42 états + 18 actions = 60 dims)")


# ============================================================
# 2. RÉSEAU DE NEURONES — FONCTION DE RÉCOMPENSE R(s,a)
# ============================================================

class RewardNetwork(nn.Module):
    """
    Réseau qui apprend R(s,a) — la fonction de récompense cachée.

    Architecture :
        Entrée  : [s(t), a(t)] = 60 dims
        Couches : 60 → 128 → 128 → 64 → 1
        Activations : Tanh (bornées, bon pour IRL)
        Sortie  : scalaire r(s,a)
    """
    def __init__(self, input_dim=60, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.1),           # régularisation légère
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# 3. ALGORITHME MaxEnt IRL
# ============================================================

class MaxEntIRL:
    """
    Implémentation de Maximum Entropy IRL (Ziebart et al., 2008).

    Principe :
        Loss = -E_expert[R(s,a)] + log(Z)
    où Z = partition function approximée par les trajectoires aléatoires.

    Les trajectoires aléatoires sont construites en mélangeant les actions
    (shuffle) sur les vrais états — ce qui préserve la distribution des états
    mais brise la cohérence état-action.
    """
    def __init__(self, input_dim=60, hidden_dim=128, lr=1e-3):
        self.reward_net = RewardNetwork(input_dim, hidden_dim)
        self.optimizer  = optim.Adam(
            self.reward_net.parameters(),
            lr=lr,
            weight_decay=1e-4    # L2 régularisation
        )
        self.scheduler  = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.5
        )
        self.losses            = []
        self.r_experts         = []
        self.r_randoms         = []

    def calculer_loss(self, sa_expert, sa_random):
        """
        Loss MaxEnt IRL :
            loss = -mean(R_expert) + logsumexp(R_random) - log(N)

        Maximiser R sur les trajectoires expertes,
        minimiser (via la partition) sur les trajectoires aléatoires.
        """
        r_expert = self.reward_net(sa_expert)
        r_random = self.reward_net(sa_random)

        loss_expert    = -r_expert.mean()
        loss_partition = (
            torch.logsumexp(r_random, dim=0)
            - torch.log(torch.tensor(float(len(r_random))))
        )

        loss = loss_expert + loss_partition
        return loss, r_expert.mean().item(), r_random.mean().item()

    def construire_trajectoire_aleatoire(self, batch):
        """
        Construit une trajectoire aléatoire en mélangeant les actions.
        Garde les états réels, mais associe des actions d'autres frames.
        """
        idx_random = torch.randperm(len(batch))
        sa_random  = torch.cat([
            batch[:, :42],          # états réels
            batch[idx_random, 42:]  # actions mélangées
        ], dim=1)
        return sa_random

    def entrainer(self, sa_tensor, n_epochs=150, batch_size=512):
        """
        Boucle d'entraînement principale.
        150 epochs (vs 100 en v1) pour meilleure convergence sur données lissées.
        """
        dataset    = TensorDataset(sa_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"\nEntraînement MaxEnt IRL...")
        print(f"  {len(sa_tensor)} paires (état, action)")
        print(f"  {n_epochs} epochs, batch_size={batch_size}")
        print(f"  LR initial = 1e-3, decay ×0.5 toutes les 30 epochs")
        print(f"\n  {'Epoch':>6} | {'Loss':>10} | {'R_expert':>10} | {'R_random':>10} | {'Ratio':>8}")
        print(f"  {'-'*55}")

        best_loss       = float('inf')
        best_state_dict = None

        for epoch in range(n_epochs):
            self.reward_net.train()
            epoch_loss  = 0
            epoch_r_exp = 0
            epoch_r_ran = 0
            n_batches   = 0

            for (batch,) in dataloader:
                sa_expert = batch
                sa_random = self.construire_trajectoire_aleatoire(batch)

                self.optimizer.zero_grad()
                loss, r_exp, r_rand = self.calculer_loss(sa_expert, sa_random)
                loss.backward()

                # Gradient clipping — évite les explosions de gradients
                torch.nn.utils.clip_grad_norm_(
                    self.reward_net.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                epoch_loss  += loss.item()
                epoch_r_exp += r_exp
                epoch_r_ran += r_rand
                n_batches   += 1

            avg_loss  = epoch_loss  / n_batches
            avg_r_exp = epoch_r_exp / n_batches
            avg_r_ran = epoch_r_ran / n_batches
            ratio     = avg_r_exp / max(abs(avg_r_ran), 1e-8)

            self.losses.append(avg_loss)
            self.r_experts.append(avg_r_exp)
            self.r_randoms.append(avg_r_ran)

            # Sauvegarde du meilleur modèle
            if avg_loss < best_loss:
                best_loss       = avg_loss
                best_state_dict = {k: v.clone() for k, v in
                                   self.reward_net.state_dict().items()}

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                lr_actuel = self.optimizer.param_groups[0]['lr']
                print(f"  {epoch+1:>6} | {avg_loss:>10.4f} | "
                      f"{avg_r_exp:>10.4f} | {avg_r_ran:>10.4f} | "
                      f"{ratio:>7.2f}x  [lr={lr_actuel:.2e}]")

        # Recharger le meilleur modèle
        if best_state_dict is not None:
            self.reward_net.load_state_dict(best_state_dict)
            print(f"\n  ✓ Meilleur modèle restauré (loss = {best_loss:.4f})")

        print(f"\n  Entraînement terminé !")
        return self.losses


# ============================================================
# 4. ENTRAÎNEMENT
# ============================================================

irl    = MaxEntIRL(input_dim=60, hidden_dim=128, lr=1e-3)
losses = irl.entrainer(sa_tensor, n_epochs=150, batch_size=512)


# ============================================================
# 5. CALCUL DES RÉCOMPENSES APPRISES
# ============================================================

print("\nCalcul des récompenses apprises...")
irl.reward_net.eval()

with torch.no_grad():
    recompenses_apprises = irl.reward_net(sa_tensor).numpy()

print(f"  Récompense moyenne : {recompenses_apprises.mean():.4f}")
print(f"  Récompense std     : {recompenses_apprises.std():.4f}")
print(f"  Récompense min     : {recompenses_apprises.min():.4f}")
print(f"  Récompense max     : {recompenses_apprises.max():.4f}")
print(f"  Ratio final R_exp/R_rand : {irl.r_experts[-1]/max(abs(irl.r_randoms[-1]),1e-8):.2f}x")


# ============================================================
# 6. ANALYSE PAR PARTICIPANT — via trajectoires structurées
# ============================================================

print("\nAnalyse par participant (via trajectoires.npy)...")

# Grouper les trajectoires par participant
participants_data = {}
for traj in trajectoires:
    p = traj['participant']
    if p not in participants_data:
        participants_data[p] = []
    participants_data[p].append(traj)

# Calculer l'offset de chaque trajectoire dans le tableau global
offsets = {}
idx     = 0
for traj in trajectoires:
    key = (traj['participant'], traj['condition'])
    offsets[key] = idx
    idx += traj['n_frames']

# Récompenses moyennes par participant
recomp_par_participant = []
noms_participants      = []

for p in sorted(participants_data.keys()):
    trajs      = participants_data[p]
    all_r      = []
    for traj in trajs:
        key   = (traj['participant'], traj['condition'])
        start = offsets[key]
        end   = start + traj['n_frames']
        all_r.append(recompenses_apprises[start:end])

    r_concat = np.concatenate(all_r)
    r_moy    = r_concat.mean()
    recomp_par_participant.append(r_moy)
    noms_participants.append(f"P{p}")
    print(f"  Participant {p} — {len(trajs)} conditions — R_moy = {r_moy:.4f}")


# ============================================================
# 7. VISUALISATIONS — 4 GRAPHIQUES POUR LE POSTER
# ============================================================

print("\nGénération des visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "MaxEnt IRL — Fonction de Récompense Apprise (v2)\n"
    "Biomécanique de la Marche Humaine — Données filtrées Butterworth 6Hz",
    fontsize=13, fontweight='bold'
)

# ---- Graphique 1 : Convergence de la loss ----
ax = axes[0, 0]
ax.plot(losses, color='steelblue', linewidth=2, label='Loss')
ax.plot(irl.r_experts, color='forestgreen', linewidth=1.5,
        linestyle='--', alpha=0.8, label='R_expert')
ax.plot(irl.r_randoms, color='tomato', linewidth=1.5,
        linestyle='--', alpha=0.8, label='R_random')
ax.set_title("Convergence de l'entraînement", fontsize=12)
ax.set_xlabel("Epoch")
ax.set_ylabel("Valeur")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- Graphique 2 : Distribution des récompenses ----
ax = axes[0, 1]
ax.hist(recompenses_apprises, bins=80, color='steelblue',
        alpha=0.8, edgecolor='white')
ax.axvline(recompenses_apprises.mean(), color='tomato', linewidth=2,
           linestyle='--', label=f"μ = {recompenses_apprises.mean():.2f}")
ax.set_title("Distribution de R(s,a) apprise", fontsize=12)
ax.set_xlabel("Récompense R(s,a)")
ax.set_ylabel("Fréquence")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- Graphique 3 : Évolution temporelle R(s,a) — 3 conditions ----
ax = axes[1, 0]
couleurs = ['steelblue', 'forestgreen', 'tomato']
for i, traj in enumerate(trajectoires[:3]):
    key   = (traj['participant'], traj['condition'])
    start = offsets[key]
    end   = start + traj['n_frames']
    r_traj = recompenses_apprises[start:end]
    ax.plot(r_traj, color=couleurs[i], linewidth=1.2, alpha=0.85,
            label=f"P{traj['participant']} Cond{traj['condition']}")
ax.set_title("Évolution de R(s,a) — 3 premières conditions", fontsize=12)
ax.set_xlabel("Frame (pas de temps)")
ax.set_ylabel("R(s,a)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- Graphique 4 : Récompense moyenne par participant ----
ax = axes[1, 1]
bars = ax.bar(noms_participants, recomp_par_participant,
              color='steelblue', alpha=0.8, edgecolor='white')
ax.set_title("Récompense moyenne par participant", fontsize=12)
ax.set_xlabel("Participant")
ax.set_ylabel("R(s,a) moyen")
ax.grid(alpha=0.3, axis='y')

# Valeurs au-dessus des barres
y_offset = (max(recomp_par_participant) - min(recomp_par_participant)) * 0.02
for bar, v in zip(bars, recomp_par_participant):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{v:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("resultats/IRL_resultats_v2.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Graphique sauvegardé : resultats/IRL_resultats_v2.png")


# ============================================================
# 8. RÉSUMÉ FINAL
# ============================================================

r_exp_final  = irl.r_experts[-1]
r_rand_final = irl.r_randoms[-1]
ratio_final  = r_exp_final / max(abs(r_rand_final), 1e-8)

print(f"\n{'='*60}")
print("RÉSUMÉ PHASE 2 — MaxEnt IRL v2")
print(f"{'='*60}")
print(f"  Données          : processed_v2 (filtrées Butterworth 6Hz)")
print(f"  Frames totales   : {len(sa_tensor)}")
print(f"  Trajectoires     : {len(trajectoires)}")
print(f"  Epochs           : 150")
print(f"  Loss finale      : {losses[-1]:.4f}")
print(f"  R_expert final   : {r_exp_final:.4f}")
print(f"  R_random final   : {r_rand_final:.4f}")
print(f"  Ratio R_exp/R_rand : ×{ratio_final:.2f}")
print(f"  R(s,a) moyenne   : {recompenses_apprises.mean():.4f} ± {recompenses_apprises.std():.4f}")
print(f"{'='*60}")

if ratio_final >= 3.0:
    print("  ✓ EXCELLENT : le modèle distingue bien expert vs aléatoire")
elif ratio_final >= 2.0:
    print("  ✓ BON : le modèle distingue correctement expert vs aléatoire")
else:
    print("  ⚠ FAIBLE : envisager plus d'epochs ou ajuster le lr")


# ============================================================
# 9. SAUVEGARDE
# ============================================================

torch.save(irl.reward_net.state_dict(), "resultats/reward_network_v2.pth")
np.save("resultats/recompenses_apprises_v2.npy", recompenses_apprises)

print(f"\n  Modèle sauvegardé    : resultats/reward_network_v2.pth")
print(f"  Récompenses sauveg.  : resultats/recompenses_apprises_v2.npy")
print(f"\n✓ Phase 2 v2 terminée ! Prête pour la Phase 3.")