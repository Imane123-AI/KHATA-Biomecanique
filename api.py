import os
# ============================================================
# API FLASK — IRL Biomécanique
# Backend complet pour le dashboard web
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import tempfile
import base64
from scipy import stats

app = Flask(__name__)
CORS(app)

# ============================================================
# 1. ARCHITECTURE DU RÉSEAU (identique Phase 2 v2)
# ============================================================

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
        return self.net(x).squeeze(-1)

# ============================================================
# 2. CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ============================================================

print("Chargement des modèles...")

# Réseau IRL
model = RewardNetwork()
model.load_state_dict(torch.load(
    "resultats/reward_network_v2.pth",
    map_location="cpu", weights_only=True
))
model.eval()

# Scalers
with open("data/processed_v2/scaler_etats.pkl", "rb") as f:
    scaler_etats = pickle.load(f)
with open("data/processed_v2/scaler_actions.pkl", "rb") as f:
    scaler_actions = pickle.load(f)

# Modèle de référence (Phase 6)
ref = np.load("resultats/reference_model.npy", allow_pickle=True).item()
r_distribution = ref["r_distribution"]
r_mean         = ref["r_mean"]
r_std          = ref["r_std"]
percentiles    = ref["percentiles"]

# Trajectoires et récompenses
trajectoires = np.load(
    "data/processed_v2/trajectoires.npy", allow_pickle=True
)
recompenses  = np.load("resultats/recompenses_apprises_v2.npy")

# Importance des features (Phase 5)
phase5 = np.load(
    "resultats/phase5_importances.npy", allow_pickle=True
).item()
cat_articulaire = phase5["cat_articulaire"]
imp_groupes_pct = phase5["imp_groupes_pct"]

# Résultats Phase 4 (vitesse)
phase4 = np.load(
    "resultats/phase4_resultats.npy", allow_pickle=True
).item()

# Offsets par trajectoire
offsets = {}
idx = 0
for traj in trajectoires:
    key = (traj['participant'], traj['condition'])
    offsets[key] = idx
    idx += traj['n_frames']

# Participants groupés
participants_data = {}
for traj in trajectoires:
    p = traj['participant']
    if p not in participants_data:
        participants_data[p] = []
    participants_data[p].append(traj)

print("✓ Modèles chargés avec succès")

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def calculer_score_normalite(r_moy):
    """Convertit R(s,a) en score percentile 0-100."""
    percentile = float(stats.percentileofscore(r_distribution, r_moy))
    z_score    = (r_moy - r_mean) / (r_std + 1e-8)

    if percentile >= 75:
        niveau         = "normal"
        interpretation = "Marche très proche de la référence normale"
        couleur        = "#1D9E75"
    elif percentile >= 40:
        niveau         = "attention"
        interpretation = "Légère déviation par rapport à la marche de référence"
        couleur        = "#BA7517"
    else:
        niveau         = "ecart"
        interpretation = "Écart notable par rapport à la marche de référence normale"
        couleur        = "#D85A30"

    return {
        "score":           round(percentile, 1),
        "z_score":         round(z_score, 3),
        "r_moyen":         round(r_moy, 4),
        "niveau":          niveau,
        "interpretation":  interpretation,
        "couleur":         couleur,
    }

def generer_rapport(score_info, participant_id=None):
    """Génère un rapport textuel automatique."""
    score  = score_info["score"]
    niveau = score_info["niveau"]

    intro = (
        f"Analyse biomécanique réalisée par le modèle MaxEnt IRL "
        f"(van der Zee et al., 2022). "
        f"Score de normalité calculé par rapport à une distribution "
        f"de référence construite sur 9 sujets sains."
    )

    if niveau == "normal":
        conclusion = (
            f"Le score obtenu ({score:.0f}/100) indique une marche "
            f"très proche de la référence normale. "
            f"Les paramètres biomécaniques analysés sont cohérents "
            f"avec ceux des sujets sains de la base de données."
        )
    elif niveau == "attention":
        conclusion = (
            f"Le score obtenu ({score:.0f}/100) indique une légère "
            f"déviation par rapport à la marche de référence. "
            f"Une analyse complémentaire pourrait être envisagée."
        )
    else:
        conclusion = (
            f"Le score obtenu ({score:.0f}/100) indique un écart "
            f"notable par rapport à la marche de référence normale. "
            f"Les paramètres biomécaniques présentent des différences "
            f"significatives par rapport aux sujets sains de référence."
        )

    articulations = (
        f"Contribution des articulations à la récompense apprise : "
        f"Hanche {cat_articulaire['Hanche']:.1f}%, "
        f"Genou {cat_articulaire['Genou']:.1f}%, "
        f"Cheville {cat_articulaire['Cheville']:.1f}%, "
        f"GRF {cat_articulaire['GRF']:.1f}%."
    )

    avertissement = (
        "⚠️ Ce rapport est généré par un prototype de recherche. "
        "Il ne constitue pas un outil de diagnostic clinique validé."
    )

    return {
        "introduction":   intro,
        "conclusion":     conclusion,
        "articulations":  articulations,
        "avertissement":  avertissement,
        "score_detail": {
            "percentile":     score,
            "z_score":        score_info["z_score"],
            "r_moyen":        score_info["r_moyen"],
            "interpretation": score_info["interpretation"],
        }
    }

# ============================================================
# 4. ENDPOINTS API
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    """Vérification que l'API fonctionne."""
    return jsonify({
        "status":        "ok",
        "modele":        "MaxEnt IRL v2",
        "n_frames_ref":  int(len(r_distribution)),
        "n_participants": len(participants_data),
    })

# ── GET /api/participants ─────────────────────────────────────
@app.route("/api/participants", methods=["GET"])
def get_participants():
    """Retourne les scores de tous les participants."""
    result = []
    for p in sorted(participants_data.keys()):
        all_r = []
        for traj in participants_data[p]:
            key   = (traj['participant'], traj['condition'])
            start = offsets[key]
            end   = start + traj['n_frames']
            all_r.append(recompenses[start:end])

        r_concat = np.concatenate(all_r)
        r_moy    = float(r_concat.mean())
        score    = calculer_score_normalite(r_moy)

        result.append({
            "id":           p,
            "nom":          f"Participant {p}",
            "n_conditions": len(participants_data[p]),
            "r_moyen":      round(r_moy, 4),
            "score":        score["score"],
            "niveau":       score["niveau"],
            "couleur":      score["couleur"],
        })

    return jsonify(result)

# ── GET /api/participant/<id> ─────────────────────────────────
@app.route("/api/participant/<int:participant_id>", methods=["GET"])
def get_participant(participant_id):
    """Retourne les données détaillées d'un participant."""
    if participant_id not in participants_data:
        return jsonify({"erreur": f"Participant {participant_id} non trouvé"}), 404

    trajs = participants_data[participant_id]

    # Récompenses par condition
    conditions = []
    for traj in trajs:
        key   = (traj['participant'], traj['condition'])
        start = offsets[key]
        end   = start + traj['n_frames']
        r_traj = recompenses[start:end]

        conditions.append({
            "condition": int(traj['condition']),
            "n_frames":  int(traj['n_frames']),
            "r_moyen":   round(float(r_traj.mean()), 4),
            "r_std":     round(float(r_traj.std()), 4),
            "r_serie":   r_traj[::10].tolist(),  # sous-échantillonné ×10
        })

    # Récompenses globales du participant
    all_r = np.concatenate([recompenses[offsets[(participant_id, t['condition'])]:
                             offsets[(participant_id, t['condition'])] + t['n_frames']]
                            for t in trajs])
    r_moy    = float(all_r.mean())
    score    = calculer_score_normalite(r_moy)
    rapport  = generer_rapport(score, participant_id)

    return jsonify({
        "participant": participant_id,
        "r_moyen":     round(r_moy, 4),
        "score":       score,
        "rapport":     rapport,
        "conditions":  conditions,
        "n_conditions": len(trajs),
    })

# ── GET /api/reference ────────────────────────────────────────
@app.route("/api/reference", methods=["GET"])
def get_reference():
    """Retourne la distribution de référence pour les graphiques."""
    # Histogramme (80 bins)
    counts, bin_edges = np.histogram(r_distribution, bins=80, density=True)

    # Scores de tous les participants pour comparaison
    scores_par_participant = []
    for p in sorted(participants_data.keys()):
        all_r = []
        for traj in participants_data[p]:
            key   = (traj['participant'], traj['condition'])
            start = offsets[key]
            end   = start + traj['n_frames']
            all_r.append(recompenses[start:end])
        r_moy = float(np.concatenate(all_r).mean())
        scores_par_participant.append({
            "participant": p,
            "r_moyen":     round(r_moy, 4),
        })

    return jsonify({
        "distribution": {
            "counts":    counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
        "statistiques": {
            "mean": round(r_mean, 4),
            "std":  round(r_std,  4),
            "p5":   round(percentiles["p5"],  4),
            "p25":  round(percentiles["p25"], 4),
            "p50":  round(percentiles["p50"], 4),
            "p75":  round(percentiles["p75"], 4),
            "p95":  round(percentiles["p95"], 4),
        },
        "scores_participants": scores_par_participant,
        "vitesse": {
            "vitesses":         phase4["vitesses"].tolist(),
            "r_moyennes":       phase4["r_moyennes"].tolist(),
            "r_stds":           phase4["r_stds"].tolist(),
            "vitesse_optimale": float(phase4["vitesse_optimale"]),
            "r2_polynomial":    float(phase4["r2_polynomial"]),
        }
    })

# ── GET /api/importance ───────────────────────────────────────
@app.route("/api/importance", methods=["GET"])
def get_importance():
    """Retourne l'importance des articulations (Phase 5)."""
    return jsonify({
        "par_articulation": {
            k: round(v, 2) for k, v in cat_articulaire.items()
        },
        "par_groupe": {
            k: round(v, 2) for k, v in imp_groupes_pct.items()
        },
        "articulation_dominante": max(
            cat_articulaire, key=cat_articulaire.get
        ),
    })

# ── POST /api/score ───────────────────────────────────────────
@app.route("/api/score", methods=["POST"])
def calculer_score():
    """
    Calcule le score de normalité depuis des données biomécaniques.

    Body JSON attendu :
    {
        "etats":   [[...], [...], ...],   // (T, 42) non normalisés
        "actions": [[...], [...], ...]    // (T, 18) non normalisés
    }
    """
    data = request.get_json()
    if not data or "etats" not in data or "actions" not in data:
        return jsonify({"erreur": "Champs 'etats' et 'actions' requis"}), 400

    try:
        etats_raw   = np.array(data["etats"],   dtype=np.float32)
        actions_raw = np.array(data["actions"], dtype=np.float32)

        if etats_raw.shape[1] != 42:
            return jsonify({"erreur": f"États : 42 dimensions attendues, {etats_raw.shape[1]} reçues"}), 400
        if actions_raw.shape[1] != 18:
            return jsonify({"erreur": f"Actions : 18 dimensions attendues, {actions_raw.shape[1]} reçues"}), 400

        # Normalisation
        etats_norm   = scaler_etats.transform(etats_raw)
        actions_norm = scaler_actions.transform(actions_raw)
        sa = np.concatenate([etats_norm, actions_norm], axis=1).astype(np.float32)

        # Inférence
        with torch.no_grad():
            r_vals = model(torch.tensor(sa)).numpy().flatten()

        r_moy   = float(r_vals.mean())
        score   = calculer_score_normalite(r_moy)
        rapport = generer_rapport(score)

        return jsonify({
            "r_serie":  r_vals[::max(1, len(r_vals)//200)].tolist(),
            "score":    score,
            "rapport":  rapport,
            "n_frames": len(r_vals),
        })

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500

# ── POST /api/analyze-video ───────────────────────────────────
@app.route("/api/analyze-video", methods=["POST"])
def analyze_video():
    """
    Pipeline vidéo complet :
    1. Reçoit une vidéo
    2. Extrait les angles avec MediaPipe
    3. Calcule les vitesses par différentiation
    4. Calcule R(s,a) et le score de normalité
    5. Retourne score + rapport + données pour graphiques
    """
    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        return jsonify({
            "erreur": "MediaPipe ou OpenCV non installé. Lancez : pip install mediapipe opencv-python"
        }), 500

    if "video" not in request.files:
        return jsonify({"erreur": "Fichier vidéo requis (champ 'video')"}), 400

    video_file = request.files["video"]

    # Sauvegarder temporairement
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── Extraction MediaPipe ──────────────────────────────
        import urllib.request
        model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                model_path
            )

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        landmarks_series = []

        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frame_idx += 1
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    landmarks_series.append({
                        "left_hip":    (lm[23].x, lm[23].y),
                        "right_hip":   (lm[24].x, lm[24].y),
                        "left_knee":   (lm[25].x, lm[25].y),
                        "right_knee":  (lm[26].x, lm[26].y),
                        "left_ankle":  (lm[27].x, lm[27].y),
                        "right_ankle": (lm[28].x, lm[28].y),
                    })

        cap.release()

        if len(landmarks_series) < 10:
            return jsonify({"erreur": "Vidéo trop courte ou pose non détectée"}), 400

        # ── Calcul des angles articulaires ────────────────────
        def angle_2d(a, b, c):
            """Angle en b formé par les points a-b-c (en degrés)."""
            a, b, c = np.array(a), np.array(b), np.array(c)
            v1 = a - b
            v2 = c - b
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

        T = len(landmarks_series)
        angles_hanche_G  = np.zeros(T)
        angles_genou_G   = np.zeros(T)
        angles_cheville_G = np.zeros(T)
        angles_hanche_D  = np.zeros(T)
        angles_genou_D   = np.zeros(T)
        angles_cheville_D = np.zeros(T)

        for i, lm in enumerate(landmarks_series):
            # Hanche gauche : épaule-hanche-genou (approximation)
            angles_hanche_G[i]   = angle_2d(
                (lm["left_hip"][0],   lm["left_hip"][1] - 0.2),
                lm["left_hip"], lm["left_knee"]
            )
            angles_genou_G[i]    = angle_2d(
                lm["left_hip"], lm["left_knee"], lm["left_ankle"]
            )
            angles_cheville_G[i] = angle_2d(
                lm["left_knee"], lm["left_ankle"],
                (lm["left_ankle"][0] + 0.1, lm["left_ankle"][1])
            )
            angles_hanche_D[i]   = angle_2d(
                (lm["right_hip"][0],   lm["right_hip"][1] - 0.2),
                lm["right_hip"], lm["right_knee"]
            )
            angles_genou_D[i]    = angle_2d(
                lm["right_hip"], lm["right_knee"], lm["right_ankle"]
            )
            angles_cheville_D[i] = angle_2d(
                lm["right_knee"], lm["right_ankle"],
                (lm["right_ankle"][0] + 0.1, lm["right_ankle"][1])
            )

        # ── Construction état s(t) 42-dim ─────────────────────
        # Angles (18 dims) + vitesses (18 dims) + GRF simulé (6 dims)
        dt = 1.0 / fps

        def vitesse(angles):
            """Différentiation numérique centrale."""
            v = np.gradient(angles, dt)
            return v

        etats = np.zeros((T, 42), dtype=np.float32)

        # Angles (dims 0-17) — 3 axes simulés par articulation
        etats[:, 0]  = angles_hanche_G
        etats[:, 1]  = angles_hanche_D
        etats[:, 2]  = angles_genou_G
        etats[:, 3]  = angles_genou_D
        etats[:, 4]  = angles_cheville_G
        etats[:, 5]  = angles_cheville_D
        # Remplir les autres axes avec une variation simulée
        for col in range(6, 18):
            etats[:, col] = etats[:, col % 6] * 0.1

        # Vitesses (dims 18-35)
        etats[:, 18] = vitesse(angles_hanche_G)
        etats[:, 19] = vitesse(angles_hanche_D)
        etats[:, 20] = vitesse(angles_genou_G)
        etats[:, 21] = vitesse(angles_genou_D)
        etats[:, 22] = vitesse(angles_cheville_G)
        etats[:, 23] = vitesse(angles_cheville_D)
        for col in range(24, 36):
            etats[:, col] = etats[:, 18 + (col % 6)] * 0.1

        # GRF simulé (dims 36-41) — basé sur la position verticale des chevilles
        cheville_y = np.array([lm["left_ankle"][1] for lm in landmarks_series])
        grf_sim    = np.maximum(0, 1 - (cheville_y - cheville_y.min()) /
                     (cheville_y.max() - cheville_y.min() + 1e-8))
        etats[:, 36] = 0
        etats[:, 37] = 0
        etats[:, 38] = grf_sim
        etats[:, 39] = 0
        etats[:, 40] = 0
        etats[:, 41] = grf_sim

        # Actions simulées (18 dims) — couples à zéro (non disponibles depuis vidéo)
        actions = np.zeros((T, 18), dtype=np.float32)

        # ── Calcul du score ───────────────────────────────────
        etats_norm   = scaler_etats.transform(etats)
        actions_norm = scaler_actions.transform(actions)
        sa = np.concatenate([etats_norm, actions_norm], axis=1).astype(np.float32)

        with torch.no_grad():
            r_vals = model(torch.tensor(sa)).numpy().flatten()

        r_moy   = float(r_vals.mean())
        score   = calculer_score_normalite(r_moy)
        rapport = generer_rapport(score)

        return jsonify({
            "score":   score,
            "rapport": rapport,
            "angles": {
                "hanche_G":   angles_hanche_G.tolist(),
                "hanche_D":   angles_hanche_D.tolist(),
                "genou_G":    angles_genou_G.tolist(),
                "genou_D":    angles_genou_D.tolist(),
                "cheville_G": angles_cheville_G.tolist(),
                "cheville_D": angles_cheville_D.tolist(),
            },
            "r_serie":  r_vals[::max(1, len(r_vals)//200)].tolist(),
            "n_frames": T,
            "fps":      fps,
            "avertissement": (
                "Les couples articulaires ne sont pas disponibles depuis vidéo. "
                "Score calculé sur angles et vitesses uniquement — résultat indicatif."
            )
        })

    finally:
        os.unlink(tmp_path)

# ============================================================
# 5. LANCEMENT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("API IRL Biomécanique démarrée")
    print("URL : http://localhost:5000")
    print("="*50)
    print("\nEndpoints disponibles :")
    print("  GET  /api/health")
    print("  GET  /api/participants")
    print("  GET  /api/participant/<id>")
    print("  GET  /api/reference")
    print("  GET  /api/importance")
    print("  POST /api/score")
    print("  POST /api/analyze-video")
    print("\nInstaller les dépendances :")
    print("  pip install flask flask-cors mediapipe opencv-python")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)

