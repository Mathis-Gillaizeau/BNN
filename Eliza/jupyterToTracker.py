"""
Sauvegarde automatiquement les résultats d'un run BNN ResNet (PathMNIST)
dans un fichier JSON et un fichier CSV.
"""

import json
import csv
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# Chemins de sortie
JSON_PATH = "bnn_runs.json"
CSV_PATH  = "bnn_runs.csv"

CSV_COLUMNS = [
    "id", "date", "name",
    "lr", "epochs", "batch_size", "num_monte_carlo", "milestones", "gamma",
    "prior_mu", "prior_sigma", "posterior_mu_init", "posterior_rho_init",
    "bnn_type", "moped_enable", "moped_delta",
    "train_accuracy", "val_accuracy", "test_accuracy",
    "ece", "uce", "ace", "auce"
    "thresholds_unc", "uncertain_when_inaccurate",
    "thresholds_conf", "confident_when_accurate",
    "uce_bin_uncertainty", "uce_bin_error",
    "ace_bin_confidence", "ace_bin_accuracy",
    "notes",
]


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

CLASS_TO_COL = {
    "adipose":                                  "class_adipose",
    "background":                               "class_background",
    "debris":                                   "class_debris",
    "lymphocytes":                              "class_lymphocytes",
    "mucus":                                    "class_mucus",
    "smooth muscle":                            "class_smooth_muscle",
    "normal colon mucosa":                      "class_normal_colon_mucosa",
    "cancer-associated stroma":                 "class_cancer_associated_stroma",
    "colorectal adenocarcinoma epithelium":     "class_colorectal_adenocarcinoma_epithelium",
}

CLASS_NAMES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]


def log_run(
    name: str,
    # hyperparamètres
    lr: float = 0.0,
    epochs: int = 0,
    batch_size: int = 0,
    num_monte_carlo: int = 0,
    milestones: list = [],
    gamma: float = 0.0,
    # prior
    prior_mu: float = 0.0,
    prior_sigma: float = 0.0,
    posterior_mu_init: float = 0.0,
    posterior_rho_init: float = 0.0,
    bnn_type: str = "",
    moped_enable: bool = False,
    moped_delta: float = 0.0,
    # métriques
    train_accuracy: float = 0.0,
    val_accuracy: float = 0.0,
    test_accuracy: float = 0.0,
    ece: float = 0.0,
    uce: float = 0.0,
    ace: float = 0.0,
    auce: float = 0.0,
    mean_predictive_entropy: float = 0.0,
    mean_mutual_information: float = 0.0,
    # accuracy par classe : dict {nom_classe: float (en %)}
    class_accuracy: dict ={},
    notes: str = "",
    # données pour les plots
    train_losses: list = [],
    val_losses: list = [],
    mean_pred_uncertainty_per_class: list = [],
    mean_model_uncertainty_per_class: list = [],
    uncertainty_bin_model: list = [],
    uncertainty_bin_pred: list = [],
    calib_bin_conf: list = [],
    calib_bin_acc: list = [],
    calib_bin_sizes: list = [],
    thresholds_unc: list = [],
    uncertain_when_inaccurate: list = [],
    thresholds_conf: list = [],
    confident_when_accurate: list = [],
    uce_bin_uncertainty: list = [],
    uce_bin_error: list = [],
    ace_bin_confidence: list = [],
    ace_bin_accuracy: list = [],
    auce_bin_error: list = [],
    auce_bin_uncertainty: list = [],
) -> dict:
    """
    Enregistre un run dans bnn_runs.json et bnn_runs.csv.

    Paramètres

    name : str
        Identifiant lisible du run (ex : "run_lr006_ep25_moped")
    train_accuracy : float
        Accuracy finale sur le train set en %
    val_accuracy : float
        Accuracy finale sur le val set en %
    test_accuracy : float
        Accuracy globale en % (ex : 83.5)
    ece : float
        Expected Calibration Error en % (ex : 3.2)
    uce : float
        Uncertainty Calibration Error
    ace : float
        Adaptive Calibration Error
    auce : float
        Adaptive Uncertainty Calibration Error
    mean_predictive_entropy : float
        Moyenne de l'entropie prédictive sur le test set
    mean_mutual_information : float
        Moyenne de la mutual information sur le test set
    class_accuracy : dict
        {nom_classe: accuracy_en_%}
        Les clés doivent correspondre aux noms PathMNIST.
        Peut aussi être le dict correct_pred de ton notebook
        si tu passes les % déjà calculés.
    train_losses : list
        Liste des train losses par epoch
    val_losses : list
        Liste des val losses par epoch
    mean_pred_uncertainty_per_class : list
        Incertitude prédictive moyenne par classe (longueur 9)
    mean_model_uncertainty_per_class : list
        Incertitude modèle moyenne par classe (longueur 9)
    uncertainty_bin_model : list
        Axe X de la courbe d'uncertainty (model_means)
    uncertainty_bin_pred : list
        Axe Y de la courbe d'uncertainty (pred_means)
    calib_bin_conf : list
        bin_conf du reliability diagram
    calib_bin_acc : list
        bin_acc du reliability diagram
    calib_bin_sizes : list
        bin_sizes du reliability diagram
    thresholds_unc : list
        Seuils normalisés [0,1] pour le plot Uncertain when Inaccurate
    uncertain_when_inaccurate : list
        p(uncertain | inaccurate) pour chaque seuil
    thresholds_conf : list
        Seuils [0,1] pour le plot Confidence vs Accuracy
    confident_when_accurate : list
        Fraction de prédictions correctes au-dessus de chaque seuil de confiance
    uce_bin_uncertainty : list
        Axe X du plot UCE (bin_uncertainty)
    uce_bin_error : list
        Axe Y du plot UCE (bin_error)
    ace_bin_confidence : list
        Axe X du plot ACE (bin_uncertainty adaptatif)
    ace_bin_accuracy : list
        Axe Y du plot ACE (bin_error adaptatif)

    Retourne le dict du run (pratique pour vérifier).
    """
    run_id = int(datetime.now().timestamp() * 1000)
    date   = datetime.now().strftime("%d/%m/%Y %H:%M")

    run = {
        "id":    run_id,
        "date":  date,
        "name":  name,
        "hp": {
            "lr":               lr,
            "epochs":           epochs,
            "batch_size":       batch_size,
            "num_monte_carlo":  num_monte_carlo,
            "milestones":       milestones,
            "gamma":            gamma,
        },
        "prior": {
            "prior_mu":           prior_mu,
            "prior_sigma":        prior_sigma,
            "posterior_mu_init":  posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type":               bnn_type,
            "moped_enable":       moped_enable,
            "moped_delta":        moped_delta,
        },
        "metrics": {
            "train_accuracy":           train_accuracy,
            "val_accuracy":             val_accuracy,
            "test_accuracy":            test_accuracy,
            "ece":                      ece,
            "uce":                      uce,
            "ace":                     ace,
            "auce":                    auce,
            "mean_predictive_entropy":  mean_predictive_entropy,
            "mean_mutual_information":  mean_mutual_information,
        },
        "class_accuracy": class_accuracy or {},
        "plots": {
            "train_losses":                     train_losses,
            "val_losses":                       val_losses,
            "mean_pred_uncertainty_per_class":  mean_pred_uncertainty_per_class,
            "mean_model_uncertainty_per_class": mean_model_uncertainty_per_class,
            "uncertainty_bin_model":            uncertainty_bin_model,
            "uncertainty_bin_pred":             uncertainty_bin_pred,
            "calib_bin_conf":                   calib_bin_conf,
            "calib_bin_acc":                    calib_bin_acc,
            "calib_bin_sizes":                  calib_bin_sizes,
            "thresholds_unc":                   thresholds_unc,
            "uncertain_when_inaccurate":        uncertain_when_inaccurate,
            "thresholds_conf":                  thresholds_conf,
            "confident_when_accurate":          confident_when_accurate,
            "uce_bin_uncertainty":              uce_bin_uncertainty,
            "uce_bin_error":                    uce_bin_error,
            "ace_bin_confidence":               ace_bin_confidence,
            "ace_bin_accuracy":                 ace_bin_accuracy,
            "auce_bin_error":                   auce_bin_error,
            "auce_bin_uncertainty":             auce_bin_uncertainty,
        },
        "notes": notes,
    }

    _save_json(run)
    _save_csv(run)

    print(f"[tracker] ✓ Run '{name}' sauvegardé")
    print(f"          JSON → {os.path.abspath(JSON_PATH)}")
    print(f"          CSV  → {os.path.abspath(CSV_PATH)}")
    if test_accuracy is not None:
        print(f"          Accuracy={test_accuracy:.2f}%  ECE={ece:.2f}%  "
              f"H={mean_predictive_entropy:.4f}  MI={mean_mutual_information:.4f}")
    return run


def plot_run(name: str):
    """
    Relit le JSON et refait les 4 plots d'un run à partir de son nom.
    """
    run = _find_run(name)
    if run is None:
        print(f"[tracker] Run '{name}' introuvable dans {JSON_PATH}")
        return

    p   = run.get("plots", {})
    m   = run.get("metrics", {})
    ece = m.get("ece", 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(name, fontsize=13, fontweight="bold")

    # 1 — Train / Val loss
    ax = axes[0, 0]
    tl = p.get("train_losses", [])
    vl = p.get("val_losses", [])
    if tl or vl:
        if tl: ax.plot(range(1, len(tl) + 1), tl, label='train_loss')
        if vl: ax.plot(range(1, len(vl) + 1), vl, label='val_loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    ax.set_title("Train / Val Loss")

    # 2 — Uncertainty curve
    ax = axes[0, 1]
    bm = p.get("uncertainty_bin_model", [])
    bp = p.get("uncertainty_bin_pred", [])
    if bm and bp:
        ax.plot(bm, bp, label='Predictive Uncertainty')
        ax.plot(bm, bm, label='Model Uncertainty')
        ax.set_xlabel('Model Uncertainty (MI)')
        ax.set_ylabel('Predictive Uncertainty (H)')
        ax.legend()
    ax.set_title("Uncertainty Curve")

    # 3 — Uncertainty par classe
    ax = axes[1, 0]
    mpu = p.get("mean_pred_uncertainty_per_class", [])
    mmu = p.get("mean_model_uncertainty_per_class", [])
    if mpu or mmu:
        x = np.arange(len(CLASS_NAMES))
        width = 0.4
        if mpu: ax.bar(x - width/2, mpu, width, label='Ep (aleatoric)')
        if mmu: ax.bar(x + width/2, mmu, width, label='H - Ep (epistemic)')
        ax.set_xlabel('Classe')
        ax.set_ylabel('Uncertainty')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.legend()
    ax.set_title("Uncertainty par classe")

    # 4 — Reliability diagram
    ax = axes[1, 1]
    bc = p.get("calib_bin_conf", [])
    ba = p.get("calib_bin_acc", [])
    if bc and ba:
        ax.bar(bc, ba, width=0.05, alpha=0.8, label='Outputs')
        ax.plot([0, 1], [0, 1], '--', label='Parfait')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Expected Calibration Error (ECE={ece:.2f} %)')
        ax.legend()
    else:
        ax.set_title("Reliability Diagram")

    plt.tight_layout()
    plt.show()

    # 5 — Uncertain when Inaccurate + Confidence vs Accuracy
    tu = p.get("thresholds_unc", [])
    uwi = p.get("uncertain_when_inaccurate", [])
    tc = p.get("thresholds_conf", [])
    cwa = p.get("confident_when_accurate", [])

    if (tu and uwi) or (tc and cwa):
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        fig2.suptitle(name, fontsize=13, fontweight="bold")

        ax = axes2[0]
        if tu and uwi:
            ax.plot(tu, uwi)
            ax.set_xlabel('Uncertainty thresholds')
            ax.set_ylabel('p(uncertain | inaccurate)')
        ax.set_title("Uncertain when Inaccurate (↑)")

        ax = axes2[1]
        if tc and cwa:
            ax.plot(tc, cwa)
            ax.set_xlabel('Confidence threshold κ')
            ax.set_ylabel('Fraction of accurate predictions')
        ax.set_title("Confidence vs Accuracy (↑)")

        plt.tight_layout()
        plt.show()

    # 6 — UCE + ACE
    ubu = p.get("auce_bin_uncertainty", [])
    ube = p.get("auce_bin_error", [])
    abu = p.get("ace_bin_confidence", [])
    abe = p.get("ace_bin_accuracy", [])

    if (ubu and ube) or (abu and abe):
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
        fig3.suptitle(name, fontsize=13, fontweight="bold")

        ax = axes3[0]
        if ubu and ube:
            ax.plot(ubu, ube, marker='o')
            ax.set_xlabel('Predictive Uncertainty')
            ax.set_ylabel('Error Rate')
        ax.set_title("Expected Uncertainty Error (AUCE)")

        ax = axes3[1]
        if abu and abe:
            ax.plot(abu, abe, marker='o')
            ax.set_xlabel('Predictive Uncertainty')
            ax.set_ylabel('Error Rate')
        ax.set_title("Adaptive Calibration Error (ACE)")

        plt.tight_layout()
        plt.show()


def load_runs() -> list:
    """Charge tous les runs depuis le JSON. Utile pour comparer."""
    if not os.path.exists(JSON_PATH):
        return []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary():
    """Affiche un résumé compact de tous les runs enregistrés."""
    runs = load_runs()
    if not runs:
        print("Aucun run enregistré.")
        return
    print(f"{'Nom':<45} {'Acc%':>6} {'ECE%':>6} {'H':>7} {'MI':>7}  σ prior  ρ_init  MOPED")
    print("-" * 95)
    for r in runs:
        m = r.get("metrics", {})
        p = r.get("prior", {})
        print(
            f"{r['name']:<45} "
            f"{_fmt(m.get('test_accuracy'), '.2f'):>6} "
            f"{_fmt(m.get('ece'), '.2f'):>6} "
            f"{_fmt(m.get('mean_predictive_entropy'), '.4f'):>7} "
            f"{_fmt(m.get('mean_mutual_information'), '.4f'):>7}  "
            f"{_fmt(p.get('prior_sigma'), '.1f'):>6}  "
            f"{_fmt(p.get('posterior_rho_init'), '.1f'):>6}  "
            f"{'oui' if p.get('moped_enable') else 'non'}"
        )


# Helpers internes

def _fmt(v, spec):
    return format(v, spec) if v is not None else "—"

def _find_run(name: str):
    for r in load_runs():
        if r["name"] == name:
            return r
    return None

def _save_json(run: dict):
    runs = load_runs()
    runs.insert(0, run)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def _save_csv(run: dict):
    file_exists = os.path.exists(CSV_PATH)
    p = run.get("plots", {})
    row = {
        "id":    run["id"],
        "date":  run["date"],
        "name":  run["name"],
        **{k: run["hp"].get(k) for k in ["lr", "epochs", "batch_size",
                                          "num_monte_carlo", "gamma"]},
        "milestones":   str(run["hp"].get("milestones") or ""),
        **{k: run["prior"].get(k) for k in ["prior_mu", "prior_sigma",
                                              "posterior_mu_init",
                                              "posterior_rho_init",
                                              "moped_delta"]},
        "bnn_type":     run["prior"].get("type"),
        "moped_enable": run["prior"].get("moped_enable"),
        "train_accuracy":   run["metrics"].get("train_accuracy"),
        "val_accuracy":     run["metrics"].get("val_accuracy"),
        "test_accuracy":    run["metrics"].get("test_accuracy"),
        "ece":              run["metrics"].get("ece"),
        "uce":              run["metrics"].get("uce"),
        "ace":              run["metrics"].get("ace"),
        "auce":             run["metrics"].get("auce"),
        "thresholds_unc":            json.dumps(p.get("thresholds_unc", []), cls=NumpyEncoder),
        "uncertain_when_inaccurate": json.dumps(p.get("uncertain_when_inaccurate", []), cls=NumpyEncoder),
        "thresholds_conf":           json.dumps(p.get("thresholds_conf", []), cls=NumpyEncoder),
        "confident_when_accurate":   json.dumps(p.get("confident_when_accurate", []), cls=NumpyEncoder),
        "ace_bin_confidence":       json.dumps(p.get("ace_bin_confidence", []), cls=NumpyEncoder),
        "ace_bin_accuracy":             json.dumps(p.get("ace_bin_accuracy", []), cls=NumpyEncoder),
        "auce_bin_error":              json.dumps(p.get("auce_bin_error", []), cls=NumpyEncoder),
        "auce_bin_uncertainty":        json.dumps(p.get("auce_bin_uncertainty", []), cls=NumpyEncoder),
        "notes": run.get("notes", ""),
    }

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)