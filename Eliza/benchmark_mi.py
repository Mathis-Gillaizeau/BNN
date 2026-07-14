"""
benchmark_mi_repetitions.py

Objectif : mesurer comment la Mutual Information (MI) et l'entropie predictive
se stabilisent en fonction du nombre de repetitions independantes du forward
pass sur un reseau bayesien (aucune boucle Monte Carlo interne : chaque run
complet du programme re-echantillonne les poids bayesiens via bayesian_torch,
donc N runs independants == N echantillons MC).

Pour chaque valeur de REPEATS_LIST (10, 20, 100) :
  - relance N fois un forward pass complet sur le testloader
  - empile les sorties softmax -> (N, dataset_size, n_classes)
  - calcule l'entropie / la MI par classe (aleatoire vs epistemique)
  - sauvegarde un stacked bar chart (% entropie / % MI) + un resume JSON

Crash-safe : chaque run individuel est sauvegarde sur disque des qu'il est
termine. Si le script est interrompu, il reprend au run non termine plutot
que de tout refaire.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from medmnist import INFO, PathMNIST
from torchvision.transforms import v2

# Config

REPEATS_LIST = [10, 20, 100]
MODEL_PATH = "models/effnet_con_Bayesian_sin_moped.pth"
OUTPUT_DIR = Path("benchmark_mi_output")
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(exist_ok=True)


# Fonctions usuelles

def predictive_entropy(output):
    if output.ndim == 3:
        mean_output = np.mean(output, axis=0)
        return -np.sum(mean_output * np.log(mean_output + 1e-10), axis=-1)
    return -np.sum(output * np.log(output + 1e-10), axis=-1)


def mutual_information(output):
    if output.ndim == 3:
        mean_output = np.mean(output, axis=0)
        entropy = -np.sum(mean_output * np.log(mean_output + 1e-10), axis=-1)
        expected_entropy = -np.mean(np.sum(output * np.log(output + 1e-10), axis=-1), axis=0)
        return entropy - expected_entropy
    return np.zeros(output.shape[0], dtype=float)


def get_gpu_stats():
    """Recupere l'utilisation GPU via nvidia-smi (None si pas de GPU)."""
    if DEVICE.type != "cuda":
        return {"gpu_util_pct": None, "gpu_mem_used_mb": None}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        util, mem = result.stdout.strip().split(",")
        return {"gpu_util_pct": float(util), "gpu_mem_used_mb": float(mem)}
    except Exception:
        return {"gpu_util_pct": None, "gpu_mem_used_mb": None}


def atomic_write_json(path, data):
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    shutil.move(tmp_path, path)



# Chargement modele + donnees (identique a ton notebook)

def build_model():
    net = torchvision.models.efficientnet_b0(progress=True)
    net.classifier[1] = nn.Linear(1280, 9)

    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.5,
    }
    dnn_to_bnn(net, const_bnn_prior_parameters)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(DEVICE)
    return net


def build_dataloaders():
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = PathMNIST(split="train", download=True, size=28, transform=transform)
    valset = PathMNIST(split="val", download=True, size=28, transform=transform)
    testset = PathMNIST(split="test", download=True, size=28, transform=transform)

    loaders = {
        "train": torch.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        ),
        "val": torch.utils.data.DataLoader(
            valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        ),
        "test": torch.utils.data.DataLoader(
            testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        ),
    }
    class_names = INFO["pathmnist"]["label"]
    return loaders, class_names



# Un seul forward pass complet sur le loader 

def single_pass(net, loader):
    net.eval()  # les couches bayesiennes de bayesian_torch re-echantillonnent
                # leurs poids a chaque appel, meme en mode eval
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.squeeze(1)
            logits = net(inputs)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)



# Benchmark crash-safe : accumule N runs, reprend si interrompu

def run_repeated_passes(net, loader, n_repeats, tag):
    """
    Relance `n_repeats` fois le forward pass complet, sauvegarde chaque run
    individuellement (reprise possible si le script crashe en cours de route).
    Retourne all_outputs de forme (n_repeats, N_samples, n_classes) et all_labels.
    """
    run_dir = OUTPUT_DIR / f"runs_{tag}"
    run_dir.mkdir(exist_ok=True)
    labels_path = run_dir / "labels.npy"

    outputs = []
    for run_idx in range(n_repeats):
        run_path = run_dir / f"run_{run_idx:03d}.npy"

        if run_path.exists():
            # reprise : ce run a deja ete fait lors d'une execution precedente
            probs = np.load(run_path)
        else:
            t0 = time.time()
            probs, run_labels = single_pass(net, loader)
            elapsed = time.time() - t0

            tmp_path = str(run_path) + ".tmp.npy"
            np.save(tmp_path, probs)
            shutil.move(tmp_path, run_path)

            if not labels_path.exists():
                np.save(labels_path, run_labels)

            gpu_stats = get_gpu_stats()
            print(f"[{tag}] run {run_idx + 1}/{n_repeats} "
                  f"({elapsed:.1f}s) gpu_util={gpu_stats['gpu_util_pct']}%")

        outputs.append(probs)

    labels = np.load(labels_path)
    return np.stack(outputs, axis=0), labels  # (n_repeats, N_samples, n_classes)



# Plot : decomposition entropie / MI par classe, en %

def plot_uncertainty_decomposition(all_outputs, all_labels, class_names, title, save_path):
    H = predictive_entropy(all_outputs)
    mi = mutual_information(all_outputs)
    aleatoric = H - mi

    n_classes = len(class_names)
    aleatoric_pct, epistemic_pct = [], []

    for c in range(n_classes):
        mask = (all_labels == c)
        a_mean = aleatoric[mask].mean()
        e_mean = mi[mask].mean()
        total = a_mean + e_mean
        aleatoric_pct.append(100 * a_mean / total if total > 0 else 0)
        epistemic_pct.append(100 * e_mean / total if total > 0 else 0)

    x = np.arange(n_classes)
    plt.figure(figsize=(12, 5))
    plt.bar(x, aleatoric_pct, label="Entropie (aleatorique)")
    plt.bar(x, epistemic_pct, bottom=aleatoric_pct, label="Mutual Information (epistemique)")
    plt.xlabel("Classe")
    plt.ylabel("% de l'incertitude totale")
    plt.xticks(x, list(class_names.values()), rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot sauvegarde : {save_path}")



# Tableau : MI (avg +- std) pour predictions vraies / fausses, par n_repeats

def build_mi_table(mi_by_repeats, preds_correct_by_repeats, repeats_list):
    """
    mi_by_repeats: dict {n_repeats: mi_array}
    preds_correct_by_repeats: dict {n_repeats: bool_array (True = prediction correcte)}
    """
    rows = ["Predictions fausses", "Predictions vraies"]
    table = pd.DataFrame(index=rows, columns=[str(n) for n in repeats_list])

    for n_repeats in repeats_list:
        mi = mi_by_repeats[n_repeats]
        correct = preds_correct_by_repeats[n_repeats]

        mi_wrong = mi[~correct]
        mi_right = mi[correct]

        cell_wrong = f"{mi_wrong.mean():.4f} +- {mi_wrong.std():.4f}"
        cell_right = f"{mi_right.mean():.4f} +- {mi_right.std():.4f}"

        table.loc["Predictions fausses", str(n_repeats)] = cell_wrong
        table.loc["Predictions vraies", str(n_repeats)] = cell_right

    return table


def save_mi_table(table, csv_path, image_path):
    table.to_csv(csv_path)
    print(f"Tableau MI sauvegarde : {csv_path}")

    fig, ax = plt.subplots(figsize=(2 + 2.2 * len(table.columns), 1.5 + 0.6 * len(table.index)))
    ax.axis("off")
    mpl_table = ax.table(
        cellText=table.values,
        rowLabels=table.index,
        colLabels=[f"{c} reps" for c in table.columns],
        cellLoc="center",
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    mpl_table.scale(1, 1.8)
    plt.title("MI (avg +- std) selon exactitude de la prediction", pad=20)
    plt.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Image du tableau sauvegardee : {image_path}")



# Main

def main():
    net = build_model()
    loaders, class_names = build_dataloaders()

    summary = {}

    for split_name, loader in loaders.items():
        mi_by_repeats = {}
        preds_correct_by_repeats = {}

        for n_repeats in REPEATS_LIST:
            tag = f"{split_name}_n{n_repeats}"
            print(f"\n=== [{split_name}] Benchmark avec {n_repeats} runs independants ===")

            all_outputs, all_labels = run_repeated_passes(net, loader, n_repeats, tag)

            pred_mean = all_outputs.mean(axis=0)
            preds = np.argmax(pred_mean, axis=-1)
            correct = (preds == all_labels)
            acc = correct.mean()

            H = predictive_entropy(all_outputs)
            mi = mutual_information(all_outputs)

            mi_by_repeats[n_repeats] = mi
            preds_correct_by_repeats[n_repeats] = correct

            summary[tag] = {
                "split": split_name,
                "n_repeats": n_repeats,
                "accuracy": float(acc),
                "mean_entropy": float(H.mean()),
                "mean_mi": float(mi.mean()),
            }

            plot_path = OUTPUT_DIR / f"decomposition_{tag}.png"
            plot_uncertainty_decomposition(
                all_outputs, all_labels, class_names,
                title=f"{split_name.capitalize()} - Decomposition de l'incertitude ({n_repeats} runs)",
                save_path=plot_path,
            )

            atomic_write_json(OUTPUT_DIR / "summary.json", summary)

        mi_table = build_mi_table(mi_by_repeats, preds_correct_by_repeats, REPEATS_LIST)
        print(f"\n=== [{split_name}] Tableau MI (avg +- std) selon exactitude ===")
        print(mi_table)

        save_mi_table(
            mi_table,
            csv_path=OUTPUT_DIR / f"mi_table_{split_name}.csv",
            image_path=OUTPUT_DIR / f"mi_table_{split_name}.png",
        )

    print("\n=== Resume global ===")
    for tag, res in summary.items():
        print(f"{tag}: acc={100 * res['accuracy']:.2f}% "
              f"mean_H={res['mean_entropy']:.4f} mean_MI={res['mean_mi']:.4f}")


if __name__ == "__main__":
    main()