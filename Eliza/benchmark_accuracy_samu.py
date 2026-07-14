"""
benchmark_accuracy_samu.py

Pour le modèle models/effnet_con_Bayesian.pth, appelle evaluate() 100 fois sur le
test set (CPU only, fidèle au notebook) et enregistre les résultats (accuracy,
temps, CPU usage/temp) tous les 10 appels.
Résultats sauvegardés dans benchmark_results_samu.json et benchmark_results_samu.csv.

Modèle : EfficientNet-B0 complet converti en BNN (dnn_to_bnn) avec MOPED.
La fonction evaluate() ici ne fait QU'UNE seule passe stochastique par appel
(pas de boucle MC interne) — c'est pour ça qu'on la boucle 100 fois en externe.
"""

import os
import json
import csv
import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision.transforms import v2
from medmnist import PathMNIST, INFO
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# CONFIG

CHECKPOINT_PATH   = "models/effnet_con_Bayesian.pth"
OUTPUT_JSON       = "benchmark_results_samu.json"
OUTPUT_CSV        = "benchmark_results_samu.csv"
N_TOTAL_CALLS     = 100
CHECKPOINT_EVERY  = 10
BATCH_SIZE        = 128


# DEVICE — CPU only à cause de la fonction

device = torch.device("cpu")
print(f"Device : {device}")


# DATASET (test uniquement)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset    = PathMNIST(split="test", download=True, size=28, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)


# BNN PRIOR PARAMETERS 

const_bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Reparameterization",
    "moped_enable": True,
    "moped_delta": 0.5,
}


# CPU MONITORING

def get_cpu_stats():
    """
    Retourne (cpu_percent, cpu_temp_c).
    cpu_percent toujours dispo via psutil.
    cpu_temp_c = None si aucun capteur exploitable trouvé sur la machine.
    """
    if not PSUTIL_AVAILABLE:
        return None, None

    cpu_percent = psutil.cpu_percent(interval=0.5)

    cpu_temp = None
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            # Cherche un capteur "core"/"cpu" en priorité, sinon prend le premier dispo
            for label_key in ("coretemp", "cpu_thermal", "k10temp", "zenpower"):
                if label_key in temps and temps[label_key]:
                    cpu_temp = temps[label_key][0].current
                    break
            if cpu_temp is None:
                first_key = next(iter(temps))
                if temps[first_key]:
                    cpu_temp = temps[first_key][0].current
    except Exception:
        cpu_temp = None

    return cpu_percent, cpu_temp


# EVALUATE — comme dans notebook function_Samu

def evaluate(net):
    net.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0], data[1]
            labels = labels.squeeze(1)

            logits = net(inputs)
            probs  = torch.nn.functional.softmax(logits, dim=-1)

            y_pred = torch.argmax(probs, dim=1)

            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = (all_preds == all_labels).mean()
    return acc


# LOAD / SAVE RESULTS (crash-safe)

def load_results():
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            return json.load(f)
    return {"checkpoints": [], "done": False}

def save_results(results):
    tmp = OUTPUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, OUTPUT_JSON)

    rows = []
    for checkpoint in results.get("checkpoints", []):
        rows.append({
            "call_index":    checkpoint["call_index"],
            "mean_accuracy": checkpoint["mean_accuracy"],
            "std_accuracy":  checkpoint["std_accuracy"],
            "elapsed_s":     checkpoint["elapsed_s"],
            "cpu_percent":   checkpoint.get("cpu_percent", ""),
            "cpu_temp_c":    checkpoint.get("cpu_temp_c", ""),
        })
    if rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# CHARGEMENT DU MODÈLE

def load_model():
    net = torchvision.models.efficientnet_b0(progress=True)
    net.classifier[1] = nn.Linear(1280, 9)

    dnn_to_bnn(net, const_bnn_prior_parameters)

    red_bay = torch.load(CHECKPOINT_PATH, map_location=device)
    net.load_state_dict(red_bay["model_state_dict"])

    return net


# MAIN

if __name__ == "__main__":
    if not PSUTIL_AVAILABLE:
        print("⚠ psutil non disponible → cpu_percent et cpu_temp seront vides dans les résultats.")

    results = load_results()
    already = len(results.get("checkpoints", []))

    if results.get("done", False):
        print(f"✓ Déjà complet ({already} checkpoints). Rien à faire.")
        print(f"  → {OUTPUT_JSON} / {OUTPUT_CSV}")
        exit(0)

    start_call = already * CHECKPOINT_EVERY
    if start_call > 0:
        print(f"→ Reprise depuis {already} checkpoints déjà sauvegardés (call {start_call}).")

    print(f"Chargement du modèle depuis {CHECKPOINT_PATH} ...")
    net = load_model()
    print("Modèle chargé.")

    accuracies_window = []
    t_start = time.time()

    for call_i in range(start_call, N_TOTAL_CALLS):
        acc = evaluate(net)
        accuracies_window.append(float(acc))
        print(f"  call {call_i+1:3d}/100 → acc={100*acc:.3f}%")

        if (call_i + 1) % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t_start
            cpu_percent, cpu_temp = get_cpu_stats()
            checkpoint = {
                "call_index":    call_i + 1,
                "mean_accuracy": float(np.mean(accuracies_window)),
                "std_accuracy":  float(np.std(accuracies_window)),
                "elapsed_s":     round(elapsed, 2),
                "cpu_percent":   cpu_percent,
                "cpu_temp_c":    cpu_temp,
            }
            results["checkpoints"].append(checkpoint)
            temp_str = f"{cpu_temp:.1f}°C" if cpu_temp is not None else "N/A"
            cpu_str  = f"{cpu_percent:.1f}%" if cpu_percent is not None else "N/A"
            print(f"  ── checkpoint {call_i+1} | mean={100*checkpoint['mean_accuracy']:.3f}% "
                  f"± {100*checkpoint['std_accuracy']:.3f}% | {elapsed:.1f}s "
                  f"| CPU {cpu_str} {temp_str}")
            save_results(results)
            accuracies_window = []
            t_start = time.time()

    results["done"] = True
    save_results(results)
    print(f"\n✓ Benchmark terminé. Résultats dans {OUTPUT_JSON} et {OUTPUT_CSV}.")