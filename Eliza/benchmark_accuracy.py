"""
benchmark_accuracy.py

Pour chaque modèle dans models/Mats/, appelle evaluate() 100 fois sur le test set
et enregistre les résultats (accuracy, temps, GPU power/temp) tous les 10 appels.
Résultats sauvegardés dans benchmark_results.json et benchmark_results.csv.

Modèles supportés :
  - BayMoped_n{n}_run{y}.pth   → hybrid split (n couches BNN) avec MOPED
  - BayNoMoped_run{y}.pth      → full BNN sans MOPED
  - BayMoped_full_run{y}.pth   → full BNN avec MOPED
  - EffNetBase_run{y}.pth      → EfficientNet déterministe (baseline)
"""

import os
import re
import json
import csv
import time
import subprocess
import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision.transforms import v2
from medmnist import PathMNIST, INFO
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn


# CONFIG

MODELS_DIR       = "models/Mats"
BASE_EFFNET_PATH = "models/effNet.pth"   # poids DNN de base pour initialiser les BNN
OUTPUT_JSON      = "benchmark_results.json"
OUTPUT_CSV       = "benchmark_results.csv"
N_TOTAL_CALLS    = 100
CHECKPOINT_EVERY = 10
NUM_MONTE_CARLO  = 100
BATCH_SIZE       = 128

# Couches BNN pour les modèles hybrides
HYBRID_N_VALUES  = [1, 6, 19, 32, 162, 201]


# DEVICE

device = torch.device(
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available() else "cpu"
)
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

BNN_PRIOR_MOPED = {
    "prior_mu": 0.0, "prior_sigma": 1.0,
    "posterior_mu_init": 0.0, "posterior_rho_init": -3.0,
    "type": "Reparameterization",
    "moped_enable": True, "moped_delta": 0.5,
}
BNN_PRIOR_NO_MOPED = {
    "prior_mu": 0.0, "prior_sigma": 1.0,
    "posterior_mu_init": 0.0, "posterior_rho_init": -3.0,
    "type": "Reparameterization",
    "moped_enable": False, "moped_delta": 0.5,
}


# GPU MONITORING (nvidia-smi)

def get_gpu_stats():
    """Retourne (power_W, temp_C) via nvidia-smi. Retourne (None, None) si indisponible."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            power = float(parts[0].strip())
            temp  = float(parts[1].strip())
            return power, temp
    except Exception:
        pass
    return None, None


# SPLIT MODEL (pour les hybrides)

def split_model_graph(model: nn.Module, n: int):
    traced  = torch.fx.symbolic_trace(model)
    nodes   = list(traced.graph.nodes)
    module_nodes = [node for node in nodes if node.op == "call_module"]
    if n >= len(module_nodes):
        raise ValueError(f"Cannot split off {n} layers; model only has {len(module_nodes)} module calls.")
    first_node_of_b = module_nodes[-n]
    cut_idx = nodes.index(first_node_of_b)
    nodes_A = nodes[:cut_idx]
    nodes_B = nodes[cut_idx:]
    boundary_nodes = []
    for node in nodes_B:
        for in_node in node.all_input_nodes:
            if in_node in nodes_A and in_node not in boundary_nodes:
                boundary_nodes.append(in_node)
    graph_A = torch.fx.Graph()
    env_A   = {}
    for node in nodes_A:
        env_A[node] = graph_A.node_copy(node, lambda x: env_A[x])
    output_args_A = tuple(env_A[node] for node in boundary_nodes)
    graph_A.output(output_args_A[0] if len(output_args_A) == 1 else output_args_A)
    model_A = torch.fx.GraphModule(traced, graph_A)
    graph_B = torch.fx.Graph()
    env_B   = {}
    for node in boundary_nodes:
        env_B[node] = graph_B.placeholder(node.name)
    for node in nodes_B:
        env_B[node] = graph_B.node_copy(node, lambda x: env_B[x])
    model_B = torch.fx.GraphModule(traced, graph_B)
    return model_A, model_B


# EVALUATE — version hybride (model_A + model_B)

def evaluate_hybrid(model_A, model_B, loader, num_monte_carlo, device):
    """Réplique exacte de evaluate() du notebook ResNetBayesianLayer."""
    model_A.eval()
    model_B.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.squeeze(1)

            output_mc = []
            for _ in range(num_monte_carlo):
                features = model_A(inputs)
                logits   = model_B(features)
                probs    = torch.nn.functional.softmax(logits, dim=-1)
                output_mc.append(probs)

            output    = torch.stack(output_mc)
            pred_mean = output.mean(dim=0)
            y_pred    = torch.argmax(pred_mean, dim=1)

            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = (all_preds == all_labels).mean()
    return acc


# EVALUATE — version full BNN (net unique)

def evaluate_full(net, loader, num_monte_carlo, device):
    """Réplique exacte de evaluate() du notebook ResNetBayesian."""
    net.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.squeeze(1)

            output_mc = []
            for _ in range(num_monte_carlo):
                logits = net(inputs)
                probs  = torch.nn.functional.softmax(logits, dim=-1)
                output_mc.append(probs)

            output    = torch.stack(output_mc)
            pred_mean = output.mean(dim=0)
            y_pred    = torch.argmax(pred_mean, dim=1)

            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = (all_preds == all_labels).mean()
    return acc


# EVALUATE — version déterministe (EffNetBase)

def evaluate_deterministic(net, loader, device):
    """Évalue un modèle déterministe classique (pas de MC sampling)."""
    net.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.squeeze(1)
            logits = net(inputs)
            y_pred = torch.argmax(logits, dim=1)
            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return (all_preds == all_labels).mean()


# LOAD RESULTS (crash-safe)

def load_results():
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            return json.load(f)
    return {}

def save_results(results):
    # Sauvegarde atomique JSON
    tmp = OUTPUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, OUTPUT_JSON)

    # CSV (réécrit entièrement à chaque sauvegarde)
    rows = []
    for model_key, model_data in results.items():
        for checkpoint in model_data.get("checkpoints", []):
            rows.append({
                "model":           model_key,
                "model_type":      model_data.get("model_type", ""),
                "n_layers":        model_data.get("n_layers", ""),
                "run_id":          model_data.get("run_id", ""),
                "call_index":      checkpoint["call_index"],
                "mean_accuracy":   checkpoint["mean_accuracy"],
                "std_accuracy":    checkpoint["std_accuracy"],
                "elapsed_s":       checkpoint["elapsed_s"],
                "gpu_power_w":     checkpoint.get("gpu_power_w", ""),
                "gpu_temp_c":      checkpoint.get("gpu_temp_c", ""),
            })
    if rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# BENCHMARK UN MODÈLE

def benchmark_model(model_key, model_type, n_layers, run_id, checkpoint_path, results):
    """
    Appelle evaluate() N_TOTAL_CALLS fois et sauvegarde tous les CHECKPOINT_EVERY appels.

    model_type : "hybrid_moped" | "full_moped" | "full_no_moped" | "effnet_base"
    n_layers   : int (pour hybrid_moped) ou None
    """
    print(f"\n{'='*60}")
    print(f"  {model_key}")
    print(f"  type={model_type}, n={n_layers}, run={run_id}")
    print(f"{'='*60}")

    # Déjà terminé ?
    if model_key in results:
        done = results[model_key].get("done", False)
        already = len(results[model_key].get("checkpoints", []))
        if done:
            print(f"  → déjà complet ({already} checkpoints), on passe.")
            return
        else:
            print(f"  → reprise depuis {already} checkpoints déjà sauvegardés.")
            start_call = already * CHECKPOINT_EVERY
    else:
        results[model_key] = {
            "model_type": model_type,
            "n_layers":   n_layers,
            "run_id":     run_id,
            "checkpoints": [],
            "done": False,
        }
        start_call = 0

    # Chargement du modèle
    if model_type == "hybrid_moped":
        # EfficientNet B0 → split → dnn_to_bnn sur model_B
        net = torchvision.models.efficientnet_b0(progress=False)
        net.classifier[1] = nn.Linear(1280, 9)
        base = torch.load(BASE_EFFNET_PATH, map_location=device)
        net.load_state_dict(base["model_state_dict"])
        model_A, model_B = split_model_graph(net, n_layers)
        dnn_to_bnn(model_B, BNN_PRIOR_MOPED)
        model_A.to(device)
        model_B.to(device)
        # Charge les poids fine-tunés de model_B
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_B.load_state_dict(ckpt["model_state_dict"])
        eval_fn = lambda: evaluate_hybrid(model_A, model_B, testloader, NUM_MONTE_CARLO, device)

    elif model_type in ("full_moped", "full_no_moped"):
        prior = BNN_PRIOR_MOPED if model_type == "full_moped" else BNN_PRIOR_NO_MOPED
        net = torchvision.models.efficientnet_b0(progress=False)
        net.classifier[1] = nn.Linear(1280, 9)
        # Charge d'abord les poids DNN de base pour MOPED (ou pas mais on applique dnn_to_bnn quand mm)
        base = torch.load(BASE_EFFNET_PATH, map_location=device)
        net.load_state_dict(base["model_state_dict"])
        dnn_to_bnn(net, prior)
        net.to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        eval_fn = lambda: evaluate_full(net, testloader, NUM_MONTE_CARLO, device)

    elif model_type == "effnet_base":
        net = torchvision.models.efficientnet_b0(progress=False)
        net.classifier[1] = nn.Linear(1280, 9)
        ckpt = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.to(device)
        eval_fn = lambda: evaluate_deterministic(net, testloader, device)

    else:
        raise ValueError(f"model_type inconnu : {model_type}")

    # Boucle 100 appels
    accuracies_window = []   # accuracies sur la fenêtre courante (10 appels)
    t_start = time.time()

    for call_i in range(start_call, N_TOTAL_CALLS):
        acc = eval_fn()
        accuracies_window.append(float(acc))
        print(f"  call {call_i+1:3d}/100 → acc={100*acc:.3f}%")

        # Sauvegarde tous les CHECKPOINT_EVERY appels
        if (call_i + 1) % CHECKPOINT_EVERY == 0:
            elapsed   = time.time() - t_start
            power, temp = get_gpu_stats()
            checkpoint = {
                "call_index":    call_i + 1,
                "mean_accuracy": float(np.mean(accuracies_window)),
                "std_accuracy":  float(np.std(accuracies_window)),
                "elapsed_s":     round(elapsed, 2),
                "gpu_power_w":   power,
                "gpu_temp_c":    temp,
            }
            results[model_key]["checkpoints"].append(checkpoint)
            print(f"  ── checkpoint {call_i+1} | mean={100*checkpoint['mean_accuracy']:.3f}% "
                  f"± {100*checkpoint['std_accuracy']:.3f}% | {elapsed:.1f}s "
                  f"| GPU {power}W {temp}°C")
            save_results(results)
            accuracies_window = []   # reset fenêtre
            t_start = time.time()   # reset timer pour la prochaine fenêtre

    results[model_key]["done"] = True
    save_results(results)
    print(f"  ✓ {model_key} terminé.")


# DÉCOUVERTE DES FICHIERS

def discover_models(models_dir):
    """
    Retourne une liste de dicts décrivant chaque modèle trouvé dans models_dir.
    Convention :
      BayMoped_n{n}_run{y}.pth    → hybrid_moped,  n_layers=n
      BayMoped_full_run{y}.pth    → full_moped,     n_layers=None
      BayNoMoped_run{y}.pth       → full_no_moped,  n_layers=None
      EffNetBase_run{y}.pth       → effnet_base,    n_layers=None
    """
    patterns = [
        (r"^BayMoped_n(\d+)_run(\d+)\.pth$",  "hybrid_moped"),
        (r"^BayMoped_full_run(\d+)\.pth$",    "full_moped"),
        (r"^BayNoMoped_run(\d+)\.pth$",       "full_no_moped"),
        (r"^EffNetBase_run(\d+)\.pth$",       "effnet_base"),
    ]

    found = []
    for fname in sorted(os.listdir(models_dir)):
        for pattern, model_type in patterns:
            m = re.match(pattern, fname)
            if m:
                groups = m.groups()
                if model_type == "hybrid_moped":
                    n_layers = int(groups[0])
                    run_id   = int(groups[1])
                else:
                    n_layers = None
                    run_id   = int(groups[0])
                found.append({
                    "fname":      fname,
                    "path":       os.path.join(models_dir, fname),
                    "model_type": model_type,
                    "n_layers":   n_layers,
                    "run_id":     run_id,
                    "key":        fname.replace(".pth", ""),
                })
                break
    return found


# MAIN

if __name__ == "__main__":
    results = load_results()
    models  = discover_models(MODELS_DIR)

    if not models:
        print(f"Aucun modèle trouvé dans {MODELS_DIR}. Vérifie les noms de fichiers.")
        exit(1)

    print(f"\n{len(models)} modèle(s) trouvé(s) dans {MODELS_DIR} :")
    for m in models:
        status = "✓ done" if results.get(m["key"], {}).get("done") else "à faire"
        print(f"  [{status}] {m['fname']}  (type={m['model_type']}, n={m['n_layers']}, run={m['run_id']})")

    for m in models:
        benchmark_model(
            model_key      = m["key"],
            model_type     = m["model_type"],
            n_layers       = m["n_layers"],
            run_id         = m["run_id"],
            checkpoint_path= m["path"],
            results        = results,
        )

    print(f"\n✓ Benchmark terminé. Résultats dans {OUTPUT_JSON} et {OUTPUT_CSV}.")