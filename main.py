import argparse
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
from src.dataset import load_graphs
from src.models import GCN, GraphSAGE, GAT
from src.train import train_model

def run_benchmark(seed=None):
    num_seeds = 10
    seeds = range(num_seeds) if seed is None else [seed]
    embedding_size = 128
    learning_rate = 0.005
    num_classes = 1
    
    print("Loading data...")
    (x_train, adj_train, _, y_train,
     x_val, adj_val, _, y_val,
     x_test, adj_test, _, y_test) = load_graphs(augment=False)

    models_to_test = [
        {"name": "GCN", "class": GCN, "params": {"hidden_dim": embedding_size, "num_layers": 2, "dropout": 0.5}},
        {"name": "SAGE", "class": GraphSAGE, "params": {"hidden_dim": embedding_size, "num_layers": 2, "aggregator": "pooling", "dropout": 0.5}},
        {"name": "GAT", "class": GAT, "params": {"hidden_units": embedding_size // 8, "num_heads": 8, "num_layers": 2, "dropout": 0.6}}
    ]

    results = {}

    for m_spec in models_to_test:
        name = m_spec["name"]
        print(f"\nBenchmarking {name} across {len(seeds)} seed(s)...")
        
        roc_aucs = []
        f1s = []
        best_seed_val_auc = -1.0
        
        for seed in seeds:
            tf.keras.backend.clear_session()
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
            model = m_spec["class"](num_classes=num_classes, **m_spec["params"])
            
            history = train_model(
                model, x_train, adj_train, y_train, x_val, adj_val, y_val,
                lr=learning_rate, epochs=1000, patience=100, batch_size=32, model_name=f"{name}_seed_{seed}"
            )

            # @Marcos: seleccionar pesos con el ROC-AUC de test filtraba
            # información del conjunto reservado y sesgaba su evaluación.
            # EarlyStopping ya restauró los mejores pesos por validación.
            best_val_auc = max(history["val_roc_auc"])
            if best_val_auc > best_seed_val_auc:
                best_seed_val_auc = best_val_auc
                os.makedirs("models/weights", exist_ok=True)
                model.save_weights(f"models/weights/{name}.weights.h5")
            
            # Inference
            logits = model([x_test, adj_test], training=False)
            probs = tf.nn.sigmoid(logits).numpy()[:, 0]
            preds = (probs > 0.5).astype(int)
            
            auc = roc_auc_score(y_test, probs)
            f1 = f1_score(y_test, preds)
            
            roc_aucs.append(auc)
            f1s.append(f1)
            print(f"  Seed {seed}: ROC-AUC = {auc:.4f}, F1 = {f1:.4f}")
            
        results[name] = {
            "roc_auc_mean": np.mean(roc_aucs),
            "roc_auc_std": np.std(roc_aucs),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s)
        }
        
    print("\n" + "="*30)
    print("FINAL BENCHMARK RESULTS")
    print("="*30)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  ROC-AUC: {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")
        print(f"  F1-Score: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the BACE GNN benchmark")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run only this random seed. Omit to benchmark the default 10 seeds (0-9).",
    )
    args = parser.parse_args()
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be a non-negative integer")
    return args


if __name__ == "__main__":
    run_benchmark(seed=parse_args().seed)
