"""
evaluate.py
-----------
Standalone evaluation script for GCN / SAGE / GAT trained on BACE for new-arch.

Outputs
-------
  results/benchmark.csv          — per-model metric table for benchmarking
  results/evaluation_chart.png   — full multi-panel visualization
"""

import os
import time
import argparse
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
)

from src.dataset import load_graphs
from src.models import GCN, GraphSAGE, GAT, GCNLayer, SAGELayer, GATLayer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, x, adj, y, batch_size=32):
    inputs = (x, adj)

    ds = (
        tf.data.Dataset.from_tensor_slices((inputs, y))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    all_probs, all_preds, all_labels = [], [], []
    t0 = time.perf_counter()
    for batch_inputs, labels in ds:
        logits = model(batch_inputs, training=False)
        probs  = tf.nn.sigmoid(logits).numpy()[:, 0]
        preds  = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    elapsed = time.perf_counter() - t0

    all_probs = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_probs, all_preds, all_labels, elapsed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, preds, probs_pos, elapsed, n_params, model_name):
    cm          = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    precision   = tp / (tp + fp + 1e-9)
    recall      = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1          = 2 * precision * recall / (precision + recall + 1e-9)
    mcc         = matthews_corrcoef(labels, preds)
    kappa       = cohen_kappa_score(labels, preds)
    bal_acc     = balanced_accuracy_score(labels, preds)

    fpr, tpr, _  = roc_curve(labels, probs_pos)
    roc_auc      = auc(fpr, tpr)
    ap           = average_precision_score(labels, probs_pos)
    prec_curve, rec_curve, _ = precision_recall_curve(labels, probs_pos)

    n_test          = len(labels)
    ms_per_sample   = (elapsed / n_test) * 1000

    return {
        "model":            model_name,
        "accuracy":         accuracy,
        "balanced_acc":     bal_acc,
        "precision":        precision,
        "recall":           recall,
        "specificity":      specificity,
        "f1":               f1,
        "mcc":              mcc,
        "kappa":            kappa,
        "roc_auc":          roc_auc,
        "avg_precision":    ap,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_params":         n_params,
        "ms_per_sample":    ms_per_sample,
        "confusion_matrix": cm,
        "fpr":              fpr,
        "tpr":              tpr,
        "prec_curve":       prec_curve,
        "rec_curve":        rec_curve,
    }


def count_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

SCALAR_FIELDS = [
    "model", "accuracy", "balanced_acc", "precision", "recall",
    "specificity", "f1", "mcc", "kappa", "roc_auc", "avg_precision",
    "tp", "tn", "fp", "fn", "n_params", "ms_per_sample",
]


def export_csv(metrics_list, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCALAR_FIELDS)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({k: (f"{m[k]:.6f}" if isinstance(m[k], float) else m[k])
                             for k in SCALAR_FIELDS})
    print(f"Benchmark CSV saved → {out_path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

PALETTE = {
    "GCN":       "#4361EE",
    "SAGE":      "#3A86FF",
    "GAT":       "#F72585",
}
DARK_BG  = "#0D1117"
PANEL_BG = "#161B22"
TEXT     = "#E6EDF3"
GRID     = "#21262D"
ACCENT   = "#30363D"


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7)


def plot_confusion_matrix(ax, cm, model_name):
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im   = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels_txt = ["Non-inh.", "Inhibitor"]
    for i in range(2):
        for j in range(2):
            color = "white" if norm[i, j] > 0.5 else TEXT
            ax.text(j, i, f"{cm[i,j]}\n({norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_txt, fontsize=7)
    ax.set_yticklabels(labels_txt, fontsize=7, rotation=90, va="center")
    _ax_style(ax, title=f"{model_name}", xlabel="Predicted", ylabel="True")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7, colors=TEXT)


def plot_roc_curves(ax, metrics_list):
    ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1)
    for m in metrics_list:
        c = PALETTE[m["model"]]
        ax.plot(m["fpr"], m["tpr"], color=c, lw=2,
                label=f"{m['model']}  AUC={m['roc_auc']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT, framealpha=0.8)
    _ax_style(ax, title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")


def plot_pr_curves(ax, metrics_list):
    for m in metrics_list:
        c = PALETTE[m["model"]]
        ax.plot(m["rec_curve"], m["prec_curve"], color=c, lw=2,
                label=f"{m['model']}  AP={m['avg_precision']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT, framealpha=0.8)
    _ax_style(ax, title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")


def plot_metric_bars(ax, metrics_list, metric_key, title, fmt=".3f"):
    names  = [m["model"] for m in metrics_list]
    values = [m[metric_key] for m in metrics_list]
    colors = [PALETTE[n] for n in names]
    bars   = ax.bar(names, values, color=colors, width=0.5, zorder=3)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:{fmt}}", ha="center", va="bottom",
                color=TEXT, fontsize=8, fontweight="bold")
    _ax_style(ax, title=title, ylabel=metric_key.replace("_", " ").title())


def plot_radar(ax, metrics_list):
    metric_keys   = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]
    N   = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_facecolor(PANEL_BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, color=TEXT, fontsize=7)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], color=TEXT, fontsize=6)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, lw=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)

    for m in metrics_list:
        vals   = [m[k] for k in metric_keys]
        vals  += vals[:1]
        c      = PALETTE[m["model"]]
        ax.plot(angles, vals, color=c, lw=2, label=m["model"])
        ax.fill(angles, vals, color=c, alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT, framealpha=0.8)
    ax.set_title("Metric Radar", color=TEXT, fontsize=9, fontweight="bold", pad=14)


def plot_stat_table(ax, metrics_list):
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")

    cols   = ["Model", "Acc", "Bal.Acc", "F1", "MCC", "κ", "AUC", "AP", "Params", "ms/sample"]
    rows   = []
    for m in metrics_list:
        rows.append([
            m["model"],
            f"{m['accuracy']:.4f}",
            f"{m['balanced_acc']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['mcc']:.4f}",
            f"{m['kappa']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{m['avg_precision']:.4f}",
            f"{m['n_params']:,}",
            f"{m['ms_per_sample']:.3f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(ACCENT)
        if row == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color=TEXT, fontweight="bold")
        else:
            cell.set_facecolor(PANEL_BG if row % 2 == 0 else "#1A2030")
            cell.set_text_props(color=TEXT)

    ax.set_title("Full Benchmark Statistics", color=TEXT,
                 fontsize=9, fontweight="bold", pad=10)


def build_chart(metrics_list, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_models = len(metrics_list)

    fig = plt.figure(figsize=(22, 20), facecolor=DARK_BG)
    fig.suptitle(
        f"GNN Benchmark — BACE Dataset (New Arch Final)  |   {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98
    )

    gs_top    = gridspec.GridSpec(1, 3, figure=fig,
                                  top=0.93, bottom=0.73, hspace=0.4, wspace=0.35)
    gs_mid    = gridspec.GridSpec(1, 3, figure=fig,
                                  top=0.68, bottom=0.48, hspace=0.4, wspace=0.35)
    gs_cm     = gridspec.GridSpec(1, n_models, figure=fig,
                                  top=0.43, bottom=0.23, hspace=0.4, wspace=0.35)
    gs_bottom = gridspec.GridSpec(1, 1, figure=fig,
                                  top=0.18, bottom=0.01)

    ax_acc   = fig.add_subplot(gs_top[0, 0])
    ax_f1    = fig.add_subplot(gs_top[0, 1])
    ax_mcc   = fig.add_subplot(gs_top[0, 2])

    ax_roc   = fig.add_subplot(gs_mid[0, 0])
    ax_pr    = fig.add_subplot(gs_mid[0, 1])
    ax_radar = fig.add_subplot(gs_mid[0, 2], polar=True)

    ax_cms   = [fig.add_subplot(gs_cm[0, i]) for i in range(n_models)]
    ax_table = fig.add_subplot(gs_bottom[0, 0])

    plot_metric_bars(ax_acc,  metrics_list, "accuracy",     "Test Accuracy")
    plot_metric_bars(ax_f1,   metrics_list, "f1",           "F1 Score")
    plot_metric_bars(ax_mcc,  metrics_list, "roc_auc",      "ROC-AUC")

    plot_roc_curves(ax_roc, metrics_list)
    plot_pr_curves(ax_pr,   metrics_list)
    plot_radar(ax_radar,    metrics_list)

    for ax, m in zip(ax_cms, metrics_list):
        plot_confusion_matrix(ax, m["confusion_matrix"], m["model"])

    plot_stat_table(ax_table, metrics_list)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Evaluation chart saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained GNN models on BACE test set")
    p.add_argument("--models_dir",  default="models/weights",
                   help="Directory containing <name>.weights.h5 files")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--out_dir",     default="results")
    return p.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(42)
    np.random.seed(42)

    print("Loading data …")
    (_, _, _, _,
     _, _, _, _,
     x_test, adj_test, _, y_test) = load_graphs(augment=False)

    print(f"Test set: {len(y_test)} samples  "
          f"({np.sum(y_test==1)} inhibitors / {np.sum(y_test==0)} non-inhibitors)")

    model_specs = [
        {"name": "GCN", "class": GCN, "params": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.5}},
        {"name": "SAGE", "class": GraphSAGE, "params": {"hidden_dim": 128, "num_layers": 2, "aggregator": "pooling", "dropout": 0.5}},
        {"name": "GAT", "class": GAT, "params": {"hidden_units": 16, "num_heads": 8, "num_layers": 2, "dropout": 0.6}}
    ]

    metrics_list = []
    print("\nLoading and Evaluating models …")
    for spec in model_specs:
        name = spec["name"]
        weights_path = os.path.join(args.models_dir, f"{name}.weights.h5")
        if not os.path.exists(weights_path):
            print(f"  Warning: Weights file not found: {weights_path}. Skipping.")
            continue
            
        model = spec["class"](num_classes=1, **spec["params"])
        # Build model by running one batch
        model([x_test[:1], adj_test[:1]])
        model.load_weights(weights_path)
        
        n_params = count_params(model)
        print(f"  {name:12s} loaded  ({n_params:,} params)  ← {weights_path}")

        probs, preds, labels, elapsed = run_inference(
            model, x_test, adj_test, y_test, batch_size=args.batch_size
        )
        m          = compute_metrics(labels, preds, probs, elapsed, n_params, name)
        metrics_list.append(m)

        print(classification_report(labels, preds,
                                    target_names=["Non-inhibitor", "Inhibitor"]))
        print(f"  ROC-AUC       : {m['roc_auc']:.4f}")
        print(f"  MCC           : {m['mcc']:.4f}")
        print(f"  Cohen κ       : {m['kappa']:.4f}")
        print(f"  Balanced Acc  : {m['balanced_acc']:.4f}")
        print(f"  Avg Precision : {m['avg_precision']:.4f}")
        print(f"  Params        : {n_params:,}")
        print(f"  ms/sample     : {m['ms_per_sample']:.3f}")

    if not metrics_list:
        print("No models evaluated. Exiting.")
        return

    csv_path   = os.path.join(args.out_dir, "benchmark.csv")
    chart_path = os.path.join(args.out_dir, "evaluation_chart.png")

    export_csv(metrics_list, csv_path)
    build_chart(metrics_list, chart_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
