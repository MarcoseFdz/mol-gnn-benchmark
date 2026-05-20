import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, x_test, adj_test, y_test, edge_test=None, batch_size=32):
    if edge_test is not None:
        test_inputs = (x_test, adj_test, edge_test)
    else:
        test_inputs = (x_test, adj_test)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, y_test)).batch(batch_size, drop_remainder=True)

    all_preds = []
    all_labels = []

    for inputs, labels in test_dataset:
        logits = model(inputs, training=False)
        preds = tf.argmax(logits, axis=-1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    print(classification_report(all_labels, all_preds, target_names=["Non-inhibitor", "Inhibitor"]))
    return accuracy, all_preds

def plot_training_curves(histories):
    plt.figure(figsize=(12, 4))
    for name, hist in histories.items():
        plt.plot(hist["loss"], label=f"{name} Train Loss")
        plt.plot(hist["val_loss"], linestyle="--", label=f"{name} Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_curves.png")
    plt.close()

def plot_accuracy_comparison(results):
    names = list(results.keys())
    accs = [results[name]["test_acc"] for name in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, accs, color=["blue", "green", "orange"])
    plt.title("Test Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig("accuracy_comparison.png")
    plt.close()

def plot_confusion_matrices(preds, y_test):
    fig, axes = plt.subplots(1, len(preds), figsize=(15, 4))
    if len(preds) == 1:
        axes = [axes]
    y_test_cropped = y_test[:len(list(preds.values())[0])]
    for ax, (name, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test_cropped, pred)
        im = ax.matshow(cm, cmap=plt.cm.Blues)
        ax.set_title(f"{name} Confusion Matrix")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.savefig("confusion_matrices.png")
    plt.close()
