"""Evaluate trained models on each task's test set.

Example:
    python evaluate.py --data-dir data --runs-dir runs/effnet

Writes per task <runs-dir>/task<N>_<key>/report.txt and confusion_matrix.png,
plus a summary in <runs-dir>/results.json and <runs-dir>/results.md.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow import keras

import config
from data import load_split, make_dataset
from train import task_dir


def plot_confusion(cm, task, path):
    fig, ax = plt.subplots(figsize=(1.4 * task.num_classes + 2.5, 1.2 * task.num_classes + 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=task.classes, yticklabels=task.classes, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Task {task.number}: {task.title}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_task(task, args):
    out = task_dir(args.runs_dir, task)
    model = keras.models.load_model(os.path.join(out, "model.keras"))
    X, y = load_split(args.data_dir, task, "test")
    ds = make_dataset(X, y, np.arange(len(y)), args.batch_size)
    y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)

    labels = list(range(task.num_classes))
    report = classification_report(y, y_pred, labels=labels, target_names=task.classes,
                                   digits=3, zero_division=0)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(report)
    plot_confusion(confusion_matrix(y, y_pred, labels=labels), task,
                   os.path.join(out, "confusion_matrix.png"))

    result = {
        "task": task.number,
        "title": task.title,
        "test_images": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    print(f"\n=== Task {task.number}: {task.title} ===\n{report}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--runs-dir", default="runs/default")
    p.add_argument("--tasks", type=int, nargs="+", default=list(config.TASKS_BY_NUMBER))
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = p.parse_args()

    results = [evaluate_task(config.TASKS_BY_NUMBER[n], args) for n in args.tasks]

    with open(os.path.join(args.runs_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    lines = ["| Task | Test images | Accuracy | Macro F1 |", "|---|---:|---:|---:|"]
    lines += [f"| {r['task']}. {r['title']} | {r['test_images']} | "
              f"{r['accuracy']:.1%} | {r['macro_f1']:.3f} |" for r in results]
    table = "\n".join(lines) + "\n"
    with open(os.path.join(args.runs_dir, "results.md"), "w") as f:
        f.write(table)
    print(table)


if __name__ == "__main__":
    main()
