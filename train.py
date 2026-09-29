"""Train one model per Φ-Net task.

Example:
    python train.py --data-dir data --tasks 1 2 3 --out-dir runs/effnet

For each task this writes to <out-dir>/task<N>_<key>/:
    model.keras      best model (lowest validation loss)
    history.json     per-epoch metrics
    curves.png       accuracy / loss curves
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

import config
from data import load_split, make_dataset, train_val_indices
from model import BACKBONES, PRETRAINED, build_model, unfreeze_top


def task_dir(out_dir, task):
    return os.path.join(out_dir, f"task{task.number}_{task.key}")


class KeepBest(keras.callbacks.Callback):
    """Restore the weights with the lowest val_loss when training ends.

    Unlike EarlyStopping(restore_best_weights=True) this also restores when
    training runs to the last epoch, and it can start from a previous phase's
    best so fine-tuning never leaves the model worse than it started.
    """

    def __init__(self, best=np.inf):
        super().__init__()
        self.best = best

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best:
            self.best = logs["val_loss"]
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


def fit(model, lr, train_ds, val_ds, epochs, class_weight, patience, best=np.inf):
    """Train, leave the model at its best epoch, and return (history, best val_loss)."""
    # A fresh optimizer per phase and per model: nothing is shared between tasks.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    keep_best = KeepBest(best)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, verbose=2,
        shuffle=False,  # the dataset reshuffles itself each epoch
        callbacks=[
            keep_best,
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                              patience=max(2, patience // 3), min_lr=1e-7),
        ])
    return history.history, keep_best.best


def plot_history(history, task, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], label="val")
        ax.set_title(f"Task {task.number} ({task.title}): {metric}")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_task(task, args):
    out = task_dir(args.out_dir, task)
    os.makedirs(out, exist_ok=True)

    X, y = load_split(args.data_dir, task, "train")
    train_idx, val_idx = train_val_indices(y, config.VAL_FRACTION, config.SEED)
    train_ds = make_dataset(X, y, train_idx, args.batch_size, shuffle=True, seed=config.SEED)
    val_ds = make_dataset(X, y, val_idx, args.batch_size)

    class_weight = None
    if not args.no_class_weights:
        present = np.unique(y[train_idx])
        weights = compute_class_weight("balanced", classes=present, y=y[train_idx])
        class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    print(f"\n=== Task {task.number}: {task.title} "
          f"({len(train_idx)} train / {len(val_idx)} val, backbone={args.backbone}) ===")
    model = build_model(task, args.backbone, X.shape[1:], weights=args.weights)

    if args.backbone in PRETRAINED:
        history, best = fit(model, config.HEAD_LR, train_ds, val_ds, args.head_epochs,
                            class_weight, args.patience)
        if args.fine_tune_epochs > 0:
            unfreeze_top(model, config.FINE_TUNE_LAYERS)
            fine, best = fit(model, config.FINE_TUNE_LR, train_ds, val_ds, args.fine_tune_epochs,
                             class_weight, args.patience, best)
            history = {k: history[k] + fine[k] for k in history if k in fine}
    else:
        history, best = fit(model, config.CUSTOM_LR, train_ds, val_ds, args.epochs,
                            class_weight, args.patience)

    model.save(os.path.join(out, "model.keras"))
    history = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    plot_history(history, task, os.path.join(out, "curves.png"))
    print(f"Saved {out}/model.keras (best val_loss {best:.4f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="folder holding task<N>/task<N>_{X,y}_{train,test}.npy (or the files directly)")
    p.add_argument("--out-dir", default="runs/default")
    p.add_argument("--tasks", type=int, nargs="+", default=list(config.TASKS_BY_NUMBER))
    p.add_argument("--backbone", choices=BACKBONES, default="efficientnetb0")
    p.add_argument("--weights", default="imagenet",
                   help="pretrained weights for the backbone ('imagenet' or 'none')")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    p.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    p.add_argument("--epochs", type=int, default=config.CUSTOM_EPOCHS,
                   help="epochs for the custom backbone")
    p.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--mixed-precision", action="store_true",
                   help="use float16 compute (faster on recent GPUs)")
    args = p.parse_args()
    if args.weights == "none":
        args.weights = None

    keras.utils.set_random_seed(config.SEED)
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print("GPUs:", tf.config.list_physical_devices("GPU") or "none")

    for number in args.tasks:
        train_task(config.TASKS_BY_NUMBER[number], args)


if __name__ == "__main__":
    main()
