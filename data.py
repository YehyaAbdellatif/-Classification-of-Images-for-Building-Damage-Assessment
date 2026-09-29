"""Loading the Φ-Net .npy arrays and turning them into tf.data pipelines.

Arrays are opened with mmap_mode='r' and only the rows of each batch are read
from disk, so a task never has to fit in memory as a whole.
"""

import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def _find(data_dir, task, split, kind):
    name = f"task{task.number}_{kind}_{split}.npy"
    for path in (os.path.join(data_dir, f"task{task.number}", name), os.path.join(data_dir, name)):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"{name} not found in {data_dir}/task{task.number}/ or {data_dir}/"
    )


def load_split(data_dir, task, split):
    """Return (X, y) for 'train' or 'test'. X is memory-mapped; y is integer class ids."""
    X = np.load(_find(data_dir, task, split, "X"), mmap_mode="r")
    y = np.load(_find(data_dir, task, split, "y"))
    if len(X) != len(y):
        raise ValueError(
            f"task{task.number} {split}: {len(X)} images but {len(y)} labels; "
            "the X and y files do not belong together"
        )
    if X.ndim != 4 or X.shape[-1] != 3:
        raise ValueError(f"task{task.number} {split}: expected (N, H, W, 3) images, got {X.shape}")
    # Labels may be stored one-hot or as class ids.
    y = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(np.int64)
    if y.max() >= task.num_classes:
        raise ValueError(
            f"task{task.number} {split}: label {y.max()} but task has {task.num_classes} classes"
        )
    return X, y


def pixel_scale(X):
    """Models expect raw 0-255 pixels. Returns the factor that gets X there."""
    sample = np.asarray(X[: min(len(X), 64)])
    return 255.0 if sample.dtype.kind == "f" and sample.max() <= 1.0 else 1.0


def train_val_indices(y, val_fraction, seed):
    """Stratified split of row indices, so no image data is copied."""
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=val_fraction, random_state=seed, stratify=y)


def make_dataset(X, y, indices, batch_size, shuffle=False, seed=None):
    scale = pixel_scale(X)
    image_shape = X.shape[1:]

    def load(batch_idx):
        batch_idx = np.sort(batch_idx)  # sorted reads are faster on a memmap
        images = np.asarray(X[batch_idx], dtype=np.float32) * scale
        return images, y[batch_idx].astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices(np.asarray(indices))
    if shuffle:
        ds = ds.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    def tf_load(batch_idx):
        images, labels = tf.numpy_function(load, [batch_idx], (tf.float32, tf.int32))
        images.set_shape((None, *image_shape))
        labels.set_shape((None,))
        return images, labels

    return ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
