"""Classify new images with the trained models.

Example:
    python predict.py --runs-dir runs/effnet photo1.jpg photo2.png
"""

import argparse
import os

import numpy as np
from tensorflow import keras

import config
from train import task_dir


def load_models(runs_dir, tasks):
    models = {}
    for task in tasks:
        path = os.path.join(task_dir(runs_dir, task), "model.keras")
        if os.path.exists(path):
            models[task] = keras.models.load_model(path)
        else:
            print(f"Skipping task {task.number} ({task.title}): no model at {path}")
    return models


def load_image(path):
    # Models take raw 0-255 RGB pixels at the training resolution.
    image = keras.utils.load_img(path, target_size=config.IMAGE_SIZE, color_mode="rgb")
    return np.expand_dims(keras.utils.img_to_array(image), 0)


def predict(models, image):
    results = []
    for task, model in models.items():
        probs = model.predict(image, verbose=0)[0]
        best = int(np.argmax(probs))
        results.append((task, task.classes[best], float(probs[best])))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--runs-dir", default="runs/default")
    p.add_argument("--tasks", type=int, nargs="+", default=list(config.TASKS_BY_NUMBER))
    args = p.parse_args()

    models = load_models(args.runs_dir, [config.TASKS_BY_NUMBER[n] for n in args.tasks])
    for path in args.images:
        print(f"\n{path}")
        for task, label, confidence in predict(models, load_image(path)):
            print(f"  {task.title:<20} {label:<18} ({confidence:.0%})")


if __name__ == "__main__":
    main()
