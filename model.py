"""Model construction.

Every model takes raw 0-255 RGB pixels, so training, evaluation and prediction
all feed images the same way. Augmentation lives inside the model and is only
active during training.
"""

from tensorflow import keras
from tensorflow.keras import layers

PRETRAINED = {
    # Both EfficientNet families include their own input rescaling.
    "efficientnetb0": keras.applications.EfficientNetB0,
    "efficientnetv2b0": keras.applications.EfficientNetV2B0,
}
BACKBONES = (*PRETRAINED, "custom")


def augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(40 / 360, fill_mode="nearest"),
            layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
            layers.RandomZoom(0.3, fill_mode="nearest"),
        ],
        name="augmentation",
    )


def _custom_features(x, kernel_size):
    """The original notebook's CNN, with global pooling instead of Flatten.

    Flatten -> Dense(512) on a 56x56x128 map was ~205M parameters per task;
    global average pooling brings the whole model under 1M.
    """
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in ((16, 32), (64, 128)):
        for f in filters:
            x = layers.Conv2D(f, kernel_size, padding="same", activation="relu",
                              kernel_initializer="he_normal")(x)
        x = layers.MaxPool2D(2)(x)
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, kernel_size, padding="same", activation="relu",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    return layers.Dropout(0.5)(x)


def build_model(task, backbone, input_shape, weights="imagenet"):
    inputs = keras.Input(shape=input_shape)
    x = augmentation()(inputs)
    if backbone in PRETRAINED:
        base = PRETRAINED[backbone](include_top=False, weights=weights, input_shape=input_shape)
        base.trainable = False
        # training=False keeps the backbone's BatchNorm statistics fixed even
        # after it is unfrozen for fine-tuning.
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
    elif backbone == "custom":
        x = _custom_features(x, task.kernel_size)
    else:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {BACKBONES}")
    # float32 output keeps softmax stable under mixed precision.
    outputs = layers.Dense(task.num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name=f"task{task.number}_{task.key}")


def backbone_of(model):
    for layer in model.layers:
        if layer.name.startswith("efficientnet"):
            return layer
    return None


def unfreeze_top(model, n_layers):
    """Make the last n_layers of the pretrained backbone trainable (BatchNorm stays frozen)."""
    base = backbone_of(model)
    base.trainable = True
    for i, layer in enumerate(base.layers):
        frozen = i < len(base.layers) - n_layers or isinstance(layer, layers.BatchNormalization)
        layer.trainable = not frozen
