"""Task definitions and default hyperparameters for the 8 Φ-Net recognition tasks.

Class names are listed in label-index order (index 0 first). They follow the
order used by the original Colab notebook; check them against the Φ-Net label
definitions for your copy of the dataset before trusting predictions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    number: int
    key: str
    title: str
    classes: tuple
    # Kernel size for the from-scratch "custom" backbone only.
    kernel_size: int = 3

    @property
    def num_classes(self):
        return len(self.classes)


TASKS = (
    Task(1, "scene_level", "Scene level", ("Pixel", "Object", "Structure")),
    Task(2, "damage_state", "Damage state", ("Damaged", "Undamaged")),
    Task(3, "spalling", "Spalling condition", ("Spalling", "Non-spalling")),
    Task(4, "material", "Material type", ("Steel", "Other")),
    Task(5, "collapse_mode", "Collapse mode", ("Non-collapse", "Partial collapse", "Full collapse")),
    Task(6, "component_type", "Component type", ("Beam", "Column", "Wall", "Other")),
    Task(7, "damage_level", "Damage level", ("No damage", "Minor", "Moderate", "Heavy"), kernel_size=5),
    Task(8, "damage_type", "Damage type", ("No damage", "Flexural", "Shear", "Combined"), kernel_size=5),
)

TASKS_BY_NUMBER = {t.number: t for t in TASKS}

IMAGE_SIZE = (224, 224)
SEED = 42
VAL_FRACTION = 0.25
BATCH_SIZE = 32

# Phase 1 trains the new classification head on a frozen backbone; phase 2
# unfreezes the top of the backbone and fine-tunes at a lower learning rate.
# The "custom" backbone has nothing pretrained, so it trains in one phase.
HEAD_EPOCHS = 20
FINE_TUNE_EPOCHS = 30
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_LAYERS = 40
CUSTOM_EPOCHS = 150
CUSTOM_LR = 1e-4
EARLY_STOPPING_PATIENCE = 10
