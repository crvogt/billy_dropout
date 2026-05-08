"""Inject nn.Dropout2d modules into the YOLOv8 detection head.

Placement spec (Stochastic-YOLO recipe, translated to YOLOv8's decoupled head):
two Dropout2d modules in each classification branch (cv3) at all three
detection scales — six modules total. Regression branches (cv2) remain
deterministic so spatial detection performance stays stable.

The injection is performed once before fine-tuning; the resulting
checkpoint embeds the dropout modules. Inference uses the standard
"eval everywhere except dropouts" pattern already implemented in
MCDropoutYOLO._enable_dropout_only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


DEFAULT_DROPOUT_P = 0.25


def inject_dropout_into_yolov8_cls_head(
    detection_model: "nn.Module", p: float = DEFAULT_DROPOUT_P
) -> int:
    """Insert two ``nn.Dropout2d(p)`` modules into each YOLOv8 cls branch.

    For each scale ``i`` in the head's ``cv3`` ModuleList, the original
    branch is the Sequential ``[Conv, Conv, Conv2d]``. We replace it with
    ``[Conv, Dropout2d, Conv, Dropout2d, Conv2d]`` so dropout fires
    immediately before the second-to-last conv and immediately before the
    final 1x1 prediction conv.

    Args:
        detection_model: an ultralytics ``DetectionModel`` (i.e. ``YOLO(...).model``).
        p: dropout probability per channel.

    Returns:
        number of Dropout2d modules added (6 for stock YOLOv8).
    """
    import torch.nn as nn

    if not (0.0 < p < 1.0):
        raise ValueError(f"dropout p must be in (0, 1); got {p}")

    if not hasattr(detection_model, "model"):
        raise ValueError("expected an ultralytics DetectionModel (with .model attribute)")
    head = detection_model.model[-1]
    if not (hasattr(head, "cv3") and len(head.cv3) > 0):
        raise ValueError("model head is not YOLOv8 Detect (missing cv3 cls branches)")

    n_added = 0
    for i, branch in enumerate(head.cv3):
        layers = list(branch.children())
        if len(layers) < 3:
            raise ValueError(
                f"cv3[{i}] has {len(layers)} layers; YOLOv8 cls branch expects "
                "[Conv, Conv, Conv2d]. Architecture has changed; revisit "
                "placement before re-injecting."
            )
        head.cv3[i] = nn.Sequential(
            layers[0],
            nn.Dropout2d(p=p),
            layers[1],
            nn.Dropout2d(p=p),
            layers[2],
        )
        n_added += 2
    return n_added


def count_dropout_modules(detection_model: "nn.Module") -> int:
    """Count nn.Dropout2d modules anywhere in the model.

    Used by tests and by the training script's post-injection sanity check.
    """
    import torch.nn as nn

    return sum(1 for m in detection_model.modules() if isinstance(m, nn.Dropout2d))


def enable_dropout_train_mode(model: "nn.Module") -> None:
    """Set the model to eval mode globally, then force Dropout2d submodules
    back into train mode. Used by every stochastic forward pass.

    Important: ultralytics' YOLO.predict resets training flags inside its
    Predictor pipeline, so this helper must be called *after* the Predictor
    has set up the model — or the inference must bypass the Predictor and
    call ``model(tensor)`` directly. See MCDropoutYOLO.predict.
    """
    import torch.nn as nn

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            module.train()
