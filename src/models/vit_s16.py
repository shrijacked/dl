from __future__ import annotations

import argparse
import torch.nn as nn

from ..model_architectures import build_vit_s16 as _build_vit_s16_arch
from .train_utils import TrainingConfig, add_common_cli, run_training


def build_vit_s16(num_classes: int) -> nn.Module:
    """Build ViT-S/16 using the shared model_architectures builder."""
    model, _recipe = _build_vit_s16_arch(num_classes=num_classes, pretrained=True)
    return model


def main() -> None:
    defaults = TrainingConfig(
        model_name="vit_s16",
        input_channels=1,
        input_size=224,
        epochs=50,
        batch_size=64,
        lr=5e-4,
        momentum=0.9,
        weight_decay=5e-2,
        step_size=20,
        gamma=0.1,
        num_workers=4,
        seed=42,
    )

    parser = argparse.ArgumentParser(description="Train ViT-S/16 on OrganAMNIST")
    add_common_cli(parser, defaults)
    _ = parser.parse_args()  # values are read via env in run_training; CLI sets defaults

    run_training(build_vit_s16, defaults)


if __name__ == "__main__":
    main()


