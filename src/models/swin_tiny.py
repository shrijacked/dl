from __future__ import annotations

import argparse
import torch.nn as nn

from ..model_architectures import build_swin_tiny as _build_swin_tiny_arch
from .train_utils import TrainingConfig, add_common_cli, run_training


def build_swin_tiny(num_classes: int) -> nn.Module:
    """Build Swin Transformer Tiny using the shared model_architectures builder."""
    model, _recipe = _build_swin_tiny_arch(num_classes=num_classes, pretrained=True)
    return model


def main() -> None:
    defaults = TrainingConfig(
        model_name="swin_tiny",
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

    parser = argparse.ArgumentParser(description="Train Swin-Tiny on OrganAMNIST")
    add_common_cli(parser, defaults)
    _ = parser.parse_args()  # values are read via env in run_training; CLI sets defaults

    run_training(build_swin_tiny, defaults)


if __name__ == "__main__":
    main()


