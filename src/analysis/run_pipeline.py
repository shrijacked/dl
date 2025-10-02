from __future__ import annotations

import logging
from pathlib import Path

from .config import OUTPUT_CONFIG
from .geometric import run_geometric_analysis
from .image_stats import run_image_stats
from .label_analysis import run_label_analysis
from .latent_structure import run_latent_structure
from .quality_checks import run_quality_checks
from .robustness import run_robustness_probes
from .utils import ensure_output_directories


def setup_logging() -> None:
    ensure_output_directories()
    log_path = OUTPUT_CONFIG.reports / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    setup_logging()
    logging.info("Starting analysis pipeline")

    try:
        logging.info("Running label analysis")
        run_label_analysis()

        logging.info("Running image statistics")
        run_image_stats()

        logging.info("Running quality checks")
        run_quality_checks()

        logging.info("Running robustness probes")
        run_robustness_probes()

        logging.info("Running latent structure analysis")
        run_latent_structure()

        logging.info("Running geometric analysis")
        run_geometric_analysis()

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Pipeline failed: %s", exc)
        raise

    logging.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
