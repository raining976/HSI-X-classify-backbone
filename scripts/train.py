import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DATASET_LABELS, create_experiment_config_from_cli
from src.runner import ExperimentRunner


if __name__ == "__main__":
    config = create_experiment_config_from_cli()

    if config.dataset_type in DATASET_LABELS:
        print(f"当前数据集: {DATASET_LABELS[config.dataset_type]} ({config.dataset_type})")

    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()
 