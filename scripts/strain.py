import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import create_quick_config
from src.runner import ExperimentRunner

if __name__ == "__main__":
    config = create_quick_config(
        experiment_name="smoke_training",
        dataset_type=0,
        model_name="FGMambaHX",
        epochs=1,
        batch_size=128,
    )
    config.optimizer_name = "adamw"
    config.weight_decay = 0.01
    config.label_smoothing = 0.1
    config.warmup_epochs = 1
    config.min_learning_rate = 1e-6
    config.enable_training = True
    config.enable_testing = False
    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()
