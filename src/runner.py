from dataclasses import asdict
import multiprocessing
import traceback
from typing import Any, List

from src.config import DATASET_LABELS, ExperimentConfig, ConfigManager, create_quick_config, create_experiment_config_from_cli
from src.email_notifier import notify_experiment_result
from src.model_registry import MODEL_REGISTRY

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def _run_single_experiment_from_dict(config_dict, dataset_type, model_name) -> None:
    child_config_dict = dict(config_dict)
    child_config_dict["dataset_type"] = dataset_type
    child_config_dict["model_name"] = model_name
    child_config_dict["actual_epoch_nums"] = None

    child_config = ExperimentConfig(**child_config_dict)
    child_runner = ExperimentRunner(child_config)
    child_runner.run_experiment()


def _run_single_experiment_with_error_queue(config_dict, dataset_type, model_name, error_queue) -> None:
    try:
        _run_single_experiment_from_dict(config_dict, dataset_type, model_name)
    except Exception:
        error_queue.put(
            (
                DATASET_LABELS.get(dataset_type, str(dataset_type)),
                model_name,
                traceback.format_exc(),
            )
        )
        raise


def _chunks(values: List[Any], size: int) -> List[List[Any]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.config_manager = ConfigManager(self.config)

        import src.config as config_module
        config_module._init(self.config)

    def run_experiment(self) -> None:
        if self._is_batch_experiment():
            self.run_batch_experiments()
            return

        print(f"开始运行实验: {self.config.experiment_name}")
        print(f"数据集: {self._get_dataset_name()}")
        print(f"模型: {self.config.model_name}")
        print("-" * 50)

        try:
            if self.config.enable_training:
                print("开始训练...")
                self.run_training()
                print("训练完成!")

            if self.config.enable_testing:
                print("开始测试...")
                self.run_testing()
                print("测试完成!")

            if self.config.enable_email_notification and self.config.enable_testing:
                try:
                    notify_experiment_result(self.config)
                    print("邮件通知已发送!")
                except Exception as e:
                    print(f"邮件通知失败: {str(e)}")

            print(f"实验 {self.config.experiment_name} 完成!")

        except Exception as e:
            print(f"实验运行出错: {str(e)}")
            raise

    def run_batch_experiments(self) -> None:
        datasets = self._get_dataset_types()
        models = self._get_model_names()
        base_config_dict = asdict(self.config)

        self._validate_batch_values(datasets, models)

        print(f"开始批量实验: {self.config.experiment_name}")
        print(f"数据集队列: {self._format_dataset_names(datasets)}")
        print(f"模型队列: {models}")
        print("模型并行数: 2")
        print("-" * 50)

        for dataset_type in datasets:
            print(f"开始数据集 {DATASET_LABELS[dataset_type]} ({dataset_type})")
            if len(models) == 1:
                _run_single_experiment_from_dict(base_config_dict, dataset_type, models[0])
                continue

            context = multiprocessing.get_context("spawn")
            for model_batch in _chunks(models, 2):
                print(f"启动模型批次: {model_batch}")
                processes = []
                error_queue = context.Queue()
                for model_name in model_batch:
                    process = context.Process(
                        target=_run_single_experiment_with_error_queue,
                        args=(base_config_dict, dataset_type, model_name, error_queue),
                        name=f"{DATASET_LABELS[dataset_type]}-{model_name}",
                    )
                    process.start()
                    processes.append(process)

                failed = []
                for process in processes:
                    process.join()
                    if process.exitcode != 0:
                        failed.append(f"{process.name}(exitcode={process.exitcode})")

                if failed:
                    errors = []
                    while not error_queue.empty():
                        dataset_name, model_name, error = error_queue.get()
                        errors.append(f"{dataset_name}-{model_name} traceback:\n{error}")
                    error_detail = "\n\n".join(errors)
                    if error_detail:
                        print(error_detail)
                    raise RuntimeError(
                        f"数据集 {DATASET_LABELS[dataset_type]} 的并行模型运行失败: {', '.join(failed)}"
                    )
            print(f"数据集 {DATASET_LABELS[dataset_type]} ({dataset_type}) 完成")

        print(f"批量实验 {self.config.experiment_name} 完成!")

    def run_training(self) -> None:
        from src.trainer import myTrain
        print(self.config_manager.get_taskInfo())
        myTrain(self.config.dataset_type, self.config.model_name)

    def run_testing(self) -> None:
        from src.evaluator import myTest
        myTest(self.config.dataset_type, self.config.model_name)

    def _get_dataset_name(self) -> str:
        if isinstance(self.config.dataset_type, list):
            return self._format_dataset_names(self.config.dataset_type)
        return DATASET_LABELS[self.config.dataset_type]

    def _get_dataset_types(self) -> List[int]:
        return _as_list(self.config.dataset_type)

    def _get_model_names(self) -> List[str]:
        return _as_list(self.config.model_name)

    def _is_batch_experiment(self) -> bool:
        return isinstance(self.config.dataset_type, list) or isinstance(self.config.model_name, list)

    def _validate_batch_values(self, datasets: List[int], models: List[str]) -> None:
        invalid_datasets = [dataset for dataset in datasets if dataset not in DATASET_LABELS]
        if invalid_datasets:
            raise ValueError(f"未知 dataset_type: {invalid_datasets}")

        invalid_models = [model for model in models if model not in MODEL_REGISTRY]
        if invalid_models:
            raise ValueError(f"未知 model_name: {invalid_models}")

    def _format_dataset_names(self, datasets: List[int]) -> str:
        return ", ".join(f"{DATASET_LABELS.get(dataset, 'Unknown')} ({dataset})" for dataset in datasets)

    def print_config(self) -> None:
        print("当前实验配置:")
        print(f"  实验名称: {self.config.experiment_name}")
        print(f"  数据集: {self._get_dataset_name()} (类型: {self.config.dataset_type})")
        print(f"  模型: {self.config.model_name}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  训练轮数: {self.config.epochs}")
        print(f"  批次大小: {self.config.batch_size}")
        print(f"  PCA通道数: {self.config.channels}")
        print(f"  窗口大小: {self.config.window_size}")
        print(f"  CUDA设备: {self.config.cuda_device}")
        print(f"  启用训练: {self.config.enable_training}")
        print(f"  启用测试: {self.config.enable_testing}")
        print(f"  启用可视化: {self.config.enable_visualization}")
        print(f"  启用t-SNE: {self.config.enable_tsne}")
        print(f"  模型保存路径: {self.config.model_save_path}")
        print("-" * 50)


def quick_run(dataset_type: int = 0,
              model_name: str = "FusAtNet",
              lr: float = 0.0001,
              epochs: int = 1,
              channels: int = 30,
              window_size: int = 11,
              cuda_device: str = "cuda:0",
              enable_visualization: bool = True,
              enable_tsne: bool = False,
              enable_training: bool = True,
              enable_testing: bool = True,
              experiment_name: str = "quick_experiment") -> None:
    config = create_quick_config(
        dataset_type=dataset_type,
        model_name=model_name,
        lr=lr,
        epochs=epochs,
        enable_visualization=enable_visualization,
        enable_tsne=enable_tsne,
        experiment_name=experiment_name
    )

    config.cuda_device = cuda_device
    config.enable_training = enable_training
    config.enable_testing = enable_testing
    config.channels = channels
    config.window_size = window_size
    config._setup_paths()

    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()


def myTask(lr: float, epoch_nums: int, datasetType: int,
           cuda: str = 'cuda:0',
           net: str = 'FusAtNet',
           visualization: bool = True,
           tsne: bool = False) -> None:
    config = ExperimentConfig(
        dataset_type=datasetType,
        model_name=net,
        learning_rate=lr,
        epochs=epoch_nums,
        cuda_device=cuda,
        enable_visualization=visualization,
        enable_tsne=tsne,
        enable_training=True,
        enable_testing=True,
        experiment_name=f"legacy_task_{datasetType}"
    )

    runner = ExperimentRunner(config)
    runner.run_experiment()
