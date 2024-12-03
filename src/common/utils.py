import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    ランダムシードを設定し、再現性を確保するための関数。

    Args:
        seed (int): 再現性を担保するために設定するシード値。デフォルトは42。

    この関数は、Pythonのrandomモジュール、NumPy、およびPyTorchのシードを設定します。
    また、PyTorchでCUDAを使用する際に、再現性のある動作を保証するための設定を行います。

    Example:
        seed_everything(1234)
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")


def create_experiment_id_and_path(experiment_path: str) -> None:
    """
    実験結果を保存するためのディレクトリを作成する関数。

    Args:
        experiment_path (str): 実験用のベースディレクトリのパス。

    Raises:
        ValueError: 指定された `experiment_path` が存在しない場合に発生。

    この関数は、損失、モデルの重み、図を保存するためのサブディレクトリを作成します。
    ベースディレクトリが既に存在することを前提としています。

    Example:
        create_experiment_id_and_path("/path/to/experiment")
    """

    # 実験ディレクトリが存在することを確認
    if not os.path.exists(experiment_path):
        raise ValueError("experiment_pathに存在しないディレクトリが指定されました。")

    # 損失、重み、図のサブディレクトリを作成
    os.mkdir(os.path.join(experiment_path, "csv"))
    os.mkdir(os.path.join(experiment_path, "loss"))
    os.mkdir(os.path.join(experiment_path, "weight"))
    os.mkdir(os.path.join(experiment_path, "figure"))
