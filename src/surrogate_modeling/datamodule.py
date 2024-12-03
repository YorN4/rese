import glob
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from einops import rearrange
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MazdaCAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        num_control: str,
        num_pred: int,
        data_size: int,
        val_size: int,
        test_size: int,
        in_len: int,
        out_len: int,
        batch_size: int,
        use_decimate: bool = True,
        decimate_range: float = 1.0,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.num_control = num_control
        self.num_pred = num_pred
        self.data_size = data_size
        self.val_size = val_size
        self.test_size = test_size
        self.in_len = in_len
        self.out_len = out_len
        self.batch_size = batch_size
        self.use_decimate = use_decimate
        self.decimate_range = decimate_range

        self.data, self.columns = load_data(
            self.dataset_path,
            self.num_control,
            self.num_pred,
            self.in_len,
            self.out_len,
            self.use_decimate,
            self.decimate_range,
        )
        train_index_list, self.test_index_list = train_test_split(
            np.arange(self.data_size),
            test_size=self.test_size,
        )
        self.train_index_list, self.val_index_list = train_test_split(
            train_index_list,
            test_size=self.val_size,
        )
        self.mean_list, self.std_list = find_mean_std(self.data, self.train_index_list)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = create_dataset(
                self.data,
                self.train_index_list,
                mode="train",
                mean_list=self.mean_list,
                std_list=self.std_list,
            )

            self.val_dataset = create_dataset(
                self.data,
                self.val_index_list,
                mode="val",
                mean_list=self.mean_list,
                std_list=self.std_list,
            )

        elif stage == "test":
            self.test_dataset = create_dataset(
                self.data,
                self.test_index_list,
                mode="test",
                mean_list=self.mean_list,
                std_list=self.std_list,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=2)


def decimate(data, dt=1.0):
    """
    時系列データを線形補間し、ダウンサンプリングする。

    Args:
        data (np.ndarray): 2次元配列で、最初の列が時刻、残りの列が対応するデータ値を表す。
        dt (float, optional): ダウンサンプリングのための時間間隔。デフォルトは1。

    Returns:
        np.ndarray: 指定された時間間隔で補間されたダウンサンプリングデータを含む2次元配列。各行はその時間間隔での補間値を表す。
    """

    new_data = []
    pick_time = dt

    for i in range(1, len(data)):
        if data[i, 0] > pick_time:
            x = [data[i - 1, 0], data[i, 0]]
            y = [data[i - 1, :], data[i, :]]

            a, b = np.polyfit(x, y, 1)
            interpolated_value = a * pick_time + b

            new_data.append(interpolated_value)
            pick_time += dt

    new_data = np.array(new_data)
    new_data = new_data.reshape(new_data.shape[0], -1)

    return new_data


def preprocess_data(data, num_control, use_decimate, decimate_range):
    """
    入力データを前処理し、制御データやターゲットデータを抽出する。

    Args:
        x_past (np.ndarray): 時刻データと対応する値を含む2次元配列。
        num_control (int): 制御特徴量として使用する列の数。
        use_decimate (bool): データのダウンサンプリングを行うかどうかを指定する。
        decimate_range (float): データのダウンサンプリングの間隔を設定する。

    Returns:
        tuple: 前処理されたデータのタプル。
            - x_past: 元のデータ（時刻列を除く）。
            - x_control: 制御データ。
            - ground_truth: ターゲットデータ。
    """

    if use_decimate:
        x_past = decimate(data, decimate_range)[:, 1:]
    else:
        x_past = data[:, 1:]

    x_control = x_past[:, :num_control]
    ground_truth = x_past[:, num_control:]

    return x_past, x_control, ground_truth


def generate_sequences(x_past, x_control, ground_truth, in_len, out_len):
    """
    入力データからシーケンスを生成し、モデルの学習や予測に使用できる形式に整形する。

    Args:
        x_past (np.ndarray): シーケンスを生成するための2次元配列データ。
        x_control (np.ndarray): 制御シーケンスを生成するためのデータ。
        ground_truth (np.ndarray): ターゲットシーケンスを生成するためのデータ。
        in_len (int): モデルに入力するシーケンスの長さ。
        out_len (int): モデルが予測するシーケンスの長さ。

    Returns:
        tuple: 生成されたシーケンスのタプル。
            - x_past_list: モデル入力用のシーケンス。
            - x_control_list: 制御値のシーケンス。
            - gt_list: ターゲットシーケンス。
    """

    x_past_list, x_control_list, ground_truth_list = [], [], []

    for t in range(x_past.shape[0] - in_len - out_len + 1):
        x_past_list.append(x_past[t : t + in_len])
        x_control_list.append(x_control[t + in_len : t + in_len + out_len])
        ground_truth_list.append(ground_truth[t + in_len : t + in_len + out_len])

    return np.array(x_past_list), np.array(x_control_list), np.array(ground_truth_list)


def get_column_names(
    data_path: str,
    num_pred: int,
) -> dict:
    """
    データセットから特徴量と単位の名前を抽出し、列名を整理する。

    Args:
        data_path (str): データが保存されているファイルのパス。
        num_pred (int): 予測する特徴量の数。

    Returns:
        columns (dict): 特徴量名と単位を含む辞書。以下のキーを含む:
            - "feature_name": すべての特徴量の名前。
            - "feature_unit": すべての特徴量の単位。
            - "pred_name": 予測する特徴量の名前。
            - "pred_unit": 予測する特徴量の単位。
    """

    columns = {
        "feature_name": [],
        "feature_unit": [],
        "pred_name": [],
        "pred_unit": [],
    }

    columns["feature_name"] = [col.split(".")[0] for col in pd.read_csv(data_path, skiprows=0, dtype=str).columns]
    columns["feature_unit"] = [col.split(".")[0] for col in pd.read_csv(data_path, skiprows=1, dtype=str).columns]

    columns["pred_name"] = columns["feature_name"][-num_pred:]
    columns["pred_unit"] = columns["feature_unit"][-num_pred:]

    return columns


def load_data(
    dataset_path: str,
    num_control: int,
    num_pred: int,
    in_len: int,
    out_len: int,
    use_decimate: bool = True,
    decimate_range: float = 1.0,
):
    """
    データセットを読み込み、前処理を行い、学習や評価に使用するためのデータを生成する。

    Args:
        dataset_path (str): データセットが保存されているディレクトリのパス。
        num_control (int): 制御値の数。
        num_pred (int): 予測する特徴量の数。
        in_len (int): モデルに入力するシーケンスの長さ。
        out_len (int): モデルが予測するシーケンスの長さ。
        use_decimate (bool, optional): データのダウンサンプリングを行うかどうかを指定する。デフォルトは True である。
        decimate_range (float, optional): データのダウンサンプリングの間隔を設定する。デフォルトは 1.0 である。

    Returns:
        data (dict): 前処理されたデータを格納した辞書。以下のキーを含む:
            - "x_past": モデルの入力データ。
            - "x_control": 予測に使用する制御データ。
            - "ground_truth": 正解データ。
            - "case_name": データセット内のケース番号。

        columns (dict): 各特徴量に関する情報を格納した辞書。以下のキーを含む:
            - "feature_name": 各特徴量の名前。
            - "feature_unit": 各特徴量の単位。
            - "pred_name": 予測する特徴量の名前。
            - "pred_unit": 予測する特徴量の単位。
    """

    dataset_list = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
    data = {
        "x_past": [],
        "x_control": [],
        "ground_truth": [],
        "case_name": [],
    }

    print("Data Loading")
    for case_name, dt in tqdm(enumerate(dataset_list)):
        csv_data = pd.read_csv(dt, skiprows=1).values
        x_past, x_control, ground_truth = preprocess_data(csv_data, num_control, use_decimate, decimate_range)

        if dt == dataset_list[0]:
            data_len = x_past.shape[0]

        if x_past.shape[0] == data_len:
            x_past_seq, x_control_seq, ground_truth_seq = generate_sequences(x_past, x_control, ground_truth, in_len, out_len)
            data["x_past"].append(x_past_seq)
            data["x_control"].append(x_control_seq)
            data["ground_truth"].append(ground_truth_seq)
            data["case_name"].append(f"case{str(case_name + 1).zfill(4)}")

    data["x_past"] = np.array(data["x_past"])
    data["x_control"] = np.array(data["x_control"])
    data["ground_truth"] = np.array(data["ground_truth"])

    columns = get_column_names(dataset_list[0], num_pred)

    return data, columns


def find_mean_std(
    data: dict,
    index_list: list,
):
    input_data = np.array([data["x_past"][index] for index in index_list])

    mean_list = input_data.mean(axis=(0, 1, 2)).tolist()
    std_list = input_data.std(axis=(0, 1, 2)).tolist()

    return mean_list, std_list


def create_dataset(
    data: dict,
    index_list: list,
    mode: str,
    mean_list: list,
    std_list: list,
) -> tuple:
    """
    データセットを作成し、標準化を行う。

    Args:
        data (dict): 入力データ、制御データ、ターゲットデータを含む辞書。
        index_list (list): データセットを構築するために使用するデータのインデックスリスト。
        mode (str): 処理モード。"train", "valid", "test" のいずれか。
        mean_list (list, optional): 特徴量の平均値リスト。テストや検証時に使用する。
        std_list (list, optional): 特徴量の標準偏差リスト。テストや検証時に使用する。

    Returns:
        tuple: MazdaDataset オブジェクト、平均値リスト、標準偏差リスト。
    """

    dataset = {}
    x_past_array, x_control_array, ground_truth_array, case_name = [], [], [], []

    for index in index_list:
        x_past_array.append(data["x_past"][index])
        x_control_array.append(data["x_control"][index])
        ground_truth_array.append(data["ground_truth"][index])
        case_name.append(data["case_name"][index])

    x_past_array = np.array(x_past_array)
    x_control_array = np.array(x_control_array)
    ground_truth_array = np.array(ground_truth_array)

    num_control = x_control_array.shape[3]

    print(f"{mode.capitalize()} Standardization")

    print("[input]")
    for i in tqdm(range(x_past_array.shape[3])):
        x_past_array[:, :, :, i] = (x_past_array[:, :, :, i] - mean_list[i]) / std_list[i]

    print("[spec]")
    for i in tqdm(range(x_control_array.shape[3])):
        x_control_array[:, :, :, i] = (x_control_array[:, :, :, i] - mean_list[i]) / std_list[i]

    if mode != "test":
        # trainとvalの場合はground_truthも標準化
        print("[gt]")
        for i in tqdm(range(ground_truth_array.shape[3])):
            ground_truth_array[:, :, :, i] = (ground_truth_array[:, :, :, i] - mean_list[i + num_control]) / std_list[i + num_control]

        dataset["x_past"] = rearrange(x_past_array, "a b c d -> (a b) c d")
        dataset["x_control"] = rearrange(x_control_array, "a b c d -> (a b) c d")
        dataset["ground_truth"] = rearrange(ground_truth_array, "a b c d -> (a b) c d")
        dataset["case_name"] = case_name * x_past_array.shape[1]
    else:
        # testの場合はground_truthは標準化しない(標準化前の値と比較するため)
        dataset["x_past"] = x_past_array[:, 0, :, :]
        dataset["x_control"] = x_control_array
        dataset["ground_truth"] = ground_truth_array[:, :, -1, :]
        dataset["case_name"] = case_name

    return MazdaDataset(dataset)


class MazdaDataset(Dataset):
    def __init__(self, data):
        super(MazdaDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data["x_past"])

    def __getitem__(self, index):
        return {
            "x_past": torch.Tensor(self.data["x_past"][index]),
            "x_control": torch.Tensor(self.data["x_control"][index]),
            "ground_truth": torch.Tensor(self.data["ground_truth"][index]),
            "case_name": self.data["case_name"][index],
        }
