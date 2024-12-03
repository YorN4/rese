import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from model.BaseTransformer import BaseTransformer
from model.DeepOLSTM import DeepOLSTM
from model.DeepONet import DeepONet
from model.DeepOTransformer import DeepOTransformer
from model.LSTM import LSTM
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error
from torch import nn


def build_model(
    dataset_cfg: DictConfig,
    settings_cfg: DictConfig,
    model_cfg: DictConfig,
    model_name: str,
) -> nn.Module:
    """
    指定されたモデル設定に基づいてニューラルネットワークモデルを構築し、返却します。

    この関数は、データセット構成、設定構成、およびモデル構成を受け取り、
    指定されたモデル名に従って適切なモデルクラスをインスタンス化します。
    モデルの種類としては、以下をサポートしています:

    - BaseTransformer ("bt")
    - LSTM ("lstm")
    - DeepONet ("don")
    - DeepOLSTM ("dol")
    - DeepOTransformer ("dot")

    各モデルには、対応するパラメータが設定から引き継がれます。
    サポートされていないモデル名が指定された場合には、`ValueError` がスローされます。

    Args:
        dataset_cfg (DictConfig): データセットに関する設定を含む辞書。
        settings_cfg (DictConfig): モデルの一般設定を含む辞書。
        model_cfg (DictConfig): モデルの具体的な構成を含む辞書。

    Returns:
        nn.Module: 指定された構成に基づいてインスタンス化されたニューラルネットワークモデル。

    Raises:
        ValueError: 無効なモデル名が指定された場合に発生します。
    """
    if model_name == "bt":
        return BaseTransformer(
            num_control=dataset_cfg.num_control,
            num_pred=dataset_cfg.num_pred,
            num_all_features=dataset_cfg.num_all_features,
            out_len=settings_cfg.out_len,
            dim=model_cfg.dim,
            depth=model_cfg.depth,
            heads=model_cfg.heads,
            fc_dim=model_cfg.fc_dim,
            dropout=model_cfg.dropout,
            emb_dropout=model_cfg.emb_dropout,
        )

    elif model_name == "lstm":
        return LSTM(
            num_control=dataset_cfg.num_control,
            num_pred=dataset_cfg.num_pred,
            num_all_features=dataset_cfg.num_all_features,
            out_len=settings_cfg.out_len,
            dim=model_cfg.dim,
            depth=model_cfg.depth,
            dropout=model_cfg.dropout,
        )

    elif model_name == "don":
        return DeepONet(
            num_control=dataset_cfg.num_control,
            num_pred=dataset_cfg.num_pred,
            num_all_features=dataset_cfg.num_all_features,
            out_len=settings_cfg.out_len,
            trunk_depth=model_cfg.trunk_depth,
            trunk_dim=model_cfg.trunk_dim,
            branch_depth=model_cfg.branch_depth,
            branch_dim=model_cfg.branch_dim,
            width=model_cfg.width,
        )

    elif model_name == "dol":
        return DeepOLSTM(
            num_control=dataset_cfg.num_control,
            num_pred=dataset_cfg.num_pred,
            num_all_features=dataset_cfg.num_all_features,
            out_len=settings_cfg.out_len,
            dropout=model_cfg.dropout,
            trunk_depth=model_cfg.trunk_depth,
            trunk_dim=model_cfg.trunk_dim,
            branch_depth=model_cfg.branch_depth,
            branch_dim=model_cfg.branch_dim,
            width=model_cfg.width,
        )

    elif model_name == "dot":
        return DeepOTransformer(
            num_control=dataset_cfg.num_control,
            num_pred=dataset_cfg.num_pred,
            num_all_features=dataset_cfg.num_all_features,
            out_len=settings_cfg.out_len,
            heads=model_cfg.heads,
            fc_dim=model_cfg.fc_dim,
            trunk_depth=model_cfg.trunk_depth,
            trunk_dim=model_cfg.trunk_dim,
            branch_depth=model_cfg.branch_depth,
            branch_dim=model_cfg.branch_dim,
            width=model_cfg.width,
            dropout=model_cfg.dropout,
            emb_dropout=model_cfg.emb_dropout,
        )

    else:
        raise ValueError("指定されたモデル名が無効です")


def build_criterion(criterion_name: str):
    """
    指定された損失関数名に基づいて、PyTorchの損失関数を構築して返します。

    この関数は、指定された文字列に応じて、対応するPyTorchの損失関数オブジェクトを返します。
    サポートされている損失関数は以下の通りです:

    - "mse": 平均二乗誤差損失 (`nn.MSELoss`)
    - "l1": 平均絶対誤差損失 (`nn.L1Loss`)
    - "smoothl1": 平滑化L1損失 (`nn.SmoothL1Loss`)

    指定された損失関数名がサポートされていない場合は、`ValueError` がスローされます。

    Args:
        criterion_name (str): 使用する損失関数の名前。

    Returns:
        nn.Module: 対応する損失関数オブジェクト。

    Raises:
        ValueError: 無効な損失関数名が指定された場合に発生します。
    """

    if criterion_name == "mse":
        return nn.MSELoss()
    elif criterion_name == "l1":
        return nn.L1Loss()
    elif criterion_name == "smoothl1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError("指定された損失関数名が無効です")


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model,
        model_name,
        dataset_name,
        criterion,
        lr,
        experiment_path,
        project_root,
        in_len,
        out_len,
        columns,
        mean_list,
        std_list,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.criterion = criterion
        self.lr = lr
        self.experiment_path = experiment_path
        self.project_root = project_root
        self.in_len = in_len
        self.out_len = out_len
        self.columns = columns
        self.mean_list = mean_list
        self.std_list = std_list

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def extract_batch_data(self, batch):
        """
        バッチからデータを抽出する。
        """

        x_past = batch["x_past"]
        x_control = batch["x_control"]
        ground_truth = batch["ground_truth"]
        case_name = batch["case_name"]

        return x_past, x_control, ground_truth, case_name

    def training_step(self, batch, _):
        x_past, x_control, ground_truth, _ = self.extract_batch_data(batch)
        out = self.model(x_past, x_control)
        loss = self.criterion(out, ground_truth)
        if loss is None:
            return None

        batch_size = len(x_past)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, _):
        x_past, x_control, ground_truth, _ = self.extract_batch_data(batch)
        out = self.model(x_past, x_control)
        loss = self.criterion(out, ground_truth)
        if loss is None:
            return None

        batch_size = len(x_past)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss

    def on_test_start(self):
        """テスト開始時に呼び出され、予測モデルの各評価指標を格納するリストを初期化する。"""
        num_pred = self.columns["pred_name"]
        self.predict_time_list = []
        self.loss_list_dict = {
            "ADE": [[] for _ in range(len(num_pred))],
            "FDE": [[] for _ in range(len(num_pred))],
        }

    def calculate_evaluation(
        self,
        ground_truth: np.ndarray,
        pred: np.ndarray,
        case_name: str,
        save_path: str,
        columns: dict,
    ) -> dict:
        """
        評価指標（ADE, FDE）を計算し、結果をファイルに保存する。

        Args:
            ground_truth (np.ndarray): 正解データの配列。
            pred (np.ndarray): 予測データの配列。
            case_name (str): ケースの名前。
            save_path (str): 結果を保存するディレクトリのパス。
            columns (dict): 特徴量名や単位を含む辞書。

        Returns:
            dict: ADEとFDEを含む辞書。
        """

        ade_list = []
        fde_list = []

        pred_name = columns["pred_name"]
        pred_unit = columns["pred_unit"]

        for i in range(ground_truth.shape[1]):
            ade_list.append(mean_absolute_error(ground_truth[:, i], pred[:, i]))
            fde_list.append(abs(ground_truth[-1, i] - pred[-1, i]))

        with open(os.path.join(save_path, "loss.txt"), "w") as f:
            f.write(f"{case_name}\n")
            f.write("---------------------------\n")
            f.write("ADE\n")
            for i in range(len(pred_name)):
                f.write(f"{pred_name[i]}[{pred_unit[i]}]: {ade_list[i]}\n")
            f.write("---------------------------\n")
            f.write("FDE\n")
            for i in range(len(pred_name)):
                f.write(f"{pred_name[i]}[{pred_unit[i]}]: {fde_list[i]}\n")
            f.write("\n評価指標\n")
            f.write("1. ADE: 全時刻の平均絶対誤差\n")
            f.write("2. FDE: 最後の時刻の絶対誤差\n")

        return {
            "ADE": ade_list,
            "FDE": fde_list,
        }

    def test_step(self, batch, _):
        x_past_data, x_control_data, ground_truth, case_name = self.extract_batch_data(batch)

        x_past_data = x_past_data.detach().cpu().numpy().copy()
        x_control_data = x_control_data.detach().cpu().numpy().copy()
        ground_truth = ground_truth.detach().cpu().numpy().copy()[0]
        case_name = case_name[0]

        save_path = os.path.join(self.experiment_path, "figure", case_name)
        os.mkdir(save_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 予測時間計測開始
        start_time = time.time()

        # ここで予測する
        for i in range(0, ground_truth.shape[0] - 1, self.out_len):
            x_past = torch.from_numpy(x_past_data[:, -self.in_len :].astype(np.float32)).clone().to(device)
            x_control = torch.from_numpy(x_control_data[:, i].astype(np.float32)).clone().to(device)

            y_pred = self.model(x_past, x_control)
            y_pred = y_pred.detach().cpu().numpy().copy()

            next_data = np.concatenate([x_control_data[:, i + 1], y_pred], axis=-1)
            x_past_data = np.concatenate([x_past_data, next_data], axis=1)

        # 予測時間計測終了
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.predict_time_list.append(elapsed_time)

        # スケーリングを戻す
        for i in range(x_past_data.shape[-1]):
            x_past_data[:, :, i] = x_past_data[:, :, i] * self.std_list[i] + self.mean_list[i]

        x_past_data = x_past_data[0, (self.in_len - 1) : (self.in_len - 1) + ground_truth.shape[0], -ground_truth.shape[1] :]

        # 評価と評価結果の保存
        loss_dict = self.calculate_evaluation(ground_truth, x_past_data, case_name, save_path, self.columns)
        self.save_evaluation_result(loss_dict, ground_truth, x_past_data, case_name, save_path, self.columns)

    def save_evaluation_result(self, loss_dict, ground_truth, y_pred, case_name, figure_path, columns):
        """評価結果を保存し、グラフを生成して保存する。"""

        pred_name = columns["pred_name"]
        pred_unit = columns["pred_unit"]

        for i in range(len(pred_name)):  # 各予測変数ごとに損失リストを更新
            self.loss_list_dict["ADE"][i].append([case_name, loss_dict["ADE"][i]])
            self.loss_list_dict["FDE"][i].append([case_name, loss_dict["FDE"][i]])

        for i in range(len(pred_name)):
            self.save_fig(
                ground_truth[:, i],
                y_pred[:, i],
                os.path.join(figure_path, f"{pred_name[i]}.png"),
                title=f"{case_name}",
                xlabel="Time",
                ylabel=f"{pred_name[i]}[{pred_unit[i]}]",
                label1="ground_truth",
                label2="y_pred",
            )

        pd_out_csv = pd.DataFrame(y_pred, columns=pred_name)
        pd_out_csv.to_csv(os.path.join(self.experiment_path, "csv", f"{case_name}.csv"))

        # いろんな実験の結果をまとめてグラフ化するために、予測結果を project_root/predict_csvs にCSVで保存しておく
        predict_csv_path = os.path.join(
            self.project_root,
            "predict_csvs",
            self.dataset_name,
            self.model_name,
            f"inlen_{str(self.in_len).zfill(2)}",
        )
        os.makedirs(predict_csv_path, exist_ok=True)
        pd_out_csv.to_csv(os.path.join(predict_csv_path, f"{case_name}.csv"))

    def save_fig(self, ground_truth, y_pred, save_path, title, xlabel, ylabel, label1, label2, ymax=None):
        """
        2つのデータセットをプロットし、結果を画像ファイルとして保存する。

        Args:
            ground_truth (np.ndarray): グラフにプロットする1つ目のデータ。
            y_pred (np.ndarray): グラフにプロットする2つ目のデータ。
            save_path (str): 保存先のファイルパス。
            title (str): グラフのタイトル。
            xlabel (str): x軸のラベル。
            ylabel (str): y軸のラベル。
            label1 (str): 1つ目のデータの凡例ラベル。
            label2 (str): 2つ目のデータの凡例ラベル。
            ymax (float, optional): y軸の最大値。指定しない場合はデフォルトの範囲を使用する。
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ground_truth, label=label1)
        sns.lineplot(data=y_pred, label=label2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ymax is not None:
            plt.ylim(0, ymax)
        plt.legend(loc="best")
        plt.savefig(save_path)
        plt.close()

    def on_test_end(self):
        """テスト終了時に損失リストの処理と不要な重みファイルの削除を行う。"""
        pred_name = self.columns["pred_name"]

        self.mean_dict = {index_name: {} for index_name in ["ADE", "FDE"]}
        for index_name in ["ADE", "FDE"]:
            for i, pred in enumerate(pred_name):
                self.mean_dict[index_name][pred] = self.process_loss_list(
                    self.loss_list_dict[index_name][i],  # i 番目の予測変数の損失リストを処理
                    index_name,
                    pred,
                )

        predict_time_list = np.array(self.predict_time_list)
        predict_time_metrics = {
            "predict_time_max": np.max(predict_time_list),
            "predict_time_min": np.min(predict_time_list),
            "predict_time_mean": np.mean(predict_time_list),
            "predict_time_std": np.std(predict_time_list),
            "predict_time_median": np.median(predict_time_list),
        }
        self.logger.log_metrics(predict_time_metrics)

        # 不要な重みファイルを削除
        weight_path = os.path.join(self.experiment_path, "weight")
        if os.path.exists(weight_path):
            shutil.rmtree(weight_path)

    def process_loss_list(self, loss_list, index_name, pred):
        """
        損失リストを処理し、CSVファイルに保存、統計量を計算し、mlflowに記録する。

        Args:
            loss_list (list): 評価指標の損失値を含むリスト。
            index_name (str): 評価指標の名前（ADE、FDEなど）。
            pred (str): 予測変数の名前。

        Returns:
            float: 損失リストの平均値。
        """
        # リストをDataFrameに変換し、ソート
        df_loss_list = pd.DataFrame(loss_list, columns=["id", "loss"]).set_index("id")
        df_sorted_loss_list = df_loss_list.sort_values(by="loss")

        # CSVに保存
        csv_path = os.path.join(os.path.join(self.experiment_path, "loss"), f"loss_{pred}_{index_name}.csv")
        df_sorted_loss_list.to_csv(csv_path)

        # mlflowに統計量を記録
        metrics = {
            f"{pred}_{index_name}_max": df_sorted_loss_list["loss"].max(),
            f"{pred}_{index_name}_min": df_sorted_loss_list["loss"].min(),
            f"{pred}_{index_name}_mean": df_sorted_loss_list["loss"].mean(),
            f"{pred}_{index_name}_std": df_sorted_loss_list["loss"].std(),
            f"{pred}_{index_name}_median": df_sorted_loss_list["loss"].median(),
        }
        self.logger.log_metrics(metrics)

        return metrics[f"{pred}_{index_name}_mean"]

    def get_evaluation_mean(self):
        """
        評価指標の統計量を返す。

        Returns:
            list: "kW"を含むADEおよびFDEの平均値を含むリスト。
        """

        return [value for metric in ["ADE", "FDE"] for key, value in self.mean_dict[metric].items() if "kW" in key]
