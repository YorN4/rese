import os
import shutil
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datamodule import MazdaCAEDataModule
from lit_model_utils import LitModel, build_criterion, build_model
from mlflow.tracking import MlflowClient
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from common.environments import DATASET_PATH, MLFLOW_TRACKING_URI, PROJECT_ROOT
from common.utils import create_experiment_id_and_path, seed_everything


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_ROOT, "src", "surrogate_modeling", "hydra"),
    config_name="config",
)
def main(cfg: DictConfig):
    # 現在の作業ディレクトリを取得し、実験パスとして設定
    experiment_path = os.getcwd()

    # 実験IDを辞書に追加し、設定を更新
    cfg = OmegaConf.create(dict({"e_id": os.path.split(experiment_path)[-1]}, **cfg))

    # ランダムシードを設定
    seed_everything(seed=cfg.settings.seed)

    # 実験IDとパスを作成
    create_experiment_id_and_path(experiment_path=experiment_path)

    # データモジュールの初期化 (データセットのロードと分割)
    dm = MazdaCAEDataModule(
        dataset_path=os.path.join(DATASET_PATH, cfg.dataset.data_path),
        num_control=cfg.dataset.num_control,
        num_pred=cfg.dataset.num_pred,
        data_size=cfg.dataset.data_size,
        val_size=cfg.dataset.val_size,
        test_size=cfg.dataset.test_size,
        in_len=cfg.settings.in_len,
        out_len=cfg.settings.out_len,
        batch_size=cfg.settings.batch_size,
        use_decimate=cfg.settings.use_decimate,
        decimate_range=cfg.settings.decimate_range,
    )

    # モデルと損失関数を構築
    model = build_model(cfg.dataset, cfg.settings, cfg.model, cfg.model.name)
    criterion = build_criterion(cfg.settings.criterion)

    # LitModelクラスのインスタンスを作成
    lit_model = LitModel(
        model=model,
        model_name=cfg.model.name,
        dataset_name=cfg.dataset.name,
        criterion=criterion,
        lr=cfg.settings.lr,
        experiment_path=experiment_path,
        project_root=PROJECT_ROOT,
        in_len=cfg.settings.in_len,
        out_len=cfg.settings.out_len,
        columns=dm.columns,
        mean_list=dm.mean_list,
        std_list=dm.std_list,
    )

    # MLFlowのロガーを設定
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.settings.mlflow_exp_name,
        run_name=cfg.e_id,
        tracking_uri=MLFLOW_TRACKING_URI,
    )

    # Early Stoppingコールバックを設定 (val_lossの監視)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=50, verbose=False, mode="min")

    # モデルチェックポイントのコールバックを設定 (最良のモデルを保存)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=os.path.join(experiment_path, "weight"),
        filename="best",
    )

    # ハイパーパラメータをMLFlowにログ
    mlf_logger.log_hyperparams(cfg)

    # PyTorch Lightningのトレーナーを設定
    trainer = pl.Trainer(
        max_epochs=cfg.settings.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    try:
        # モデルを学習
        trainer.fit(model=lit_model, datamodule=dm)

        # モデルの精度検証
        trainer.test(model=lit_model, datamodule=dm, ckpt_path=checkpoint_callback.best_model_path)

        # 学習済みモデルを保存
        if cfg.settings.is_save_model:
            torch.save(lit_model.state_dict(), os.path.join(experiment_path, "model.pth"))

        # train.logファイルを削除
        os.remove(os.path.join(experiment_path, "train.log"))

        # 実験の成果物をMLFlowにアップロード
        mlf_logger.experiment.log_artifacts(
            local_dir=experiment_path,
            run_id=mlf_logger.run_id,
        )

        # 実験ディレクトリを削除 (MLFlowで生成物を管理するため)
        shutil.rmtree(experiment_path)

        return lit_model.get_evaluation_mean()

    except Exception as e:
        # エラーが発生した場合、MLFlowにエラーログを残す
        client = MlflowClient()
        client.set_tag(mlf_logger.run_id, "status", "failed")
        client.set_tag(mlf_logger.run_id, "error", str(e))

        # エラーが発生した場合、すべての評価指標に無限大の値を返す(Hydra-Optuna-Sweeper)
        return [float("inf") for _ in range(len(dm.columns["target_name"]) * 2)]


if __name__ == "__main__":
    main()
