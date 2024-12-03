# サロゲートモデル化のコード

## モデル概要図
![モデル概要図](https://github.com/naoto120424/mazda_cae/blob/2d177f5419f90ea731c8c29f21e6470ad5103ba8/img/nets.svg)


## 学習の実行
最初に `surrogate_modeling` ディレクトリに移動する。
```
cd src/surrogate_modeling
```

基本的には以下のコマンドで学習を実行することができる。
```
pdm run python train.py
```

しかし、学習においてさまざまなパラメータを変更することができるのでその管理について説明する。

### 学習のパラメータ
パラメータはHydraというライブラリを用いて管理している。`~/surrogate_modeling/hydra/config.yaml`(`train.py`の`@hydra.main`内の `config_path` と `config_name` によって決められる。) に従ってパラメータを設定している。

`config.yaml`
```yaml
defaults:
  - _self_
  - model: dot
  - settings: default
# 以下略
```

`- ` が先頭についた行は更にネストされたパラメータを持ち、そのディレクトリ内のyamlファイルを設定することができる。例えば、`model`の設定は`~/hydra/model/dot.yaml`によって行われれ、`settings`の設定は`~/hydra/settings/default.yaml`によって行われる。指定するyamlファイルによって異なるパラメータ構造も使用することができる。

<details>
<summary>dot.yamlとdefault.yamlの中身</summary>

`~/single/hydra/model/dot.yaml`
```yaml
name: dot
heads: 8
fc_dim: 1024
trunk_depth: 2
trunk_dim: 256
branch_depth: 2
branch_dim: 128
width: 32
dropout: 0.0
emb_dropout: 0.5
```

`~/single/hydra/settings/default.yaml`
```yaml
# 学習環境
is_debug: False # デバッグモードかどうか
use_decimate : True # データの間引きをするか
decimate_range: 1.0 # データの間引き間隔。デフォルトは0.1秒刻みデータを1.0秒間隔にする。
in_len: 10
out_len: 1
seed: 42 # シード値
epochs: 500 # エポック数
batch_size: 1024 # バッチサイズ
lr: 0.001 # 学習率
criterion: mse # 損失関数
is_save_model: False # モデルを保存するかどうか(False推奨)

# mlflow
mlflow_exp_name: pytorch_lightning_test # mlflowのexperiment名
```

</details>

本リポジトリでは、`model`の設定はモデルの構造に関するパラメータを、`settings`の設定は学習に関するパラメータを設定している。`model` のデフォルトとしては`dot`を、`settings`のデフォルトとしては`default`を設定している。

### パラメータの変更
初期パラメータからハードコーディングせずにインタラクティブにパラメータを変更することができる。例えば、`model`のパラメータを変更する場合は以下のようにコマンドを実行する。以下のようにすることで、`model`のパラメータを`dot`から`bt`に変更することができる。

```
pdm run python train.py model=bt
```

また、`model` の中のパラメータを変更することもできる。例えば、`model`の`dim`と `settings` の `mlflow_exp_name` のパラメータを変更する場合は以下のようにコマンドを実行する。以下のようにすることで、`model`の`width`のパラメータを`32`から`64`に、`settings`の`mlflow_exp_name`のパラメータを`pytorch_lightning_test`から`dol_test`に変更することができる。何個でもパラメータを変更することができる。

```
pdm run python train.py model.width=64 settings.mlflow_exp_name=dot_test
```

パラメータの詳細は `~/surrogate_modeling/hydra/` ディレクトリ内のyamlファイルを参照してください。

## (上級) Optunaを用いたパラメータ探索の実行

<details>


学習を実行するときに、単一の実行ではなく、複数のパラメータを変更して学習を行いたい場合がある。そのような場合には、以下のようにコマンドを実行することで、複数のパラメータを変更して学習を行うことができる。以下のように実行することで、`model`のパラメータが`dot`と`dol`の両方を実行することができる。これは `model` と `settings` の中のパラメータについても同様である。

```
pdm run python train.py -m 'model=choice(dot, dol)'
```

コマンドラインから実行できるが、[`~/surrogate_modeling/hydra/config.yaml`](./hydra/config.yaml) 内の `hydra.sweeper.params` でデフォルトで探索する範囲や指定のパラメータを設定できる。また、組み合わせの数が多く全探索を行うのが難しい場合は、試行数を以下のように `hydra.sweeper.n_trials` 指定することで制限できる。これもコマンドラインから実行できるが、[`~/single/hydra/config.yaml`](./hydra/config.yaml) 内の `hydra.sweeper.n_trials` でデフォルトで試行数を設定できる。


```
pdm run python train.py -m 'model=choice(dot, dol)' hydra.sweeper.n_trials=10
```


### 評価値の監視
パラメータ探索をする際に、監視対象となる評価値を指定することができる。`@hydra.main` の関数の返り値が評価値となる。例えば、以下のようにすることで、`val_loss`を監視評価値として指定することができる。

`~/train.py`
```python
@hydra.main(config_path='~/hydra', config_name='config')
def main(cfg: DictConfig) -> float:
    # 以下略
    return val_loss
```

そして、この評価値が小さくなるように改善するか、大きくなるように改善するかを指定することができる。[`~/single/hydra/config.yaml`](./hydra/config.yaml) 内の `hydra.sweeper.direction` で指定することができる。ここでは、単一の評価値も指定でき、複数の評価値を指定することもできる。改善する方向は `minimize` と `maximize` が指定できる。

`~/hydra/config.yaml`
```yaml
# 以上略
hydra:
  sweeper:
    params: null
    direction: minimize
# 以下略
```

パラメータの探索のアルゴリズムはOptunaというライブラリを用いている。ランダム探索やベイズ最適化などのアルゴリズムを指定することができる。[`~/single/hydra/config.yaml`](./hydra/config.yaml) 内の `hydra/sweeper/sampler` で指定することができる。詳細はOptunaのドキュメントを参照してください。

### 学習経過の保存
optunaの機能を使って最適化している過程が保存される。保存先は `~/single/hydra_experiment/.sqlite/` 内に保存される。このデータベースファイルを用いて、最適化の過程を確認することができる。実行前に `~/single/hydra_experiment/.sqlite/` のディレクトリがないとエラーが発生するので、事前にディレクトリを作成しておくこと。

<注意> strageパラメータを削除すると先ほどのディレクトリは必要ないです。

</details>

## 結果の確認
学習の過程で作成されたログや推論結果はmlflowのArtifactsから確認することができる。以下のようなディレクトリ構造になっている。
```
|-- .hydra
|   |-- config.yaml <- パラメータの設定ファイル
|   |-- hydra.yaml <- hydraの設定ファイル(気にしなくて良い)
|   `-- overrides.yaml <- パラメータの変更履歴
|-- csv
|   `-- caseXXXX.csv <- 各ケースの出力結果
|-- figure
|   |-- caseXXXX <- テストデータの各ケース毎の結果のディレクトリ
|   |   |-- YYYY.png <- 各ケースの結果の画像
|   |   `-- loss.txt <- 各ケースの各評価指標における誤差の情報
|-- loss
|   |-- loss_YYYY_ADE <- 各テストケースのADEにおける誤差を集約したファイル
|   `-- loss_YYYY_FDE <- 各テストケースのFDEにおける誤差を集約したファイル
`-- (settings.is_save_modelがTrueの時)lit_model.pth <- 学習済みモデル(モデル構造や重みなどが全て保存されている)
```

### 参考
HydraやOptunaについての詳しい使い方はドキュメントや参考サイトを参照してください。

#### Hydra
- [公式ドキュメント](https://hydra.cc/docs/intro/)
- [Hydraを用いたPython・機械学習のパラメータ管理方法](https://zenn.dev/kwashizzz/articles/ml-hydra-param)
- [5分でできるHydraによるパラメーター管理](https://qiita.com/Isaka-code/items/3a0671306629756895a6)
- etc...

#### Optuna
- [公式ドキュメント](https://optuna.readthedocs.io/en/stable/)
- [Hydraの公式ドキュメント内のOptuna Sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/#configuring-through-commandline-override)
- [【機械学習】Optunaを使って効率よくハイパーパラメータを調整しよう](https://zenn.dev/robes/articles/d53ff6d665650f)
- [(書籍)Optunaによるブラックボックス最適化](https://www.amazon.co.jp/Optuna%E3%81%AB%E3%82%88%E3%82%8B%E3%83%96%E3%83%A9%E3%83%83%E3%82%AF%E3%83%9C%E3%83%83%E3%82%AF%E3%82%B9%E6%9C%80%E9%81%A9%E5%8C%96-%E4%BD%90%E9%87%8E-%E6%AD%A3%E5%A4%AA%E9%83%8E/dp/4274230104/ref=sr_1_1?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&crid=821F8CM9LMR9&keywords=optuna&qid=1707806023&sprefix=optuna%2Caps%2C196&sr=8-1) ← かなりおすすめ
- etc...
