# マツダとの共同研究リポジトリ
マツダ株式会社との共同研究の実験ソースコードを管理するリポジトリ。gitで管理することを想定した構成になっている。

## 使用するツールやライブラリ
各種ツールについては以下の通りである。それぞれの詳細な説明については公式ドキュメントや参考サイトを参照されたい。

名前 | 用途
--- | ---
VSCode | 統合IDE
Docker | 環境構築
docker-compose | 複数のコンテナを一括で管理するためのツール
pyenv | Pythonのバージョン管理ツール
pdm | Pythonのパッケージ管理ツール


## コードを開発するための環境構築
DevContainerというVisual Studio Code(VSCode)のコンテナ開発環境を利用して開発を行うことを想定している。<br>
jupyterlabを利用するには[学習実行方法](#学習実行方法)の手順を別のターミナルで起動する必要がある。
### 初回のみの設定
1. [ローカル端末] VSCodeをインストールする [link](https://code.visualstudio.com/)
    - リモートで開発する場合は「Remote - SSH」という拡張機能のインストールとsshの設定が必要になる。参考: [VSCodeを使ってサーバー環境にSSHリモート接続手順](https://blog.masuyoshi.com/vscode%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E3%82%B5%E3%83%BC%E3%83%90%E3%83%BC%E7%92%B0%E5%A2%83%E3%81%ABssh%E3%83%AA%E3%83%A2%E3%83%BC%E3%83%88%E6%8E%A5%E7%B6%9A%E6%89%8B%E9%A0%86/)
1. [ローカル端末] VSCodeの拡張機能「Remote - Containers」をインストールする [link](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
1. [ローカル端末] リモート端末に向けてssh接続
    ```
    ssh [アカウント名]@[リモート端末のIPアドレス]
    例 -> ssh naka@10.80.12.34
    ```
1. [リモート端末] Dockerをインストールする [link](https://www.docker.com/products/docker-desktop)<br>
    サーバー室の端末にはDockerはインストール済み。<br>
    研究室の自分のPCで環境構築する場合はインストールが必要。
1. [リモート端末] docker-composeをインストールする [link](https://docs.docker.com/compose/install/)
    以下コマンドでインストールできる
    ```
    sudo -i

    curl -L https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    exit

    docker-compose --version
    ```
1. [リモート端末] ターミナルを開き、GitHubからcloneできるようにする。[link](https://qiita.com/majka/items/aba924de8d4a92f75dbb)<br>
    - gitのユーザー設定<br>
        ```
        git config --global user.name [名前]
        git config --global user.email [メールアドレス]
        ```

    - ssh-keyの作成<br>
        ```
        ssh-keygen -t rsa -C [メールアドレス]
        ```
        ディレクトリに~/.ssh/id_rsaが作成される

    - GitHubに登録するためkeyをコピー<br>
        ```
        cat ~/.ssh/id_rsa.pub
        ```
        で出てきたkeyを丸ごとコピーする。

    - GitHubで登録<br>
        GitHubのページに移りアカウント設定の"SSH and GPG keys"を選択。<br>
        右上の "New SSH Key"をクリックし
            ```
            Title に自分のわかる名前 (このkeyを使用してssh接続するを指す名称を推奨します 例:lab 200)
            keyにクリップボードから公開鍵をペースト
            ```
        をした後に "Add SSH key" をクリックして完了。

    - 端末での確認

1. [リモート端末] 本リポジトリをクローンし、VSCodeで開く。
    ```
    git clone
    ```

1. 以下コマンドを実行し、`docker-compose.yaml`と`.devcontainer.json` 作成する。
    ```
    cp docker-compose.yaml.default docker-compose.yaml
    cp .devcontainer.json.default .devcontainer.json
    ```

1. `docker-compose.yaml` ファイルの以下部分を編集する
    ```
    # 以上略
        volumes:
          - .:/home/dockeruser/code/
          - [データセットの絶対パス]:/home/dockeruser/dataset/
          - [mlflowのトラッキングURIの絶対パス]:/home/dockeruser/mlflow_experiment/
    # 以下略
    ```

    パラメータ | 説明 | 例
    --- | --- | ---
    プロジェクトのルートディレクトリの絶対パス | 本リポジトリのルートディレクトリの絶対パス | `/Users/username/researchment/mazda`
    データセットの絶対パス | データセットを格納するディレクトリの絶対パス | `/Users/username/dataset/mazda`
    mlflowのトラッキングURIの絶対パス | mlflowで実験データを格納するためのディレクトリの絶対パス | `/Users/username/mlflow_experiment`

    <details>
    <summary>例</summary>

    ```
    # 以上略
        volumes:
          - .:/home/dockeruser/code/
          - /Users/username/dataset/mazda:/home/dockeruser/dataset/
          - /Users/username/mlflow_experiment:/home/dockeruser/mlflow_experiment/
    # 以下略
    ```

    </details>

    gpuを利用する場合は以下のように設定する
    - gpuデバイスが1つの場合
        ```
        # 以上略
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      count: 1
                      capabilities: [gpu]
        ```
    - gpuデバイスが複数の場合
        ```
        # 以上略
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      device_ids: ['0', '1']
                      capabilities: [gpu]
        ```

    詳しくは[ComposeにおけるGPUアクセスの有効化](https://matsuand.github.io/docs.docker.jp.onthefly/compose/gpu-support/)を参照されたい。

1. `.devcontainer.json` ファイルをの以下部分を編集する
    ```
            # 以上略
            "mounts": [
                {
                    "type":"bind",
                    "source":"[プロジェクトのルートディレクトリの絶対パス]",
                    "target":"/home/dockeruser/code"
                },
                {
                    "type":"bind",
                    "source":"[mlflowのトラッキングURIの絶対パス]",
                    "target":"/home/dockeruser/mlflow_experiment/"
                },
                {
                    "type":"bind",
                    "source":"[データセットの絶対パス]",
                    "target":"/home/dockeruser/dataset/"
                }
            ],
            # 以下略
    ```

    <details>
    <summary>例</summary>

    ```
            # 以上略
            "mounts": [
                {
                    "type":"bind",
                    "source":"/Users/username/researchment/mazda",
                    "target":"/home/dockeruser/code"
                },
                {
                    "type":"bind",
                    "source":"/Users/username/mlflow_experiment",
                    "target":"/home/dockeruser/mlflow_experiment/"
                },
                {
                    "type":"bind",
                    "source":"/Users/username/dataset/mazda",
                    "target":"/home/dockeruser/dataset/"
                }
            ],
            # 以下略
    ```
    </details>

1. VSCodeを開き、左下の「><」をクリックし、「Reopen in Container」を選択する。または、コマンドパレットを開き(cmd/ctl + shift + p)、「Remote-Containers: Reopen in Container」を選択する。

1. 既存のpdm.lockとpyproject.tomlからパッケージをインストールする
    ```
    pdm install
    ```

### 開発するたびの設定
1. VSCodeを開き、左下の「><」をクリックし、「Reopen in Container」を選択する。または、コマンドパレットを開き(ctrl + shift + p)、「Remote-Containers: Reopen in Container」を選択する。

### その他の必要なコマンド
- pythonライブラリのインストール方法
    ```
    pdm add [ライブラリ名]
    ```

    <details>
    <summary>例</summary>

    ```
    pdm add pandas
    ```

    </details>

## 学習実行方法
開発環境で構築したコンテナ内(DevConatainer内)では学習が不可能である。そのため、別のコンテナを作成し、学習を行う。

### 初回のみの設定
1. (リモートサーバーの場合)sshでリモートサーバーにログインする
    ```
    ssh [ユーザー名]@[ホスト名]
    ```

1. Dockerfileとdocker-compose.yamlからイメージとコンテナを作成し、起動する
    ```
    docker-compose up -d
    ```

1. tmux環境から作成したコンテナをデタッチモードで起動する
    - tmux環境を利用することで、ローカルPCから切断してもコンテナ内のプロセスが終了しないようにする。つまり学習が途中で途切れることがない。
    ```
    # tmuxを起動
    tmux

    # tmux 画面内で以下のコマンドを実行
    docker exec -it mazda-rc-container bash
    ```

1. 学習のためのデータセットを準備する。データセットの準備については、[データセットの準備](./DATA.md)を参照されたい。

1. コンテナ内で以下のコマンドを実行して学習を開始する。　学習については、[シングルタスク学習](./single/README.md)と[マルチタスク学習](./multi/README.md)の各README.mdを参照されたい。

1. tmux環境を抜ける。実行中でも抜けることができる。
    ```
    # tmux 画面内で以下のキーを押す
    ctrl + b, d
    ```

1. コンテナを停止する(**学習が止まるので注意！**)
    - 停止していない場合、メインメモリを大量に消費することがあるので、学習が終わると停止することを推奨する。
    ```
    docker-compose stop
    ```
### 2回目以降
1. (リモートサーバーの場合)sshでリモートサーバーにログインする
    ```
    ssh [ユーザー名]@[ホスト名]
    ```

1. コンテナを再起動する
    ```
    docker-compose start
    ```

1. tmux環境から作成したコンテナをデタッチモードで起動する
    - tmuxをまだ起動していない場合
        ```
        tmux

        # tmux 画面内で以下のコマンドを実行
        docker exec -it mazda-dl-container bash
        ```

    - tmuxを起動している場合
        ```
        tmux a
        ```

### 各種サービスのアクセス方法
- jupyterlab
    - リモートで実行している場合
        ```
        http://<リモートのipアドレス>:18888
        ```
    - ローカルで実行している場合
        ```
        http://localhost:18888
        ```
- mlflow
    - ブラウザで以下のURLにアクセスする
        ```
        http://<リモートのipアドレス>:5555
        ```
    - ローカルで実行している場合
        ```
        http://localhost:5555
        ```

<!-- ## 次のステップ -->
<!-- - [データセットの準備](./DATA.md) -->
