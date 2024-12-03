# NVIDIAのCUDAイメージをベースとして使用
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# 環境変数を設定してインタラクティブモードを回避
ENV DEBIAN_FRONTEND=noninteractive

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libreadline-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11.4をソースからビルドしてインストール
RUN wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz \
    && tar xzf Python-3.11.4.tgz \
    && cd Python-3.11.4 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.11.4 Python-3.11.4.tgz

# Pythonのバージョンを3.11.4に設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1
RUN update-alternatives --set python3 /usr/local/bin/python3.11

# pipをアップグレード
RUN python3 -m ensurepip
RUN python3 -m pip install --upgrade pip

# 非rootユーザーを作成
RUN useradd -m -s /bin/bash dockeruser
USER dockeruser
WORKDIR /home/dockeruser/code

# pdmをインストール
RUN pip install --user pdm

# パスを通す
ENV PATH="/home/dockeruser/.local/bin:${PATH}"

# コンテナ起動時に実行するコマンドを設定
CMD [ "/bin/bash" ]
