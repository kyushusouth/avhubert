# AVHuBERT の使い方

このガイドでは、AVHuBERT モデルを使用するためのセットアップ手順を説明します。

## 1. Python の設定

Python のバージョンは 3.10 以下を推奨します。3.11 系を使用すると、他のライブラリとの依存関係に起因するエラーが発生し、特に後述する fairseq から PyTorch への移行が難しくなる可能性があります。仮想環境はお好みの方法で設定してください。

## 2. fairseq のインストール

fairseq というライブラリをインストールする必要があります。以下の URL から fairseq のリポジトリにアクセスし、インストール手順に従ってください。

[fairseq GitHub リポジトリ](https://github.com/facebookresearch/fairseq)

## 3. 事前学習済みチェックポイントのダウンロード

事前学習済みのチェックポイントは公式のリポジトリからダウンロード可能です。以下のリンクからアクセスしてください。

[AV-HuBERT GitHub リポジトリ](https://github.com/facebookresearch/av_hubert)

本リポジトリでは、「LRS3 + VoxCeleb2 (En)」で学習され、「No finetuning」の状態である「AV-HuBERT Base」と「AV-HuBERT Large」を利用するプログラムを実装しています。それ以外については非対応です。

## 4. モデルについて

AVHuBERT のモデルは`avhubert.py`にまとめられています。本家のリポジトリでは fairseq を利用した実装となっていますが、fairseq は自然言語処理などに特化した便利なライブラリです。ただし、複雑でカスタム性に欠けるため、本リポジトリでは本家 fairseq のプログラムからモデル部分を抜き出し、シンプルな PyTorch のモデルとして再実装しました。

## 5. 本家 fairseq の事前学習済みチェックポイントの読み込みと、PyTorch チェックポイントへの変換

`load_from_original.py`を実行することで、`avhubert.py`を利用して fairseq のチェックポイントを読み込むことができます。また、一度読み込んでしまえば、そのモデルのパラメータを PyTorch のチェックポイントとして保存し直すことが可能です。fairseq のチェックポイントをそのまま利用すると、ハイパーパラメータ管理として用いられる hydra と競合するため、PyTorch のチェックポイントとして保存し直すことを推奨します。

## 6. hydra の config 設定

5.で PyTorch のチェックポイントを保存した後、次は hydra を使用してハイパーパラメータを設定します。`conf`ディレクトリ下に`base.yaml`と`large.yaml`のファイルがあり、それぞれ AV-HuBERT のモデルサイズに対応しています。基本的に設定する必要があるのは`ckpt_path`のみです。ここに PyTorch のチェックポイントのパスを指定してください。

また、`load_pretrained_weight`というパラメータにより、事前学習済みモデルを読み込むか否かを切り替えられるようにしています。7.で試してみてください。

## 7. PyTorch のチェックポイントの読み込み

`load_from_torch.py`を使用して、5.で保存した PyTorch のチェックポイントを読み込むプログラムを実装しています。こちらを試してみてください。
