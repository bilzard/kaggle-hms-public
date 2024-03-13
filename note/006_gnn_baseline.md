# GNNベースラインの実装について

現状、1d/2dの複合モデルよりも、1d/2dを独立して最適化したモデルのensembleが強いので1d/2d以外にもう一つベースラインを作っておきたい。

以下ではGNNベースラインの実装方法について議論する。

## 考えられる実装

### GNNの入力特徴の抽出

#### filter bankを直接入力特徴とする

EEG-GNNの論文(Reference の[5])の実装ではfilter bankをconcatしているが、40Hzの信号を1/16でSTFTした64x128の特徴を入力とすると128 x 64 = **8192**と次元がかなり大きくなってしまう。

ちなみに、中央の32frame(12.8sec)のみだと32x64=**2048**。ただ、filter bankの時系列をconcatすること自体には正当な根拠はなく、もう少し洗練された特徴抽出をしたい。

#### filter bankを2D CNNでエンコードする

`(d ch b) c f t`の画像とみなして2d-cnnでエンコードし、適当にプールすると、`(d ch b) c`の形状のチャネルごとのembeddingが得られる。

T=128を丸ごと処理させるとそれなりに時間がかかり、また2Dモデルとほぼ同じなので
中央の32 frame(12.8sec)を2D cnnでエンコードする方式が良いかもしれない。

### GNNの実装

GNNを真面目に実装しても良いが、TransformerでGNNの振る舞いを擬似的に実現することを目指す。

ノードとエッジの数が大量に(~1000以上)ある場合は計算量的な問題でGNNを使うしかないが、高々20チャネルであればTransformerでも擬似的な振る舞いを実現できると考えられるためである。

#### グラフ表現のパターンについて

EEGデータおけるprobe間のadjacencyの表現には以下のようなパターンがある。

1. probeどうしの幾何学的な距離の近さ
2. embedding同士の正規化ガウス距離
3. embedding同士のcorrelation

EEG+GNNの論文を見ていると、個々のモデルよりもこれらのアンサンブルが強そうに見える。

#### Graph-convのtransformerによる代替

transformerでGNNのadjacencyに相当する概念は、dot-product attentionがある。
隣接ノード以外の信号をmaskすれば1が擬似的に実現できる。
2, 3はdot-product attentionでq, kの代わりに入力信号xそのものを代入したもので代替できる。
