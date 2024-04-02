# CVのチューニング

## 目的

1. testにlow-qualityが少量混ざっている前提でnelder-meadによりweightを決定し、既存の方法（high-qualityのみ）で決定したweightと比較してCVとLBにどの程度影響が出るのか確認する
1. testはhigh-qualityのみで、各seizure typeの確率分布のみtrainと乖離している前提で、class weightを調整し、既存の方法（high-qualityのみ）で決定したweightと比較してCVとLBにどの程度影響が出るのか確認する
1. EEGごとのweightやGTの集約方法の違いによる影響をみる

## 手法

CVの目的関数にweightを設定する

$$
\mathcal{L}_\text{kl-div} = \frac{\sum_i \sum_c y_{i,c}\log(y_{i,c} / \tilde{y}_{i, c}) \cdot w_c \cdot w_i}{\sum_c w_c \sum_i w_i}
$$

* $w_i$: sample weight: vote数に基づくサンプルごとのweight
* $w_c$: seizure typeに基づくweight

## 結果

結果を表1にまとめる。

**表1: 目的関数を変えてnelder-meadで最適化した場合のCVへの影響**

| GT            | lb | ub | n_vote < lb | lb <= n_vote < ub | ub <= n_vote | class_weight                        | CV(old_weight) | CV(new_weight) | diff    |
|---------------|----|----|-------------|-------------------|--------------|-------------------------------------|----------------|----------------|---------|
| mean          | 3  | 10 | 0           | 0                 | 1            | 1,1,1,1,1,1                         | 0.1947         | 0.1943         | +0.000  |
| mean          | 3  | 10 | 0           | 0.2               | 0.8          | 1,1,1,1,1,1                         | 0.3565         | 0.3472         | +0.009  |
| mean          | 4  | 10 | 0           | 0.2               | 0.8          | 1,1,1,1,1,1                         | 0.2107         | 0.2098         | +0.001  |
| mean          | 5  | 10 | 0           | 0.2               | 0.8          | 1,1,1,1,1,1                         | 0.2024         | 0.2018         | +0.001  |
| mean          | 4  | 9  | 0           | 0.2               | 0.8          | 1,1,1,1,1,1                         | 0.2112         | 0.2104         | +0.001  |
| mean          | 4  | 8  | 0           | 0.2               | 0.8          | 1,1,1,1,1,1                         | 0.2121         | 0.2113         | +0.001  |
| mean          | 4  | 9  | 0           | 0                 | 1            | 1,1,1,1,1,1                         | 0.1957         | 0.1953         | +0.000  |
| weighted mean | 4  | 9  | 0           | 0                 | 1            | 1,1,1,1,1,1                         | 0.1998         | 0.1996         | +0.000  |
| max           | 4  | 9  | 0           | 0                 | 1            | 1,1,1,1,1,1                         | 0.2035         | 0.2034         | +0.000  |
| mean          | 4  | 9  | 0           | 0                 | 1            | 1.33, 1.0, 1.0, 1.36, 1.38, 0.77 | 0.1957         | 0.1945         | +0.001  |

### 1. low-qualityが少量混ざっているケース

trainのlow-quality(3 <= n_votes < 10)が20%CVに混ざっていたケース(2行目)では、CVの目的関数の変更により0.009程度の無視できない影響が確認された。一方で、CVのレンジがLBのレンジと0.1以上乖離しており、「testにlow-qualityが少量混ざっている」仮説については、少なくともpublic testデータに関しては可能性は低いと考えられる（もしtestにtrainのlow-qualityと同等のデータが少量混ざっているなら、LBのレンジはもっと大きな値をとるはず(0.3xなど)）。

### 2. high-qualityのみでseizure typeごとの確率分布のみ異なるケース

LB probingにより推定されたtestにおけるラベル分布とtrainのラベル分布を比較し、class weightを決定した(表2)。このclass weightを使用した目的関数によって決定したmodel weightは、class weight=1で決定したmodel weightと比較してCVに0.001程度の影響があることがわかった（表1の最後の行）。

**表2: train/testのラベル分布により決定したclass weight**

| class  | p_oof_hq | p_test | p_test/p_oof_hq |
|--------|----------|--------|-----------------|
| sizure | 0.105    | 0.14   | 1.33            |
| lpd    | 0.163    | 0.163  | 1.00            |
| gpd    | 0.12     | 0.119  | 0.99            |
| lrda   | 0.069    | 0.094  | 1.36            |
| grda   | 0.108    | 0.149  | 1.38            |
| other  | 0.435    | 0.335  | 0.77            |


### 3. EEGごとのweight/GT labelの集約方法の違いによる影響

EEGごとのラベルの集約方法を変えた時のCVに与える影響は無視できる程度だった（表1、8-9行目）。

## まとめ

1. low-qualityが少数混ざっていると仮定した場合、LBで観測されるレンジとCVのレンジに大きな乖離（0.01以上）が生じるため、この可能性は低いと考えられる。
1. testにおけるseizure typeごとの確率分布がtrainと乖離していることがLB Probingよりわかっている。この影響を加味し、trainのクラスの分布をtestの推定分布に近づけるように調整したclass weightを適用したところ、CVに0.001程度のわずかな影響が確認された。
1. EEGごとのlabel/weightの集約方法の違いは、少なくともnelder-meadによるweight決定には大きな影響を及ぼさない。

## この後のアクション

1. class weightを調整した目的関数を用いてnelder-mead方で決定したmodel weightでsubmitし、LBに与える影響を確認する。

## 参考

- [1] 実験に使用したnotebook(local): https://github.com/bilzard/kaggle-hms-bilzard/blob/main/notebook/local_cv.ipynb
- [2] [1]をKaggle Notebook化したバージョン https://www.kaggle.com/code/tatamikenn/hms-cv-nelder-mead-v2/notebook
