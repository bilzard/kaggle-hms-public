# 参考文献リスト

## NLL(Noisy Label Learning)

- **Co-Teaching**[1]: 2つのネットワークにより相互にnoisy sampleの情報を教え合う。バッチサンプルのうち、相手のネットワークにおけるラベルとのlossが大きいもの上位1-Rをnoisy labelとして学習サンプルから除外する。
- **Co-Teaching+**[2]: Co-Teachingにおいて、2つのネットワークの独立性を維持するため、双方の予測クラスが異なるサンプルのみを学習サンプルとして利用する。
- **JoCoR**[3]: Co-Teachingと異なり、2つのネットワークを同じloss関数をもとに同時に学習する。また、sample selectionのさい、Co-Teaching+とは逆に、双方のネットワークの予測の確率分布が近いものを優先する。これは「clean labelの予測値の分布は教師ごとのばらつきが小さい」という原則に基づく。

## KD(Knowledge Distillation)

- **CA-MKD**[4]: 複数の教師ネットワークによるKnowledge Distillation(KD)タスクで、教師の予測値のconfidenceをもとに重みづけ平均をとる。また、生徒ネットーワークのembeddingを教師ネットワークのembeddingに近づけるような修飾lossを追加。この手法では「生徒のembeddingを教師のheadに入力した場合のloss」をembeddingの蒸留ラベルの重みとして用いていて、単純に教師の予測値と正解ラベルとのlossを重みとするよりも精度が高かったとのこと。「多少embeddingがずれていても正確な予測ができる教師」を選びやすいはずなので、正確性に加えてロバストさも加味した選別指標と言える。

## GNN(Graph Neural Network)

- **EEG-GNN**[5]
- **Dist/Corr-DCRNN**[6] (c.f. DCRNN[7])

## Contrastive Learning

- **EMAN**[8]: teacher networkのBN(Batch Norm)レイヤの統計を統計を、student networkのBNレイヤの統計のEMA(Exponential Moving Average)を使って計算するといるシンプルな手法。contrastive learning系のSOTA手法でnorm layerをEMANに置換するだけでパフォーマンスの改善を報告している。
- **Mean Teacher**[9]
- **MoCo**[10]
- **BYOL**[11]
- **SimCLR**[12]
- **FixMatch**[13]

### 詳細な説明

#### EMAN

Student-Teacherによる対照学習では多くの場合teacherとしてstudentの重みのEMA(Exponential Moving Average)が使われる場合が多いが、teacher networkのBatch Normalizationレイヤーの統計については直感的な計算方法が存在しない（通常、学習終了後にサンプルの一部を使ってteacher networkのbatchの統計を計算し直す）。また、batch normレイヤーはbatch内のサンプル間に相関ができることが知られていて、*若干リークする*。このため、既存のcontrastive learning手法ではBNの代わりにSyncBN/Shuffle BNといった、BNの統計への依存性を緩和した代替のブロックが用いられる。しかしながら、これらの代替手法は複数のGPUの利用を前提としているため計算コストが高い。これに対し、提案手法は*BNの統計もstudentのbnの統計のEMAで与える*というシンプルな解決方法をとる。単純な方法ながら、既存のcontrastive learning手法におけるnorm layerをEMANに置換しただけで、一般的なデータセットにおけるタスクのパフォーマンスの改善が報告されている。

## Reference

- [1] [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/abs/1804.06872)
- [2] [How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215)
- [3] [Combating noisy labels by agreement: A joint training method with co-regularization](https://arxiv.org/abs/2003.02752)
- [4] [Confidence-Aware Multi-Teacher Knowledge Distillation](https://arxiv.org/abs/2201.00007)
- [5] [EEG-GNN: Graph Neural Networks for Classification of Electroencephalogram (EEG) Signals](https://arxiv.org/abs/2106.09135)
- [6] [Self-Supervised Graph Neural Networks for Improved Electroencephalographic Seizure Analysis](https://arxiv.org/abs/2104.08336)
- [7] [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926)
- [8] [Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Exponential_Moving_Average_Normalization_for_Self-Supervised_and_Semi-Supervised_Learning_CVPR_2021_paper.pdf)
- [9] [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780)
- [10] [Momentum Contrast for Unsupervised Visual Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)
- [11] [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)
- [12] [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [13] [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)