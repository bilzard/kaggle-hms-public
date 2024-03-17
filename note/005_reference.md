# 参考文献リスト

## 1D signalの特徴抽出手法

- **SincNet**[14]: band-pass filterのパラメータを学習することでconv filterを学習するよりも少ないパラメータでモデル化しようとする手法。話者識別のタスクにおいてフーリエ変換系のfilter bankであるMFCC特徴やpure-CNNのfilter bankと比較して優れた性能を報告している。
- **Wavegram(PANNs)**[15]: 1D-CNNで32kHzの音声信号を`1/(5*4*4*4)=1/320`にdown sampleし、できた100Hzのfeature mapを`B (C // F) F T`の形状にreshapeすることで学習可能なfilter bankを得る手法。

## NLL(Noisy Label Learning)

- **Co-Teaching**[1]: 2つのネットワークにより相互にnoisy sampleの情報を教え合う。バッチサンプルのうち、相手のネットワークにおけるラベルとのlossが大きいもの上位1-Rをnoisy labelとして学習サンプルから除外する。
- **Co-Teaching+**[2]: Co-Teachingにおいて、2つのネットワークの独立性を維持するため、双方の予測クラスが異なるサンプルのみを学習サンプルとして利用する。
- **JoCoR**[3]: Co-Teachingと異なり、2つのネットワークを同じloss関数をもとに同時に学習する。また、sample selectionのさい、Co-Teaching+とは逆に、双方のネットワークの予測の確率分布が近いものを優先する。これは「clean labelの予測値の分布は教師ごとのばらつきが小さい」という原則に基づく。

## KD(Knowledge Distillation)

- **CA-MKD**[4]: 複数の教師ネットワークによるKnowledge Distillation(KD)タスクで、教師の予測値のconfidenceをもとに重みづけ平均をとる。また、生徒ネットーワークのembeddingを教師ネットワークのembeddingに近づけるような修飾lossを追加。この手法では「生徒のembeddingを教師のheadに入力した場合のloss」をembeddingの蒸留ラベルの重みとして用いていて、単純に教師の予測値と正解ラベルとのlossを重みとするよりも精度が高かったとのこと。「多少embeddingがずれていても正確な予測ができる教師」を選びやすいはずなので、正確性に加えてロバストさも加味した選別指標と言える。

## GNN(Graph Neural Network)

- **EEG-GNN**[5]
- **Dist/Corr-DCRNN**[6] (c.f. DCRNN[7])

## SSL(Semi-Supervised Leaning)

- **EMAN**[8]: teacher networkのBN(Batch Norm)レイヤの統計を統計を、student networkのBNレイヤの統計のEMA(Exponential Moving Average)を使って計算するといるシンプルな手法。contrastive learning系のSOTA手法でnorm layerをEMANに置換するだけでパフォーマンスの改善を報告している。
- **$\Pi$-model / temporal ensembling**: labelがないサンプルに対してpseudo labelを割り当て、生徒networkの予測値とpseudo labelとのcontrastive lossを補助的な損失として加える。pseudo labelは生徒の予測値のEMA(Exponential Moving Average)で与える。本手法は教師ラベルの作成に新たなネットワークを必要とせず、mini-batchあたりforward pass1回分ですむ。
- **Mean Teacher**[9]: templral ensemblingがモデルの予測値に対してEMAをとったのに対し、本手法はモデルのパラメータに関してEMAをとる。このような手法をとる動機は以下: trainingサンプルは1epochあたり1度しか見ないため、*教師の予測値がepoch単位でしか更新されない*。特に学習の初期段階において生徒ネットワークのパラメータの更新はペースが早いのに対し、pseudo labelの更新頻度が遅すぎるため、表現学習の速度も遅いことが想定される。これに対し、提案手法はパラメータ空間でEMAが行われるため、mini-batch単位で教師の予測を更新することができる。
- **MoCo**[10]: NLPタスクにヒントを得て、画像分類モデルにおけるラベルなしの事前学習方法を提案するもの。Mean teacherと同様、教師ネットワークには生徒モデルの重みのEMAを採用するが、contrastive lossを評価するサンプルをバッチ内からサンプルすると、バッチサイズによって負例の数が制限されてしまう。これに対して提案手法は*直近K個のmini-batchにおける教師ネットワークのembeddingをキャッシュする*ことで、mini-batch外のサンプルも負例として利用する。Ablation studyの結果では負例の数を増やすほどダウンストリームタスクでのパフォーマンスが上がることを報告している。
- **SimCLR**[12]: この手法は教師ラベルとして、*異なる種類のdata augmentationを適用したサンプルの生徒networkのembedding*を用いる。この手法では1つのmini-batchあたり$2(N-1)$ 個の負例を得ることができる。MoCoとは異なり、複数GPUで分散学習させ、全体のbatchサイズを大きくとることで負例の数を増やす戦略をとる。
- **BYOL**[11]: 教師と生徒のde-couplingをさらに推し進め、負例なしで既存のSSL手法を凌駕した手法(通常は負例を入力しないと全てのサンプルのembeddingが1点に収束するmode collapseと呼ばれる現象が起こる)。教師ネットワークとして生徒のEMAをとり、かつ生徒と教師で異なるaugmentationを適用したサンプルを入力する。mini-batchあたりのforward prop数は教師、生徒の伝搬に加え、両者に入力するサンプルを入れ替えた計4回発生する。
- **FixMatch**[13]: Contrastive lossの代わりに教師networkの予測をPseudo labelとして用いる手法。教師の予測のうち、confidenceが閾値以上のものをhard labelとして用いる。

上記をまとめると表1のようになる。

**表1: 各種SSL手法の特徴について**

| Task | Method          | How teacher label is provided?         | Has Teacher Networks? | How teacher is defined?             | How negative samples are sampled? | number of forward prop. / mini-batch | Constrastive Loss Funciton   |
|------|-----------------|----------------------------------------|-----------------------|-------------------------------------|-----------------------------------|------------------------------------|-----------------------------|
| SSL  | temporal ensembling | EMA of student's predictions         | -                     | -                                   | -                              | 1                                  | MSE                         |
| SSL  | Mean Teacher    | Teacher's predictions                   | ✔️                    | EMA of student's parameters        | -                              | 2                                  | MSE                         |
| SSL  | FixMatch        | Pseudo Label(=hard labels) by Teacher's predictions | ✔️ | EMA of student's parameters        | -                              | 2                                  | Cross Entropy(student, PL)  |
| UL   | MoCo            | Teacher's predictions                   | ✔️                    | EMA of student's parameters        | from last K mini-batches          | 2                                  | Info-NCE                    |
| UL   | SimCLR          | Student's prediction over differenet aug. views | - | -                                   | from current mini-batch           | 2                                  | NT-Xent                     |
| UL   | BYOL            | Teacher's prediction over differenet aug. view | ✔️ | EMA of student's parameters        | -                              | 4                                  | NT-Xent (w/o negative loss) |


### 補足

#### 教師networkにおけるBatch Normレイヤのrunning statについて

Student-Teacherによる対照学習では多くの場合teacherとしてstudentの重みのEMA(Exponential Moving Average)が使われる場合が多いが、teacher networkのBatch Normalizationレイヤーの統計については直感的な計算方法が存在しない（通常、学習終了後にサンプルの一部を使ってteacher networkのbatchの統計を計算し直す）。また、batch normレイヤーはbatch内のサンプル間に相関ができることが知られていて、*若干リークする*。このため、既存のcontrastive learning手法ではBNの代わりにSyncBN/Shuffle BNといった、BNの統計への依存性を緩和した代替のブロックが用いられる。しかしながら、これらの代替手法は複数のGPUの利用を前提としているため計算コストが高い。これに対し、EMANは*BNの統計もstudentのbnの統計のEMAで与える*というシンプルな解決方法をとる。単純な方法ながら、既存のcontrastive learning手法におけるnorm layerをEMANに置換しただけで、一般的なデータセットにおけるタスクのパフォーマンスの改善が報告されている。

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
- [14] [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158)
- [15] [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211)
- [16] [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242)